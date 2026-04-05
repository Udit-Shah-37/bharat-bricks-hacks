"""Hybrid retriever: Dense (FAISS/VS) + BM25 keyword search + RRF fusion.

Reciprocal Rank Fusion (RRF) combines results from multiple retrieval
strategies using: score = sum(1 / (k + rank_i)) with k=60.

Also includes cross-reference expansion: if a retrieved chunk mentions
"Section 2(6)" or "as defined in Article 21", fetch those referenced
chunks too.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict

import pandas as pd

logger = logging.getLogger(__name__)

_STOP_WORDS = frozenset(
    "a an the is was are were be been being have has had do does did "
    "will would shall should may might can could of in to for on with "
    "at by from as into through during before after above below between "
    "out off over under and but or nor not so yet both either neither "
    "each every all any few more most other some such no nor too very "
    "what which who whom this that these those i me my myself we our "
    "ours he him his she her it its they them their".split()
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP_WORDS and len(w) > 1]


class BM25:
    """Minimal BM25 implementation over a list of documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_freqs: dict[str, int] = {}
        self._doc_lens: list[int] = []
        self._avg_dl: float = 0.0
        self._n: int = 0
        self._token_lists: list[list[str]] = []

    def fit(self, documents: list[str]) -> "BM25":
        self._n = len(documents)
        self._token_lists = [_tokenize(d) for d in documents]
        self._doc_lens = [len(t) for t in self._token_lists]
        self._avg_dl = sum(self._doc_lens) / max(self._n, 1)

        df: dict[str, int] = defaultdict(int)
        for tokens in self._token_lists:
            for term in set(tokens):
                df[term] += 1
        self._doc_freqs = dict(df)
        return self

    def query(self, q: str, k: int = 10) -> list[tuple[int, float]]:
        q_tokens = _tokenize(q)
        if not q_tokens:
            return []

        scores: list[float] = [0.0] * self._n
        for term in q_tokens:
            if term not in self._doc_freqs:
                continue
            df = self._doc_freqs[term]
            idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)
            for i, tokens in enumerate(self._token_lists):
                tf = tokens.count(term)
                if tf == 0:
                    continue
                dl = self._doc_lens[i]
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / max(self._avg_dl, 1))
                scores[i] += idf * num / den

        indexed = [(i, s) for i, s in enumerate(scores) if s > 0]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:k]


_XREF_PATTERNS = [
    re.compile(r"\b(?:Section|Sec\.?)\s+(\d+[A-Z]?(?:\(\d+\))?)", re.I),
    re.compile(r"\bArticle\s+(\d+[A-Z]?)", re.I),
    re.compile(r"\bBNS\s+(?:Section\s+)?(\d+[A-Z]?(?:\(\d+\))?)", re.I),
]


def extract_cross_references(texts: list[str], limit: int = 5) -> list[str]:
    refs: Counter[str] = Counter()
    for text in texts:
        for pat in _XREF_PATTERNS:
            for m in pat.finditer(text):
                refs[m.group(0)] += 1
    return [ref for ref, _ in refs.most_common(limit)]


def expand_with_cross_refs(
    base_results: pd.DataFrame,
    all_chunks: pd.DataFrame,
    max_expansion: int = 3,
) -> pd.DataFrame:
    if all_chunks is None or all_chunks.empty:
        return base_results

    texts = base_results["text"].tolist() if "text" in base_results.columns else []
    if not texts:
        return base_results

    xrefs = extract_cross_references(texts, limit=max_expansion + 2)
    if not xrefs:
        return base_results

    existing_ids = set(base_results["chunk_id"].tolist()) if "chunk_id" in base_results.columns else set()

    expansion_rows = []
    for ref_text in xrefs:
        if len(expansion_rows) >= max_expansion:
            break
        mask = all_chunks["title"].str.contains(re.escape(ref_text), case=False, na=False, regex=True) | \
               all_chunks["text"].str.contains(re.escape(ref_text), case=False, na=False, regex=True)
        hits = all_chunks[mask]
        for _, row in hits.head(1).iterrows():
            cid = row.get("chunk_id", "")
            if cid and cid not in existing_ids:
                existing_ids.add(cid)
                r = row.to_dict()
                r["score"] = 0.5
                r["rank"] = len(base_results) + len(expansion_rows)
                expansion_rows.append(r)

    if expansion_rows:
        logger.info("Cross-reference expansion added %d chunks", len(expansion_rows))
        return pd.concat([base_results, pd.DataFrame(expansion_rows)], ignore_index=True)

    return base_results


def reciprocal_rank_fusion(*rankings: list[tuple[int, float]], k: int = 60) -> list[tuple[int, float]]:
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_idx, _) in enumerate(ranking):
            fused[doc_idx] += 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """Combines dense search + BM25 with RRF and cross-reference expansion."""

    def __init__(self, faiss_retriever, all_chunks: pd.DataFrame | None = None) -> None:
        self._faiss = faiss_retriever
        self._all_chunks = all_chunks
        self._bm25: BM25 | None = None

    def _ensure_loaded(self) -> None:
        self._faiss._load()
        if self._all_chunks is None:
            self._all_chunks = self._faiss._ci.chunks
        if self._bm25 is None and self._all_chunks is not None and not self._all_chunks.empty:
            texts = self._all_chunks["text"].fillna("").tolist()
            self._bm25 = BM25().fit(texts)
            logger.info("BM25 index built over %d chunks", len(texts))

    def search(self, query: str, k: int = 7) -> pd.DataFrame:
        self._ensure_loaded()
        assert self._all_chunks is not None

        dense_k = min(k * 2, 20)
        dense_df = self._faiss.search(query, k=dense_k)

        dense_ranking = []
        for _, row in dense_df.iterrows():
            cid = row.get("chunk_id")
            if cid is not None and "chunk_id" in self._all_chunks.columns:
                matches = self._all_chunks.index[self._all_chunks["chunk_id"] == cid].tolist()
                if matches:
                    dense_ranking.append((matches[0], float(row.get("score", 0))))

        bm25_ranking = self._bm25.query(query, k=dense_k) if self._bm25 is not None else []
        fused = reciprocal_rank_fusion(dense_ranking, bm25_ranking, k=60) if bm25_ranking else dense_ranking

        rows = []
        seen = set()
        for doc_idx, rrf_score in fused[:k]:
            if doc_idx in seen or doc_idx >= len(self._all_chunks):
                continue
            seen.add(doc_idx)
            r = self._all_chunks.iloc[doc_idx].to_dict()
            r["score"] = rrf_score
            r["rank"] = len(rows)
            rows.append(r)

        if not rows:
            return dense_df.head(k)

        result_df = pd.DataFrame(rows)
        result_df = expand_with_cross_refs(result_df, self._all_chunks, max_expansion=2)
        return result_df.head(k).reset_index(drop=True)
