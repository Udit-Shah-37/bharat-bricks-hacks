"""Hybrid retriever: Dense (FAISS/VS) + BM25 keyword search + RRF fusion.

Reciprocal Rank Fusion (RRF) combines results from multiple retrieval
strategies using: score = Σ 1/(k + rank_i) with k=60.

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

# ---------------------------------------------------------------------------
# BM25 keyword search (lightweight, no external deps)
# ---------------------------------------------------------------------------

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
    """Simple whitespace + lowercase tokenizer."""
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP_WORDS and len(w) > 1]


class BM25:
    """Minimal BM25 implementation over a list of documents (strings).

    Okapi BM25 with standard parameters k1=1.5, b=0.75.
    """

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
        """Return list of (doc_index, score) sorted by score desc."""
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

        # Top-k by score
        indexed = [(i, s) for i, s in enumerate(scores) if s > 0]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:k]


# ---------------------------------------------------------------------------
# Cross-reference expansion
# ---------------------------------------------------------------------------

# Patterns to detect section/article references in retrieved text
_XREF_PATTERNS = [
    re.compile(r"\b(?:Section|Sec\.?)\s+(\d+[A-Z]?(?:\(\d+\))?)", re.I),
    re.compile(r"\bArticle\s+(\d+[A-Z]?)", re.I),
    re.compile(r"\bBNS\s+(?:Section\s+)?(\d+[A-Z]?(?:\(\d+\))?)", re.I),
]


def extract_cross_references(texts: list[str], limit: int = 5) -> list[str]:
    """Extract section/article numbers referenced in retrieved chunks.

    Returns search queries like "BNS Section 42" or "Article 21".
    """
    refs: Counter = Counter()
    for text in texts:
        for pat in _XREF_PATTERNS:
            for m in pat.finditer(text):
                ref_num = m.group(1)
                full = m.group(0)
                refs[full] += 1

    # Return the most-referenced items (that we might not already have)
    return [ref for ref, _ in refs.most_common(limit)]


def expand_with_cross_refs(
    base_results: pd.DataFrame,
    all_chunks: pd.DataFrame,
    max_expansion: int = 3,
) -> pd.DataFrame:
    """Look up cross-referenced sections in the full chunk corpus.

    If retrieved text mentions "Section 2(6)", find that chunk in all_chunks
    and add it to results.
    """
    if all_chunks is None or all_chunks.empty:
        return base_results

    texts = base_results["text"].tolist() if "text" in base_results.columns else []
    if not texts:
        return base_results

    xrefs = extract_cross_references(texts, limit=max_expansion + 2)
    if not xrefs:
        return base_results

    existing_ids = set()
    if "chunk_id" in base_results.columns:
        existing_ids = set(base_results["chunk_id"].tolist())

    expansion_rows = []
    for ref_text in xrefs:
        if len(expansion_rows) >= max_expansion:
            break
        # Search chunk titles and text for this reference
        mask = all_chunks["title"].str.contains(re.escape(ref_text), case=False, na=False, regex=True) | \
               all_chunks["text"].str.contains(re.escape(ref_text), case=False, na=False, regex=True)
        hits = all_chunks[mask]
        for _, row in hits.head(1).iterrows():
            cid = row.get("chunk_id", "")
            if cid and cid not in existing_ids:
                existing_ids.add(cid)
                r = row.to_dict()
                r["score"] = 0.5  # synthetic score for expanded refs
                r["rank"] = len(base_results) + len(expansion_rows)
                expansion_rows.append(r)

    if expansion_rows:
        logger.info("Cross-reference expansion added %d chunks", len(expansion_rows))
        return pd.concat([base_results, pd.DataFrame(expansion_rows)], ignore_index=True)

    return base_results


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    *rankings: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Combine multiple ranked lists using RRF.

    Each ranking is [(doc_index, score), ...] sorted by score desc.
    Returns fused [(doc_index, rrf_score)] sorted by rrf_score desc.

    RRF formula: score(d) = Σ 1/(k + rank_i) for each ranking i
    """
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_idx, _) in enumerate(ranking):
            fused[doc_idx] += 1.0 / (k + rank + 1)  # rank is 0-based, +1 for 1-based

    result = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return result


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Combines dense search (FAISS) + BM25 keyword search with RRF fusion.

    Also performs cross-reference expansion on the final results.
    """

    def __init__(self, faiss_retriever, all_chunks: pd.DataFrame | None = None) -> None:
        """
        Args:
            faiss_retriever: A FaissRetriever instance (provides dense search + keyword boost)
            all_chunks: Full chunks DataFrame for BM25 and cross-ref expansion.
                        If None, will load from the FAISS index on first search.
        """
        self._faiss = faiss_retriever
        self._all_chunks = all_chunks
        self._bm25: BM25 | None = None

    def _ensure_loaded(self) -> None:
        """Ensure FAISS is loaded and BM25 index is built."""
        self._faiss._load()
        if self._all_chunks is None:
            self._all_chunks = self._faiss._ci.chunks
        if self._bm25 is None and self._all_chunks is not None and not self._all_chunks.empty:
            texts = self._all_chunks["text"].fillna("").tolist()
            self._bm25 = BM25().fit(texts)
            logger.info("BM25 index built over %d chunks", len(texts))

    def search(self, query: str, k: int = 7) -> pd.DataFrame:
        """Hybrid search: dense + BM25 fused with RRF, then cross-ref expansion."""
        self._ensure_loaded()
        assert self._all_chunks is not None

        # 1. Dense search (FAISS with keyword boost) — get more candidates
        dense_k = min(k * 2, 20)
        dense_df = self._faiss.search(query, k=dense_k)

        # Build dense ranking as [(chunk_row_idx, score)]
        dense_ranking = []
        for _, row in dense_df.iterrows():
            # Find the index in all_chunks
            cid = row.get("chunk_id")
            if cid is not None and "chunk_id" in self._all_chunks.columns:
                matches = self._all_chunks.index[self._all_chunks["chunk_id"] == cid].tolist()
                if matches:
                    dense_ranking.append((matches[0], float(row.get("score", 0))))

        # 2. BM25 keyword search
        bm25_ranking = []
        if self._bm25 is not None:
            bm25_ranking = self._bm25.query(query, k=dense_k)

        # 3. RRF fusion
        if bm25_ranking:
            fused = reciprocal_rank_fusion(dense_ranking, bm25_ranking, k=60)
        else:
            fused = dense_ranking

        # 4. Build result DataFrame from fused ranking
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

        # 5. Cross-reference expansion
        result_df = expand_with_cross_refs(result_df, self._all_chunks, max_expansion=2)

        return result_df.head(k).reset_index(drop=True)
