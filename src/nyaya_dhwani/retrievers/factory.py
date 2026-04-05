"""Unified retriever interface and backend factory."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from nyaya_dhwani.retrievers.faiss_index import CorpusIndex

logger = logging.getLogger(__name__)


@runtime_checkable
class Retriever(Protocol):
    """Uniform search interface for retrieval backends."""

    def search(self, query: str, k: int = 7) -> pd.DataFrame:
        ...


class FaissRetriever:
    """Wraps CorpusIndex + SentenceEmbedder behind the Retriever interface."""

    def __init__(self, index_dir: str | Path) -> None:
        self._index_dir = str(index_dir)
        self._ci: CorpusIndex | None = None
        self._embedder = None

    def _load(self) -> None:
        if self._ci is not None:
            return
        from nyaya_dhwani.embedder import SentenceEmbedder

        logger.info("FaissRetriever: loading index from %s", self._index_dir)
        try:
            self._ci = CorpusIndex.load(self._index_dir)
            m = self._ci.manifest
            self._embedder = SentenceEmbedder(
                model_name=m.embedding_model,
                normalize=m.normalize_embeddings,
            )
            logger.info("FaissRetriever: loaded %d vectors, model %s", m.num_vectors, m.embedding_model)
        except Exception:
            logger.exception(
                "FaissRetriever: failed to load index or embeddings from %s. "
                "Check volume/file permissions and index artifacts.",
                self._index_dir,
            )
            raise

    def search(self, query: str, k: int = 7) -> pd.DataFrame:
        try:
            self._load()
            assert self._ci is not None and self._embedder is not None
            emb = self._embedder.encode([query.strip()])
            semantic_df = self._ci.search(emb, k=k)

            from nyaya_dhwani.keyword_boost import boost_with_keywords

            return boost_with_keywords(query, semantic_df, self._ci.chunks, k=k)
        except Exception:
            logger.exception(
                "FaissRetriever.search failed (k=%s, query=%r).", k, (query or "")[:120]
            )
            raise


class FallbackRetriever:
    """Tries primary retriever, falls back on failure or empty result."""

    def __init__(self, primary: Retriever, fallback: Retriever) -> None:
        self._primary = primary
        self._fallback = fallback

    def search(self, query: str, k: int = 7) -> pd.DataFrame:
        try:
            result = self._primary.search(query, k)
            if result is not None and not result.empty:
                return result
            logger.warning("Primary retriever returned empty, falling back")
        except Exception:
            logger.warning("Primary retriever failed, falling back", exc_info=True)
        return self._fallback.search(query, k)


_LOCAL_INDEX_CACHE = "/tmp/nyaya_index"


def _download_from_volume(volume_path: str, local_dir: str) -> str:
    """Download index files from a UC Volume via the Databricks SDK."""
    local = Path(local_dir)
    if (local / "manifest.json").exists():
        logger.info("Index already cached at %s", local)
        return str(local)
    logger.info("Downloading index from Volume %s -> %s", volume_path, local)
    from databricks.sdk import WorkspaceClient  # type: ignore[import-not-found]

    try:
        w = WorkspaceClient()
        local.mkdir(parents=True, exist_ok=True)
        for item in w.files.list_directory_contents(volume_path):
            if item.is_directory:
                continue
            dest = local / item.name
            logger.info("  downloading %s", item.name)
            with w.files.download(item.path).contents as src, open(dest, "wb") as dst:
                while True:
                    chunk = src.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
    except Exception:
        logger.exception(
            "Failed to access/download index from UC Volume %s. "
            "Check READ permissions and volume path.",
            volume_path,
        )
        raise
    logger.info("Index download complete -> %s", local)
    return str(local)


def _resolve_index_dir() -> str:
    """Resolve FAISS index directory, downloading from UC Volume if needed."""
    default = "/Volumes/workspace/default/bharat_bricks_hacks/nyaya_index"
    path = os.environ.get("NYAYA_INDEX_DIR", default).strip()
    if path.startswith("/Volumes/") and not Path(path).exists():
        try:
            path = _download_from_volume(path, _LOCAL_INDEX_CACHE)
        except Exception:
            logger.warning("Could not download index from Volume; continuing with unresolved path", exc_info=True)
    if not Path(path).exists():
        logger.warning("Resolved index path does not exist: %s", path)
    return path


def get_retriever() -> Retriever:
    """Instantiate the configured retriever backend."""
    backend = os.environ.get("NYAYA_RETRIEVAL_BACKEND", "faiss").strip().lower()
    use_hybrid = os.environ.get("NYAYA_USE_HYBRID", "true").strip().lower() in ("1", "true", "yes")

    faiss_dir = _resolve_index_dir()
    faiss_ret = FaissRetriever(faiss_dir)

    if backend == "vector_search":
        endpoint = os.environ.get("NYAYA_VS_ENDPOINT_NAME", "").strip()
        index_name = os.environ.get("NYAYA_VS_INDEX_NAME", "").strip()
        if endpoint and index_name:
            try:
                from nyaya_dhwani.retrievers.vector_search import VectorSearchRetriever

                vs_ret = VectorSearchRetriever(endpoint, index_name)
                logger.info("Using VectorSearchRetriever (endpoint=%s) with FAISS fallback", endpoint)
                return FallbackRetriever(primary=vs_ret, fallback=faiss_ret)
            except Exception:
                logger.warning("Failed to init VectorSearchRetriever, using FAISS", exc_info=True)
        else:
            logger.warning(
                "NYAYA_RETRIEVAL_BACKEND=vector_search but NYAYA_VS_ENDPOINT_NAME / "
                "NYAYA_VS_INDEX_NAME not set - falling back to FAISS"
            )

    if use_hybrid:
        try:
            from nyaya_dhwani.retrievers.hybrid import HybridRetriever

            hybrid = HybridRetriever(faiss_ret)
            logger.info("Using HybridRetriever (FAISS + BM25 + RRF, index_dir=%s)", faiss_dir)
            return hybrid
        except Exception:
            logger.warning("Failed to init HybridRetriever, using plain FAISS", exc_info=True)

    logger.info("Using FaissRetriever (index_dir=%s)", faiss_dir)
    return faiss_ret
