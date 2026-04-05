"""Databricks Vector Search retriever backend.

Uses managed embeddings (Databricks computes embeddings from the ``text`` column)
and supports hybrid search + metadata filters for IPC/BNS section lookups.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from nyaya_dhwani.keyword_boost import detect_section_references

logger = logging.getLogger(__name__)

_RESULT_COLUMNS = ["chunk_id", "text", "title", "source", "doc_type"]


class VectorSearchRetriever:
    """Query a Databricks Vector Search index via the SDK."""

    def __init__(self, endpoint_name: str, index_name: str) -> None:
        self._endpoint_name = endpoint_name
        self._index_name = index_name
        self._index = None

    def _get_client(self):
        """Return the vector_search_indexes API client (not the index object)."""
        if self._index is not None:
            return self._index
        try:
            from databricks.sdk import WorkspaceClient  # type: ignore[import-not-found]

            self._index = WorkspaceClient().vector_search_indexes
            logger.info(
                "VectorSearchRetriever: connected to endpoint %s, index %s",
                self._endpoint_name,
                self._index_name,
            )
            return self._index
        except Exception:
            logger.exception(
                "VectorSearchRetriever: failed to initialize client for endpoint=%s index=%s. "
                "Check credentials and CAN_QUERY permissions.",
                self._endpoint_name,
                self._index_name,
            )
            raise

    def search(self, query: str, k: int = 7) -> pd.DataFrame:
        """Similarity search with optional metadata filters for section references."""
        client = self._get_client()
        index_name = self._index_name

        refs = detect_section_references(query)
        filters = {"doc_type": "law_mapping"} if refs else None

        try:
            mapping_rows = []
            if filters:
                try:
                    mapping_resp = client.query_index(
                        index_name=index_name,
                        columns=_RESULT_COLUMNS,
                        query_text=query,
                        num_results=3,
                        filters_json=json.dumps(filters),
                    )
                    mapping_rows = _response_to_rows(mapping_resp)
                except Exception:
                    logger.warning(
                        "Filtered VS query failed for endpoint=%s index=%s; continuing unfiltered",
                        self._endpoint_name,
                        self._index_name,
                        exc_info=True,
                    )

            resp = client.query_index(
                index_name=index_name,
                columns=_RESULT_COLUMNS,
                query_text=query,
                num_results=k,
            )
            main_rows = _response_to_rows(resp)

            seen_ids = set()
            combined = []
            for row in mapping_rows + main_rows:
                cid = row.get("chunk_id")
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                combined.append(row)

            combined = combined[:k]
            for i, row in enumerate(combined):
                row["rank"] = i

            if not combined:
                return pd.DataFrame(columns=_RESULT_COLUMNS + ["score", "rank"])
            return pd.DataFrame(combined)

        except Exception:
            logger.warning(
                "VectorSearchRetriever.search failed for endpoint=%s index=%s query=%r",
                self._endpoint_name,
                self._index_name,
                (query or "")[:120],
                exc_info=True,
            )
            return pd.DataFrame(columns=_RESULT_COLUMNS + ["score", "rank"])


def _response_to_rows(resp) -> list[dict]:
    """Convert a VS similarity_search/query_index response to a list of dicts."""
    rows = []
    if hasattr(resp, "as_dict"):
        resp = resp.as_dict()
    try:
        manifest = resp.get("manifest", {}) if isinstance(resp, dict) else getattr(resp, "manifest", {})
        result = resp.get("result", {}) if isinstance(resp, dict) else getattr(resp, "result", {})

        if isinstance(manifest, dict):
            col_names = [c.get("name", f"col_{i}") for i, c in enumerate(manifest.get("columns", []))]
        else:
            col_names = [c.name for c in (manifest.columns or [])]

        data_array = result.get("data_array", []) if isinstance(result, dict) else (result.data_array or [])

        for row_data in data_array:
            row = dict(zip(col_names, row_data))
            if "score" not in row:
                row["score"] = row.pop(col_names[-1], 0.0) if len(col_names) > len(_RESULT_COLUMNS) else 0.0
            rows.append(row)
    except Exception:
        logger.warning("Failed to parse VS response", exc_info=True)
    return rows
