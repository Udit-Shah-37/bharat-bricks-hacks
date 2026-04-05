"""Query logging for analytics — writes to Delta Lake on Databricks, CSV locally.

Every query + response is logged with metadata for analytics dashboards.
On Databricks: writes to Delta Lake table `query_logs`.
Locally / when Spark unavailable: writes to CSV file.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

CATALOG = os.environ.get("NYAYA_CATALOG", "workspace")
SCHEMA = os.environ.get("NYAYA_SCHEMA", "default")
TABLE = f"{CATALOG}.{SCHEMA}.query_logs"

# ---------------------------------------------------------------------------
# Section citation extraction
# ---------------------------------------------------------------------------

_CITE_RE = re.compile(
    r"\b(?:BNS|IPC)\s+(?:Section\s+)?(\d+[A-Z]?(?:\(\d+\))?)"
    r"|\bArticle\s+(\d+[A-Z]?)"
    r"|\bSection\s+(\d+[A-Z]?(?:\(\d+\))?)",
    re.I,
)


def _extract_cited_sections(response: str) -> list[str]:
    """Pull out all section/article numbers cited in the LLM response."""
    found: list[str] = []
    for m in _CITE_RE.finditer(response):
        full = m.group(0).strip()
        if full and full not in found:
            found.append(full)
    return found[:20]  # cap to avoid bloat


# ---------------------------------------------------------------------------
# Log entry builder
# ---------------------------------------------------------------------------

def build_log_entry(
    *,
    user_lang: str,
    query_text: str,
    query_en: str,
    domain_detected: str,
    response_en: str,
    response_time_ms: int,
    model_used: str = "",
    retrieval_backend: str = "",
) -> dict:
    """Build a structured log row."""
    cited = _extract_cited_sections(response_en)
    return {
        "query_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_lang": user_lang,
        "query_text": query_text[:2000],
        "query_en": query_en[:2000],
        "domain_detected": domain_detected,
        "sections_cited": json.dumps(cited),
        "response_time_ms": response_time_ms,
        "model_used": model_used or os.environ.get("LLM_MODEL", "unknown"),
        "retrieval_backend": retrieval_backend or os.environ.get("NYAYA_RETRIEVAL_BACKEND", "faiss"),
    }


# ---------------------------------------------------------------------------
# Writers (Delta on Databricks, CSV locally)
# ---------------------------------------------------------------------------

_spark = None


def _get_spark():
    """Try to get a SparkSession (only works on Databricks)."""
    global _spark
    if _spark is not None:
        return _spark
    try:
        from pyspark.sql import SparkSession
        _spark = SparkSession.builder.getOrCreate()
        return _spark
    except Exception:
        return None


def _write_delta(entry: dict) -> bool:
    """Write a single log entry to Delta Lake. Returns True on success."""
    spark = _get_spark()
    if spark is None:
        return False
    try:
        from pyspark.sql import Row
        row = Row(**entry)
        df = spark.createDataFrame([row])
        df.write.format("delta").mode("append").saveAsTable(TABLE)
        return True
    except Exception as e:
        logger.debug("Delta write failed (will try CSV): %s", e)
        return False


def _write_csv(entry: dict) -> None:
    """Fallback: append to a local CSV file."""
    import csv
    csv_path = os.environ.get("NYAYA_LOG_CSV", "/tmp/nyaya_query_logs.csv")
    file_exists = os.path.exists(csv_path)
    try:
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(entry.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(entry)
    except Exception as e:
        logger.warning("CSV log write failed: %s", e)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_query(
    *,
    user_lang: str,
    query_text: str,
    query_en: str,
    domain_detected: str,
    response_en: str,
    response_time_ms: int,
    model_used: str = "",
    retrieval_backend: str = "",
) -> None:
    """Log a query + response. Non-blocking, best-effort."""
    try:
        entry = build_log_entry(
            user_lang=user_lang,
            query_text=query_text,
            query_en=query_en,
            domain_detected=domain_detected,
            response_en=response_en,
            response_time_ms=response_time_ms,
            model_used=model_used,
            retrieval_backend=retrieval_backend,
        )
        if not _write_delta(entry):
            _write_csv(entry)
    except Exception as e:
        logger.warning("Query logging failed (non-fatal): %s", e)
