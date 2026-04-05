"""Application service for follow-up-aware legal triage responses."""

from __future__ import annotations

import logging
import time

from nyaya_dhwani.domain_classifier import classify_domain
from nyaya_dhwani.llm_client import chat_completions, extract_assistant_text
from nyaya_dhwani.retrievers import Retriever, get_retriever
from nyaya_dhwani.triage_engine import (
    TRIAGE_SYSTEM_PROMPT,
    build_triage_context,
    detect_clarifying_needed,
    format_clarifying_response,
    format_triage_citations,
    post_process_response,
)

logger = logging.getLogger(__name__)


def _assistant_core_text(markdown_text: str) -> str:
    if not markdown_text:
        return ""
    return markdown_text.split("\n---\n", 1)[0].strip()


def _history_to_lines(
    history_pairs: list[list[str]] | None,
    *,
    max_turns: int,
    max_chars: int,
) -> list[str]:
    if not history_pairs:
        return []

    selected = history_pairs[-max_turns:]
    lines: list[str] = []
    for idx, pair in enumerate(selected, start=1):
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        user_raw, assistant_raw = pair
        user = " ".join(str(user_raw or "").split())[:500].strip()
        assistant = " ".join(_assistant_core_text(str(assistant_raw or "")).split())[:500].strip()
        if user:
            lines.append(f"Turn {idx} user: {user}")
        if assistant:
            lines.append(f"Turn {idx} assistant: {assistant}")

    if not lines:
        return []

    text = "\n".join(lines)
    if len(text) <= max_chars:
        return lines

    truncated = text[-max_chars:]
    return truncated.splitlines()


def _build_followup_query(query_en: str, history_lines: list[str]) -> str:
    query = (query_en or "").strip()
    if not query or not history_lines or len(query.split()) >= 14:
        return query
    user_bits = [line.replace("Turn ", "").split(" user: ", 1)[1] for line in history_lines if " user: " in line]
    if not user_bits:
        return query
    previous = " | ".join(user_bits[-3:])
    return f"Previous context: {previous}\nCurrent follow-up: {query}"


class TriageService:
    """Coordinates retrieval, triage context, and LLM answer generation."""

    def __init__(self, *, system_prompt: str = TRIAGE_SYSTEM_PROMPT, top_k: int = 12) -> None:
        self._system_prompt = system_prompt
        self._top_k = top_k
        self._retriever: Retriever | None = None

    def _ensure_retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = get_retriever()
            logger.info("Retriever loaded: %s", type(self._retriever).__name__)
        return self._retriever

    def answer(self, query_en: str, history_pairs: list[list[str]] | None = None) -> tuple[str, str, str, int]:
        t0 = time.perf_counter()

        history_lines = _history_to_lines(history_pairs, max_turns=4, max_chars=2200)
        retrieval_query = _build_followup_query(query_en, history_lines)
        history_context = "\n".join(history_lines)

        domains_quick = classify_domain(retrieval_query)
        clarify_qs = detect_clarifying_needed(retrieval_query, domains_quick)
        if clarify_qs:
            clarify_response = format_clarifying_response(clarify_qs)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            domain = domains_quick[0].domain if domains_quick else "unknown"
            return (
                clarify_response,
                "(clarifying question — no retrieval yet)",
                domain,
                elapsed_ms,
            )

        retriever = self._ensure_retriever()
        chunks_df = retriever.search(retrieval_query, k=self._top_k)

        domains, action_plan, enriched_user_msg = build_triage_context(
            query_en,
            chunks_df,
            history_context=history_context,
        )

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": enriched_user_msg},
        ]
        raw = chat_completions(messages, max_tokens=3072, temperature=0.3)
        assistant_en = extract_assistant_text(raw)
        assistant_en = post_process_response(assistant_en, action_plan)

        citations = format_triage_citations(chunks_df, domains)
        domain = domains[0].domain if domains else "unknown"
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        return (assistant_en, citations, domain, elapsed_ms)
