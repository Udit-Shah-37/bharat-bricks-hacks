"""Triage engine — orchestrates domain classification, retrieval, and action plan assembly.

This is the core differentiator: instead of just answering "what is the law",
it answers "given MY situation, what does the law say about ME specifically."
"""
from __future__ import annotations

import sys
import os
# Add src/ directory to Python path for absolute imports
sys.path.insert(0, '/Workspace/Users/saisandeshk@iisc.ac.in/bharat-bricks-hacks/src')

import logging

import pandas as pd

from nyaya_dhwani.domain_classifier import classify_domain, DomainMatch
from nyaya_dhwani.action_knowledge import get_action_plan, format_action_context, ActionPlan
from nyaya_dhwani.case_knowledge import (
    assess_case_strength,
    format_strength_context,
    get_similar_cases,
    format_case_references,
    get_government_schemes,
    format_scheme_context,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Triage system prompt
# ---------------------------------------------------------------------------

TRIAGE_SYSTEM_PROMPT = """\
You are Nyaya-Sahayak, a legal first-response assistant for Indian citizens.

ABSOLUTE RULES — NEVER BREAK:
- Zero fabrication. Only cite section numbers, article numbers, case names, helplines, fees, deadlines, and portals that appear verbatim in the provided context blocks.
- Never reference your context blocks, retrieval results, or their contents to the user — not to explain an answer, not to flag a gap, not to justify what you did or didn't include. Just answer. If a section has nothing to include, silently omit it.

Goal:
- Explain what the law means for this person's specific situation.
- Give practical next steps.
- Be warm, clear, and honest.

Conversation mode:
- Read the conversation history before responding.
- If this is a follow-up, answer it directly and briefly.
- Do not restart full analysis unless new facts materially change the case.
- Do not repeat earlier sections unless asked.

Choose response style:
- Full structured format: first substantive question, or when new facts change the picture.
- Short conversational format: clarifications, confirmations, narrow follow-ups, or when the person is clearly distressed — in that case acknowledge them as a human first, then address the legal question.

Full structured format:
1) Your Situation & Applicable Law
   2-3 sentences. Name relevant domain(s): criminal, constitutional, consumer, family, labour, property. Explain why they apply to this person's facts.

2) What the Law Says
   Relevant Provisions: cite only from === STATUTES ===.
   Constitutional Rights: cite only from === CONSTITUTIONAL PROVISIONS ===.

3) What Courts Have Said
   Cite only cases from === SUPREME COURT JUDGMENTS ===.

4) How Strong Is Your Case
   Use === CASE STRENGTH ===. State: Strong / Moderate / Needs More Evidence.
   List concrete evidence and documents to gather.

5) What You Should Do Now
   Use === ACTION PLAN === exactly for helplines, fees, deadlines, portals.
   Step-by-step: where to go, what to say, what to bring, what to expect.

6) Help Available to You
   From === ACTION PLAN === and === GOVERNMENT SCHEMES ===.

7) What I Referred To
   List only items actually used in this answer.
   Group by: statutes/articles — judgments — practical resources.
   One short reason per item (5-12 words).
   Skip this section if no specific citations were used.

Tone:
- Address the person as "you", never "the user".
- Plain language. Explain any legal term you use.
- Never be preachy.
"""


# ---------------------------------------------------------------------------
# 3B.4 — Clarifying question flow
# ---------------------------------------------------------------------------

# Domain-specific follow-up questions when query is too vague
_CLARIFYING_QUESTIONS: dict[str, list[str]] = {
    "criminal": [
        "When did this incident happen (approximate date)?",
        "Where did it happen (city/state)?",
        "Do you have any evidence like photos, videos, witnesses, or a medical report?",
        "Have you already filed an FIR or police complaint?",
    ],
    "consumer": [
        "What product or service is this about?",
        "Do you have a receipt, invoice, or order confirmation?",
        "How much money is involved (approximate amount)?",
        "Have you already contacted the seller or company to complain?",
    ],
    "constitutional": [
        "Which government authority or department is involved?",
        "Do you have any written order, rejection letter, or receipt?",
        "When did you file the application or request?",
    ],
    "family": [
        "Are you married? If yes, how long have you been married?",
        "Do you have children? If yes, how many and what age?",
        "Do you have a marriage certificate or other documents?",
        "Are you currently living together or separated?",
    ],
    "labour": [
        "What type of employment — permanent, contract, or daily wage?",
        "Do you have an offer letter, appointment letter, or salary slip?",
        "How long have you worked at this place?",
        "What is your approximate monthly salary?",
    ],
    "property": [
        "Do you have a sale deed, title deed, or rental agreement?",
        "Is this about owned property or rented?",
        "Where is the property located (city/state)?",
        "How long have you been in possession or dispute?",
    ],
}

_GENERIC_QUESTIONS = [
    "Can you describe your situation in more detail?",
    "What specific problem are you facing?",
    "When did this issue start?",
    "What outcome are you hoping for?",
]


def detect_clarifying_needed(query_en: str, domains: list[DomainMatch]) -> list[str] | None:
    """Return follow-up questions if the query is too vague, else None.

    Triggers when:
    - No domain detected (completely ambiguous)
    - Query is very short (<8 words) with low confidence
    - Domain detected but no situational detail
    """
    words = query_en.strip().split()
    word_count = len(words)

    # Very short queries with no domain → ask generic questions
    if not domains and word_count < 12:
        return _GENERIC_QUESTIONS[:3]

    # Very short even with a domain match → ask domain-specific questions
    if domains and word_count < 8 and domains[0].confidence < 0.6:
        domain = domains[0].domain
        questions = _CLARIFYING_QUESTIONS.get(domain, _GENERIC_QUESTIONS)
        return questions[:3]

    # Domain found but very low confidence → suggest clarification
    if domains and domains[0].confidence < 0.3:
        domain = domains[0].domain
        questions = _CLARIFYING_QUESTIONS.get(domain, _GENERIC_QUESTIONS)
        return questions[:2]

    return None


def format_clarifying_response(questions: list[str]) -> str:
    """Format clarifying questions as a friendly markdown response."""
    lines = [
        "I want to help you with the best possible guidance. To do that, could you provide a bit more detail?\n",
        "**Please clarify:**",
    ]
    for i, q in enumerate(questions, 1):
        lines.append(f"{i}. {q}")
    lines.append("\nYou can answer any or all of these, and I'll provide specific legal guidance for your situation.")
    lines.append("\n⚖️ *This is informational guidance only. Please consult a qualified lawyer for your specific situation.*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core triage functions
# ---------------------------------------------------------------------------

def build_triage_context(
    query_en: str,
    chunks_df: pd.DataFrame,
    history_context: str = "",
) -> tuple[list[DomainMatch], ActionPlan | None, str]:
    """Classify domain, look up action plan, and build enriched context for LLM.

    Returns: (domains, action_plan, enriched_user_message)
    """
    # 1. Classify domain
    domains = classify_domain(query_en)
    logger.info("Domain classification: %s", [(d.domain, d.situation_type, f"{d.confidence:.2f}") for d in domains])

    # 2. Look up deterministic action plan
    action_plan = None
    action_context = ""
    strength_context = ""
    cases_context = ""
    schemes_context = ""

    if domains:
        primary = domains[0]
        action_plan = get_action_plan(primary.domain, primary.situation_type)
        if action_plan:
            action_context = format_action_context(action_plan)
            logger.info("Action plan found: %s / %s", primary.domain, primary.situation_type)
        else:
            logger.info("No action plan for: %s / %s", primary.domain, primary.situation_type)

        # Phase 3: Case strength assessment
        strength = assess_case_strength(query_en, primary.domain)
        strength_context = format_strength_context(strength)

        # Phase 3: Similar SC judgment references
        cases = get_similar_cases(primary.domain, primary.situation_type)
        cases_context = format_case_references(cases)

        # Phase 3: Government schemes
        schemes = get_government_schemes(primary.domain, primary.situation_type)
        schemes_context = format_scheme_context(schemes)

    # 3. Build enriched context — categorised by source type for the LLM
    statute_texts = []
    constitutional_texts = []
    sc_judgment_texts = []

    for _, row in chunks_df.iterrows():
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        doc_type = str(row.get("doc_type", "")).lower()
        title = str(row.get("title", ""))

        if doc_type in ("constitutional_law",) or "Article" in title:
            constitutional_texts.append(text)
        elif doc_type in ("sc_judgment", "sc_judgment_qa"):
            sc_judgment_texts.append(text)
        else:
            # BNS sections, consumer_law, family_law, labour_law, property_law, etc.
            statute_texts.append(text)

    # Combine RAG context + action plan + Phase 3 enrichments
    parts = []
    if statute_texts:
        parts.append("=== STATUTES (BNS / Act Provisions) ===\n" + "\n\n".join(statute_texts))
    if constitutional_texts:
        parts.append("=== CONSTITUTIONAL PROVISIONS ===\n" + "\n\n".join(constitutional_texts))
    if sc_judgment_texts:
        parts.append("=== SUPREME COURT JUDGMENTS ===\n" + "\n\n".join(sc_judgment_texts))
    if action_context:
        parts.append(action_context)
    if strength_context:
        parts.append(strength_context)
    if cases_context:
        parts.append(cases_context)
    if schemes_context:
        parts.append(schemes_context)

    combined_context = "\n\n".join(parts) if parts else "(No relevant context found)"

    conversation_block = ""
    if history_context.strip():
        conversation_block = f"\n\nConversation history:\n{history_context.strip()}"

    user_message = f"Context:\n{combined_context}{conversation_block}\n\nUser's situation: {query_en}"

    return domains, action_plan, user_message


def format_triage_citations(chunks_df: pd.DataFrame, domains: list[DomainMatch]) -> str:
    """Format citations with domain labels, grouped by source type."""
    lines: list[str] = []

    # Domain label
    if domains:
        domain_labels = [f"**{d.domain.title()}** ({d.situation_type.replace('_', ' ')})" for d in domains[:3]]
        lines.append(f"🏷️ Detected domains: {', '.join(domain_labels)}")
        lines.append("")

    # Group citations by doc_type
    statutes: list[str] = []
    constitutional: list[str] = []
    sc_judgments: list[str] = []

    for _, row in chunks_df.iterrows():
        title = str(row.get("title") or "").strip()
        source = str(row.get("source") or "").strip()
        doc_type = str(row.get("doc_type") or "").strip().lower()

        if not title and not source:
            continue

        label = title or source
        if len(label) > 120:
            label = label[:117] + "..."

        if doc_type in ("constitutional_law",) or "Article" in title:
            constitutional.append(f"- 📜 {label}")
        elif doc_type in ("sc_judgment", "sc_judgment_qa"):
            sc_judgments.append(f"- ⚖️ {label}")
        else:
            statutes.append(f"- 📖 {label}")

    if statutes:
        lines.append("**Statutes & Act Provisions:**")
        lines.extend(statutes[:5])
    if constitutional:
        lines.append("**Constitutional Provisions:**")
        lines.extend(constitutional[:5])
    if sc_judgments:
        lines.append("**Supreme Court Judgments:**")
        lines.extend(sc_judgments[:5])

    return "\n".join(lines) if lines else "(no sources)"


def post_process_response(llm_response: str) -> str:
    """Light response cleanup.

    Quick reference data is no longer hardcoded in post-processing.
    Any "what was referred" section must come from the LLM output itself.
    """
    return llm_response
