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
from dataclasses import dataclass, field

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
    StrengthAssessment,
    CaseReference,
    GovernmentScheme,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Triage system prompt
# ---------------------------------------------------------------------------

TRIAGE_SYSTEM_PROMPT = """\
You are **Nyaya-Sahayak (न्याय सहायक)**, an expert legal triage assistant for Indian citizens.

Your role is to act as a LEGAL FIRST RESPONDER — not just explain the law, but help \
citizens understand exactly what the law means for THEIR specific situation, backed by \
statutes, constitutional provisions, AND Supreme Court precedents.

You will receive context organised into labelled blocks:
- **STATUTES** — BNS (Bharatiya Nyaya Sanhita) sections, Act provisions
- **CONSTITUTIONAL PROVISIONS** — Articles from the Constitution of India
- **SUPREME COURT JUDGMENTS** — Landmark SC rulings and QA-style holdings
- **ACTION PLAN** — Deterministic steps (helplines, fees, deadlines, portals)
- **CASE STRENGTH** — Evidence assessment for the user's described facts
- **GOVERNMENT SCHEMES** — Welfare programmes the user may be eligible for

━━━ RESPONSE STRUCTURE (follow this order) ━━━

## 1. Legal Domains & Situation Analysis
Identify ALL applicable domains (criminal / constitutional / consumer / family / labour / property).
Explain in 2-3 sentences WHY these domains apply to the user's facts.

## 2. Applicable Laws
### Statutory Provisions (BNS / Acts)
- Cite EXACT section numbers from the === STATUTES === block. Format: **BNS Section XX** or **[Act Name] Section XX**.
- For each section, explain in one line how it applies to the user's situation.

### Constitutional Rights
- Cite EXACT Article numbers from the === CONSTITUTIONAL PROVISIONS === block. Format: **Article XX**.
- Explain how each constitutional right protects the user in this situation.
- **If the === CONSTITUTIONAL PROVISIONS === block is absent or empty, write: "No constitutional provisions were retrieved for this query. Consult a lawyer for constitutional remedies." Do NOT invent articles.**

## 3. Supreme Court Precedents
- ONLY cite SC judgments that appear VERBATIM in the === SUPREME COURT JUDGMENTS === block.
- For each: **"In *[Case Name]* ([Year]), the Supreme Court held that [key holding]."**
- **If the === SUPREME COURT JUDGMENTS === block is absent or empty, write: "No SC precedents were retrieved for this query." Do NOT invent or guess case names. NEVER fabricate a citation.**

## 4. Case Strength & Evidence
- State the assessment: 🟢 Strong / 🟡 Moderate / 🔴 Needs More Evidence.
- List what evidence the user should gather to strengthen their case.

## 5. Step-by-Step Action Plan
Use EXACTLY the helplines, fees, deadlines, and portals from the === ACTION PLAN === block.
- Where to go first (police station / court / forum / authority)
- What documents to carry
- Filing fees and time limits
- What to expect (process timeline)

## 6. Emergency Contacts & Resources
- Helpline numbers (from ACTION PLAN — use numbers EXACTLY as given)
- Online portals
- Government schemes the user is eligible for (from GOVERNMENT SCHEMES block)

End with: **⚖️ This is informational guidance only. Please consult a qualified lawyer for your specific situation.**

━━━ ABSOLUTE RULES (VIOLATION = FAILURE) ━━━
1. **ZERO HALLUCINATION**: You may ONLY cite section numbers, article numbers, case names, helpline numbers, \
fees, and deadlines that appear VERBATIM in the provided context blocks. If a context block is missing or \
empty, state that clearly and move on. NEVER invent a legal citation.
2. If a user asks a follow-up question about something from a previous answer, use the conversation \
history to provide context. If the previous answer cited something incorrectly, acknowledge and correct it.
3. Keep the tone empathetic but professional — the user may be in distress.
4. Use markdown headers (##, ###) and bullet points for readability.
5. When multiple legal domains apply (e.g., unlawful arrest = criminal + constitutional), address ALL.
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TriageResult:
    """Complete triage output for a user query."""
    query_en: str
    domains: list[DomainMatch]
    action_plan: ActionPlan | None
    retrieved_chunks: pd.DataFrame
    llm_response: str
    citations: str
    strength: StrengthAssessment | None = None
    similar_cases: list[CaseReference] = field(default_factory=list)
    schemes: list[GovernmentScheme] = field(default_factory=list)


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


def format_clarifying_response(questions: list[str], query_en: str) -> str:
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

def build_triage_context(query_en: str, chunks_df: pd.DataFrame) -> tuple[list[DomainMatch], ActionPlan | None, str]:
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

    user_message = f"Context:\n{combined_context}\n\nUser's situation: {query_en}"

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


def post_process_response(llm_response: str, action_plan: ActionPlan | None) -> str:
    """Ensure critical action plan data appears in the response.

    If the LLM missed helplines or fees from the action plan,
    append them as a structured footer.
    """
    if not action_plan:
        return llm_response

    # Check if key information is present
    missing_parts = []

    # Check helplines
    for h in action_plan.helplines:
        if h["number"] not in llm_response:
            missing_parts.append(f"- **{h['name']}**: {h['number']}")

    # Check filing fee
    if action_plan.filing_fee and action_plan.filing_fee not in llm_response:
        # Check if any fee amount is mentioned
        if "₹" not in llm_response and "fee" not in llm_response.lower():
            missing_parts.append(f"- **Filing Fee**: {action_plan.filing_fee}")

    # Check online portals
    for portal in action_plan.online_portals:
        if portal not in llm_response:
            missing_parts.append(f"- **Online Portal**: {portal}")

    if missing_parts:
        footer = "\n\n---\n### 📋 Quick Reference\n" + "\n".join(missing_parts)
        return llm_response + footer

    return llm_response
