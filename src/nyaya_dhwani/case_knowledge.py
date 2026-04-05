"""Case strength assessment, similar case references, and government scheme linkage.

Phase 3 enhancement module. Provides:
- Case strength assessment based on described facts vs legal elements
- Landmark SC judgment references keyed by (domain, situation_type)
- Government welfare scheme suggestions based on user's situation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 3B.1 — Case Strength Assessment
# ---------------------------------------------------------------------------

@dataclass
class StrengthAssessment:
    level: str          # "strong", "moderate", "needs_more_evidence"
    reason: str
    evidence_tips: list[str]


# Evidence signals extracted from user queries
_STRONG_SIGNALS = {
    "criminal": re.compile(
        r"(witness|cctv|video|photo|medical\s*report|mlc|fir\s*already|police\s*report|"
        r"hospital|receipt|proof|screenshot|recording|caught)", re.I),
    "consumer": re.compile(
        r"(receipt|invoice|bill|warranty|screenshot|email|order\s*id|tracking|"
        r"bank\s*statement|payment\s*proof|written\s*complaint)", re.I),
    "constitutional": re.compile(
        r"(rti\s*application|receipt|acknowledgment|written\s*denial|order\s*copy|"
        r"govt?\s*letter|official\s*document|notification)", re.I),
    "family": re.compile(
        r"(marriage\s*certificate|salary\s*slip|bank\s*statement|income\s*proof|"
        r"property\s*document|witness|message|call\s*record)", re.I),
    "labour": re.compile(
        r"(offer\s*letter|salary\s*slip|appointment|id\s*card|email|contract|"
        r"termination\s*letter|bank\s*statement|pf\s*statement)", re.I),
    "property": re.compile(
        r"(sale\s*deed|title\s*deed|registry|encumbrance|mutation|tax\s*receipt|"
        r"rent\s*receipt|agreement|survey|witness)", re.I),
}

_WEAK_SIGNALS = re.compile(
    r"(i\s*think|maybe|not\s*sure|don\'?t\s*know|no\s*proof|no\s*evidence|"
    r"no\s*witness|verbal|oral\s*agreement|nothing\s*in\s*writing)", re.I)


def assess_case_strength(query_en: str, domain: str) -> StrengthAssessment:
    """Assess user's case strength based on mentioned evidence signals."""
    strong_pat = _STRONG_SIGNALS.get(domain, _STRONG_SIGNALS.get("criminal"))
    strong_hits = len(strong_pat.findall(query_en)) if strong_pat else 0
    weak_hits = len(_WEAK_SIGNALS.findall(query_en))

    if strong_hits >= 2:
        return StrengthAssessment(
            level="strong",
            reason="You appear to have multiple forms of evidence which strengthens your position.",
            evidence_tips=["Preserve all existing evidence — do not delete messages, photos, or documents",
                           "Get certified copies of any official documents",
                           "Note down details while they are fresh in memory"],
        )
    elif strong_hits >= 1 and weak_hits == 0:
        return StrengthAssessment(
            level="moderate",
            reason="You have some evidence, but gathering additional supporting documents would strengthen your case.",
            evidence_tips=_evidence_tips_for_domain(domain),
        )
    elif weak_hits >= 1 and strong_hits == 0:
        return StrengthAssessment(
            level="needs_more_evidence",
            reason="Your case would benefit significantly from more documentation. Focus on gathering the evidence listed below.",
            evidence_tips=_evidence_tips_for_domain(domain),
        )
    else:
        # Default moderate — user described situation but didn't mention evidence
        return StrengthAssessment(
            level="moderate",
            reason="Based on the facts described, your case has merit. Strengthening it with documentary evidence is recommended.",
            evidence_tips=_evidence_tips_for_domain(domain),
        )


def _evidence_tips_for_domain(domain: str) -> list[str]:
    tips = {
        "criminal": [
            "File an FIR immediately — it becomes official record of the crime",
            "Get a medico-legal certificate if there are physical injuries",
            "Collect CCTV footage from nearby establishments (they may erase after 15-30 days)",
            "Note down names and contact details of any witnesses",
            "Preserve any communication (messages, call records) related to the incident",
        ],
        "consumer": [
            "Keep the purchase receipt/invoice and warranty card",
            "Screenshot all online communications with the seller",
            "Take photos/videos of the defective product",
            "Send a written complaint via email (creates a paper trail)",
            "Save the order confirmation and delivery details",
        ],
        "constitutional": [
            "Keep a copy of your RTI application with acknowledgment",
            "Document the denial or non-response with dates",
            "Collect any official communication or orders",
            "Note down names and designations of officials involved",
        ],
        "family": [
            "Secure your marriage certificate and other relationship documents",
            "Collect evidence of spouse's income (salary slips, tax returns, social media showing lifestyle)",
            "Keep records of any financial transactions between parties",
            "Preserve messages, emails, or call records showing conduct",
        ],
        "labour": [
            "Keep copies of your offer letter, appointment letter, and employment contract",
            "Save salary slips and bank statements showing salary credits",
            "Get a copy of the termination letter (request in writing if not given)",
            "Preserve emails and messages related to your employment and termination",
        ],
        "property": [
            "Obtain the chain of title documents (sale deed, mutation records)",
            "Get an Encumbrance Certificate from the Sub-Registrar",
            "Keep tax payment receipts and utility bills as proof of possession",
            "Document any encroachment with photographs and survey reports",
        ],
    }
    return tips.get(domain, tips["criminal"])


def format_strength_context(assessment: StrengthAssessment) -> str:
    """Format case strength assessment for injection into LLM context."""
    level_emoji = {"strong": "🟢", "moderate": "🟡", "needs_more_evidence": "🔴"}
    level_label = {"strong": "Strong", "moderate": "Moderate", "needs_more_evidence": "Needs More Evidence"}

    tips_str = "\n".join(f"  - {t}" for t in assessment.evidence_tips)
    return (
        f"\n=== CASE STRENGTH ASSESSMENT ===\n"
        f"Assessment: {level_emoji.get(assessment.level, '🟡')} {level_label.get(assessment.level, 'Moderate')}\n"
        f"Reason: {assessment.reason}\n"
        f"Evidence tips to strengthen your case:\n{tips_str}\n"
    )


# ---------------------------------------------------------------------------
# 3B.2 — Similar Case References (Landmark SC Judgments)
# ---------------------------------------------------------------------------

@dataclass
class CaseReference:
    case_name: str
    citation: str
    year: int
    key_holding: str
    relevance: str  # Why this case is relevant to user's situation


# Keyed by (domain, situation_type) — matches action_knowledge keys
_LANDMARK_CASES: dict[tuple[str, str], list[CaseReference]] = {
    ("criminal", "domestic_violence"): [
        CaseReference(
            "Arnesh Kumar v. State of Bihar", "(2014) 8 SCC 273", 2014,
            "Police should not automatically arrest in Section 498A/BNS 85 cases. Arrest must be justified with recorded reasons.",
            "Protects your rights during the complaint process — ensures due process is followed"),
        CaseReference(
            "Rajesh Sharma v. State of UP", "(2017) 8 SCC 446", 2017,
            "Family Welfare Committees in every district examine 498A complaints before arrest, unless tangible physical injuries are involved.",
            "Relevant to understanding the process after filing a complaint"),
    ],
    ("criminal", "assault"): [
        CaseReference(
            "Tehseen S. Poonawalla v. Union of India", "(2018) 9 SCC 501", 2018,
            "State has affirmative obligation to protect life and liberty. Comprehensive directions for preventing mob violence and ensuring victim compensation.",
            "Establishes the state's duty to protect you and ensure compensation"),
    ],
    ("criminal", "theft"): [
        CaseReference(
            "Lalita Kumari v. Govt of UP", "(2014) 2 SCC 1", 2014,
            "Police MUST register FIR for cognizable offences. Refusal to register FIR is illegal.",
            "If police refuse to take your complaint, cite this Supreme Court judgment"),
    ],
    ("criminal", "fraud"): [
        CaseReference(
            "Lalita Kumari v. Govt of UP", "(2014) 2 SCC 1", 2014,
            "FIR registration is mandatory when information discloses a cognizable offence.",
            "Police cannot refuse to register your fraud complaint"),
    ],
    ("criminal", "sexual_offence"): [
        CaseReference(
            "Vishaka v. State of Rajasthan", "AIR 1997 SC 3011", 1997,
            "Binding guidelines against workplace sexual harassment. Every employer must constitute an ICC. Led to the POSH Act 2013.",
            "Foundation for workplace sexual harassment complaints"),
    ],
    ("criminal", "criminal_general"): [
        CaseReference(
            "Lalita Kumari v. Govt of UP", "(2014) 2 SCC 1", 2014,
            "FIR registration is mandatory for cognizable offences. Police cannot refuse.",
            "Your fundamental right to have a complaint registered"),
        CaseReference(
            "Satender Kumar Antil v. CBI", "(2022) 10 SCC 51", 2022,
            "Bail is the rule, jail is the exception. Comprehensive bail reform guidelines.",
            "If arrested, you have strong rights to seek bail"),
    ],
    ("consumer", "defective_product"): [
        CaseReference(
            "Ambrish Kumar Shukla v. Ferrous Infrastructure", "2017 NCDRC 1118", 2017,
            "Buyers are consumers; non-delivery or defective product entitles refund with interest.",
            "Establishes your right to refund and compensation"),
    ],
    ("consumer", "service_deficiency"): [
        CaseReference(
            "Indian Medical Association v. V.P. Shantha", "(1995) 6 SCC 651", 1995,
            "Medical practitioners and hospitals provide 'service' under Consumer Protection Act. Patients are consumers.",
            "You can approach consumer forum for service deficiency"),
        CaseReference(
            "Jacob Mathew v. State of Punjab", "(2005) 6 SCC 1", 2005,
            "For criminal liability in medical negligence, the standard is 'gross negligence' — not simple error of judgment.",
            "Relevant if your claim involves professional negligence"),
    ],
    ("constitutional", "rti_denial"): [
        CaseReference(
            "CBSE v. Aditya Bandopadhyay", "(2011) 8 SCC 497", 2011,
            "RTI provides access to existing information. PIOs must provide available records but need not create new information.",
            "Clarifies the scope of information you can demand under RTI"),
    ],
    ("constitutional", "fundamental_rights"): [
        CaseReference(
            "Maneka Gandhi v. Union of India", "AIR 1978 SC 597", 1978,
            "Article 21 includes right to live with dignity. Any procedure affecting life/liberty must be just, fair, and reasonable.",
            "Foundational case for any fundamental rights violation"),
        CaseReference(
            "K.S. Puttaswamy v. Union of India", "(2017) 10 SCC 1", 2017,
            "Right to Privacy is a fundamental right under Article 21.",
            "If your privacy rights are being violated"),
    ],
    ("family", "divorce"): [
        CaseReference(
            "Shilpa Sailesh v. Varun Sreenivasan", "(2023) 2 SCC 453", 2023,
            "Supreme Court can grant divorce without the 6-month cooling period if marriage has irretrievably broken down.",
            "Fast-track divorce may be possible in your case"),
    ],
    ("family", "maintenance"): [
        CaseReference(
            "Rajnesh v. Neha", "(2021) 2 SCC 324", 2021,
            "Comprehensive maintenance guidelines: mandatory income disclosure, interim maintenance within 60 days, no double payments across proceedings.",
            "Key guidelines that courts must follow in your maintenance case"),
        CaseReference(
            "Shamima Farooqui v. Shahid Khan", "(2015) 5 SCC 705", 2015,
            "Even divorced Muslim women can claim maintenance under Section 125 CrPC (BNSS 144).",
            "Maintenance rights extend across personal laws"),
    ],
    ("labour", "wrongful_termination"): [
        CaseReference(
            "Workmen of Dimakuchi Tea Estate v. Dimakuchi Tea Estate", "(1958) 1 LLJ 500", 1958,
            "Retrenchment without compliance with Section 25F (notice + compensation) is void. Worker entitled to reinstatement with back wages.",
            "If your employer did not follow proper retrenchment procedure"),
    ],
    ("labour", "unpaid_wages"): [
        CaseReference(
            "PUDR v. Union of India (Asiad Workers Case)", "AIR 1982 SC 1473", 1982,
            "Working for less than minimum wage amounts to 'forced labour' under Article 23. Every worker entitled to minimum wage.",
            "Constitutional protection for your right to fair wages"),
    ],
    ("property", "land_dispute"): [
        CaseReference(
            "Vineeta Sharma v. Rakesh Sharma", "(2020) 9 SCC 1", 2020,
            "Daughters have equal coparcenary rights in ancestral property by birth. This right is retrospective.",
            "If you are a daughter claiming equal share in ancestral property"),
    ],
    ("property", "tenant_rights"): [
        CaseReference(
            "Raghunath Rai Bareja v. Punjab National Bank", "(2007) 2 SCC 230", 2007,
            "Landlord cannot use self-help for eviction — must follow due process. Forcible eviction is illegal.",
            "Your landlord must get a court order to evict you"),
    ],
}


def get_similar_cases(domain: str, situation_type: str) -> list[CaseReference]:
    """Return landmark SC judgments relevant to the user's situation."""
    return _LANDMARK_CASES.get((domain, situation_type), [])


def format_case_references(cases: list[CaseReference]) -> str:
    """Format case references for injection into LLM context."""
    if not cases:
        return ""
    lines = ["\n=== LANDMARK SUPREME COURT JUDGMENTS ==="]
    for c in cases:
        lines.append(
            f"\n📜 {c.case_name}, {c.citation} ({c.year})\n"
            f"   Held: {c.key_holding}\n"
            f"   Relevance to your case: {c.relevance}"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 3B.3 — Government Scheme Linkage
# ---------------------------------------------------------------------------

@dataclass
class GovernmentScheme:
    name: str
    description: str
    eligibility: str
    how_to_apply: str
    portal: str


_SCHEMES: dict[tuple[str, str], list[GovernmentScheme]] = {
    ("criminal", "domestic_violence"): [
        GovernmentScheme(
            "One Stop Centre (Sakhi) Scheme",
            "Provides integrated support to women affected by violence — medical, legal, psychological counselling, temporary shelter, and police assistance under one roof.",
            "Any woman affected by violence, regardless of age, class, caste, or marital status",
            "Visit nearest One Stop Centre or call Women Helpline 181. Centres are located in every district.",
            "wcd.nic.in",
        ),
        GovernmentScheme(
            "Mahila Shakti Kendra",
            "Community-level centres providing awareness, training, and support services for women.",
            "All women, especially those in rural areas",
            "Contact your local Anganwadi centre or District Women and Child Development office",
            "wcd.nic.in",
        ),
        GovernmentScheme(
            "Free Legal Aid (NALSA)",
            "Free legal services for women, SC/ST, persons with disability, victims of trafficking, and those whose annual income is below ₹3 lakh (₹5 lakh in some states).",
            "Women (any income level), SC/ST, disabled persons, victims of disasters/trafficking, persons below income threshold",
            "Call NALSA Helpline 15100 or visit your District Legal Services Authority",
            "nalsa.gov.in",
        ),
    ],
    ("criminal", "sexual_offence"): [
        GovernmentScheme(
            "Victim Compensation Scheme",
            "Compensation to victims of sexual offences from the state government fund. Amount varies by state — typically ₹3-10 lakh.",
            "Victims of rape, acid attacks, sexual assault, and human trafficking",
            "Applied through the District Legal Services Authority or the trial court itself",
            "nalsa.gov.in",
        ),
        GovernmentScheme(
            "One Stop Centre (Sakhi) Scheme",
            "Integrated support for women affected by violence — medical, legal, counselling, shelter, police.",
            "Any woman affected by violence",
            "Call Women Helpline 181 or visit the nearest One Stop Centre",
            "wcd.nic.in",
        ),
    ],
    ("consumer", "defective_product"): [
        GovernmentScheme(
            "Integrated Grievance Redressal (INGRAM)",
            "Portal for consumer awareness and filing complaints against unfair trade practices.",
            "All consumers",
            "Register and file online at consumerhelpline.gov.in or call 1800-11-4000",
            "consumerhelpline.gov.in",
        ),
    ],
    ("consumer", "service_deficiency"): [
        GovernmentScheme(
            "Integrated Grievance Redressal (INGRAM)",
            "Online complaint portal with mediation and escalation facility.",
            "All consumers",
            "Register at consumerhelpline.gov.in or call 1800-11-4000",
            "consumerhelpline.gov.in",
        ),
    ],
    ("family", "maintenance"): [
        GovernmentScheme(
            "Free Legal Aid (NALSA)",
            "Free legal services for women seeking maintenance — covers filing, representation, and enforcement.",
            "Women (any income level) and persons below income threshold",
            "Call NALSA 15100 or visit District Legal Services Authority",
            "nalsa.gov.in",
        ),
        GovernmentScheme(
            "Swadhar Greh Scheme",
            "Temporary shelter, food, counselling, legal aid, and skill training for women in distress.",
            "Women deserted by family, destitute widows, women released from jail, domestic violence survivors, women without support",
            "Contact District Women and Child Development office or call 181",
            "wcd.nic.in",
        ),
    ],
    ("family", "divorce"): [
        GovernmentScheme(
            "Free Legal Aid (NALSA)",
            "Free legal representation for divorce proceedings for women and persons below income threshold.",
            "Women (any income level), persons below income threshold",
            "Call NALSA 15100",
            "nalsa.gov.in",
        ),
    ],
    ("labour", "wrongful_termination"): [
        GovernmentScheme(
            "Atal Beemit Vyakti Kalyan Yojana",
            "Unemployment benefit under ESIC — 50% of average daily wages for up to 90 days to insured workers who lose their job.",
            "Workers covered under ESI scheme who have contributed for at least 2 years",
            "Apply at your nearest ESIC branch office or online at esic.gov.in",
            "esic.gov.in",
        ),
    ],
    ("labour", "unpaid_wages"): [
        GovernmentScheme(
            "e-Shram Portal",
            "National database of unorganized workers. Registration provides accident insurance cover of ₹2 lakh and access to welfare schemes.",
            "Unorganized sector workers aged 16-59",
            "Register at eshram.gov.in or visit CSC centres",
            "eshram.gov.in",
        ),
    ],
    ("constitutional", "fundamental_rights"): [
        GovernmentScheme(
            "Free Legal Aid (NALSA)",
            "Constitutional right to free legal aid under Article 39A and Legal Services Authorities Act.",
            "SC/ST, women, disabled persons, industrial workmen, victims, persons below income threshold",
            "Call 15100 or visit District Legal Services Authority",
            "nalsa.gov.in",
        ),
    ],
    ("property", "land_dispute"): [
        GovernmentScheme(
            "SVAMITVA Scheme (Survey of Villages — Modernized)",
            "Property cards to rural households using drone survey technology, establishing clear ownership for rural properties.",
            "Rural property owners",
            "Implemented by State Revenue departments. Check svamitva.nic.in for coverage",
            "svamitva.nic.in",
        ),
    ],
}


def get_government_schemes(domain: str, situation_type: str) -> list[GovernmentScheme]:
    """Return applicable government schemes for the user's situation."""
    return _SCHEMES.get((domain, situation_type), [])


def format_scheme_context(schemes: list[GovernmentScheme]) -> str:
    """Format government schemes for injection into LLM context."""
    if not schemes:
        return ""
    lines = ["\n=== APPLICABLE GOVERNMENT SCHEMES ==="]
    for s in schemes:
        lines.append(
            f"\n🏛️ {s.name}\n"
            f"   Description: {s.description}\n"
            f"   Eligibility: {s.eligibility}\n"
            f"   How to apply: {s.how_to_apply}\n"
            f"   Portal: {s.portal}"
        )
    return "\n".join(lines) + "\n"
