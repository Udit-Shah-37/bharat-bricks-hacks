"""Legal domain classifier for user queries.

Classifies user situations into legal domains to improve retrieval
and enable domain-specific action plans.
"""

from __future__ import annotations

import re
from typing import NamedTuple


class DomainMatch(NamedTuple):
    domain: str
    confidence: float  # 0.0–1.0
    situation_type: str  # sub-category for action_knowledge lookup

# Domain keywords — order matters (first match wins for situation_type)
_DOMAIN_RULES: dict[str, list[tuple[str, re.Pattern]]] = {
    "criminal": [
        ("domestic_violence", re.compile(
            r"\b(husband|wife|spouse|partner|beats?|hitt\w*|slap\w*|punch\w*|"
            r"domestic\s*violence|cruelty|dowry|harass\w*|threaten\w*|"
            r"dv\s*act|protection\s*order|streedhan)\b", re.I)),
        ("assault", re.compile(
            r"\b(assault\w*|attack\w*|hurt\w*|injur\w*|wound\w*|grievous|stab\w*|"
            r"knife|bodily\s*harm|physical\s*violence|beaten|badly)\b", re.I)),
        ("theft", re.compile(
            r"\b(stole|stolen|theft|rob\w*|robbery|burglar\w*|snatch\w*|"
            r"pickpocket|loot\w*|dacoity|extortion|broke\s*into)\b", re.I)),
        ("fraud", re.compile(
            r"\b(fraud\w*|cheat\w*|deceiv\w*|forgery|scam\w*|ponzi|"
            r"misrepresent\w*|impersonat\w*|fake\s*document)\b", re.I)),
        ("murder_homicide", re.compile(
            r"\b(murder\w*|kill\w*|homicide|culpable\s*homicide|death\s*threat)\b", re.I)),
        ("sexual_offence", re.compile(
            r"\b(rape|sexual\s*assault|molest\w*|stalk\w*|voyeurism|"
            r"sexual\s*harassment|outraging\s*modesty|eve\s*teasing|"
            r"inappropriate\s*comment\w*)\b", re.I)),
        ("kidnapping", re.compile(
            r"\b(kidnap\w*|abduct\w*|wrongful\s*confine\w*|hostage|detain\w*)\b", re.I)),
        ("criminal_general", re.compile(
            r"\b(FIR|police\s*complaint|bail|arrest\w*|criminal|"
            r"offence|offense|crime|BNS|IPC|goons?|intimidat\w*)\b", re.I)),
    ],
    "consumer": [
        ("defective_product", re.compile(
            r"\b(defective|faulty|broken|not\s*working|malfunction\w*|"
            r"stopped\s*working|product\s*issue|warranty|replace\w*|"
            r"bought.*(?:online|shop|store)|phone|laptop|appliance)\b", re.I)),
        ("service_deficiency", re.compile(
            r"\b(service\s*(?:deficiency|issue|complaint)|poor\s*service|"
            r"delay\w*|not\s*deliver\w*|delivery\s*issue|courier|"
            r"insurance\w*.*(?:denied|reject\w*|claim)|hospital\s*negligence|"
            r"overcharg\w*|hidden\s*charges|not\s*respond\w*)\b", re.I)),
        ("unfair_trade", re.compile(
            r"\b(misleading\s*ad|false\s*advertising|unfair\s*trade|"
            r"consumer\s*(?:forum|court|complaint|protection)|"
            r"refund\w*|seller\s*not\s*respond|e-?commerce)\b", re.I)),
    ],
    "constitutional": [
        ("rti_denial", re.compile(
            r"\b(RTI|right\s*to\s*information|PIO|public\s*information\s*officer|"
            r"information\s*commission|transparency|denial\s*of\s*information|"
            r"appointment\s*letter.*(?:government|govt)|"
            r"(?:government|govt).*(?:deny|denied|refusing|not\s*respond))\b", re.I)),
        ("fundamental_rights", re.compile(
            r"\b(fundamental\s*right|article\s*(?:14|15|16|17|19|20|21|22|23|24|25|32)|"
            r"right\s*to\s*(?:life|equality|freedom|education|privacy)|"
            r"discriminat\w*|untouchability|forced\s*labour|"
            r"constitutional\s*right|caste|denied\s*admission|"
            r"detain\w*.*magistrate|magistrate.*detain\w*|"
            r"without\s*produc\w*)\b", re.I)),
        ("writ_petition", re.compile(
            r"\b(writ\s*petition|habeas\s*corpus|mandamus|certiorari|"
            r"prohibition|quo\s*warranto|high\s*court\s*petition)\b", re.I)),
    ],
    "family": [
        ("divorce", re.compile(
            r"\b(divorce\w*|mutual\s*consent|separation|judicial\s*separation|"
            r"marriage.*(?:break|end|dissolv))\b", re.I)),
        ("maintenance", re.compile(
            r"\b(maintenance|alimony|section\s*125|streedhan|"
            r"wife.*(?:support|money)|husband.*(?:pay|support|left|sends?\s*no)|"
            r"no\s*money|expenses?\s*(?:not|no)|child.*(?:money|support|expense))\b", re.I)),
        ("custody", re.compile(
            r"\b(custody|child\s*(?:support|care|guardian)|visitation|"
            r"guardian|adoption)\b", re.I)),
    ],
    "labour": [
        ("wrongful_termination", re.compile(
            r"\b(terminat\w*|fired|sacked|dismiss\w*|wrongful\w*\s+(?:terminat|dismiss)\w*|"
            r"notice\s*period|retrench\w*|layoff|lay-off|severance|"
            r"company\s*fired|without\s*(?:notice|reason))\b", re.I)),
        ("unpaid_wages", re.compile(
            r"\b(unpaid\s*(?:wages|salary|dues)|wage\s*theft|salary\s*not\s*paid|"
            r"overtime|minimum\s*wage|bonus\s*not\s*paid|PF\s*not\s*deposit\w*|"
            r"not\s*(?:deposit\w*|paid)\s*(?:PF|salary|wages|gratuity))\b", re.I)),
        ("workplace_harassment", re.compile(
            r"\b(workplace\s*harass|POSH|ICC|internal\s*complaints?\s*committee|"
            r"sexual\s*harassment\s*at\s*work|hostile\s*work)\b", re.I)),
        ("labour_general", re.compile(
            r"\b(labour|labor|employ|worker|industrial\s*dispute|"
            r"trade\s*union|gratuity|EPF|ESI)\b", re.I)),
    ],
    "property": [
        ("tenant_rights", re.compile(
            r"\b(tenant|landlord|rent\w*|evict\w*|lease|rental\s*agreement|"
            r"security\s*deposit|vacancy|paying\s*guest|vacat\w*\s*(?:shop|house|flat|room))\b", re.I)),
        ("land_dispute", re.compile(
            r"\b(land\s*(?:dispute|grab|encroach\w*)|property\s*(?:dispute|fraud)|"
            r"encroach\w*|title\s*deed|registration|mutation|encumbrance|"
            r"partition|ancestral\s*property|succession|will\b|inheritance)\b", re.I)),
    ],
}


def classify_domain(query: str) -> list[DomainMatch]:
    """Classify a user query into one or more legal domains.

    Returns a list of DomainMatch sorted by confidence (descending).
    Multiple domains can match (e.g., domestic violence = criminal + family).
    """
    if not query or not query.strip():
        return []

    matches: list[DomainMatch] = []
    query_lower = query.lower()

    for domain, rules in _DOMAIN_RULES.items():
        best_situation = None
        best_score = 0.0
        total_hits = 0

        for situation_type, pattern in rules:
            hits = len(pattern.findall(query_lower))
            if hits > 0:
                total_hits += hits
                # First matching situation_type gets priority
                if best_situation is None:
                    best_situation = situation_type
                    best_score = min(0.5 + hits * 0.15, 1.0)

        if best_situation:
            # Boost confidence if multiple sub-patterns matched
            confidence = min(best_score + total_hits * 0.05, 1.0)
            matches.append(DomainMatch(domain, confidence, best_situation))

    # Sort by confidence descending
    matches.sort(key=lambda m: m.confidence, reverse=True)
    return matches


def get_primary_domain(query: str) -> DomainMatch | None:
    """Return the single most likely domain, or None if no match."""
    matches = classify_domain(query)
    return matches[0] if matches else None


def get_domain_labels(query: str) -> list[str]:
    """Return domain names only (convenience for filtering)."""
    return [m.domain for m in classify_domain(query)]
