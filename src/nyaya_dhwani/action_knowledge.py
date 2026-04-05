"""Structured action knowledge for legal triage.

Hand-crafted deterministic data for helplines, filing bodies, fees, and
step-by-step action plans. This ensures critical information (phone numbers,
fees, deadlines) is never hallucinated by the LLM.

Stored as Python dicts for Phase 1 (app-side). In Phase 2, this will be
backed by a Delta Lake table.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ActionPlan:
    domain: str
    situation_type: str
    title: str
    applicable_laws: list[str]
    helplines: list[dict[str, str]]  # [{"name": "...", "number": "..."}]
    filing_body: str
    filing_fee: str
    time_limit: str
    steps: list[str]
    what_to_prove: list[str]
    what_to_expect: str
    online_portals: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Knowledge base — hand-crafted action plans
# ---------------------------------------------------------------------------

_ACTION_PLANS: dict[tuple[str, str], ActionPlan] = {
    # -----------------------------------------------------------------------
    # CRIMINAL
    # -----------------------------------------------------------------------
    ("criminal", "domestic_violence"): ActionPlan(
        domain="criminal",
        situation_type="domestic_violence",
        title="Domestic Violence / Cruelty by Husband or Relatives",
        applicable_laws=[
            "BNS Section 85 — Cruelty by husband or relative of husband",
            "BNS Section 86 — Dowry death",
            "Protection of Women from Domestic Violence Act, 2005",
            "Constitution of India, Article 21 — Right to life and personal liberty",
        ],
        helplines=[
            {"name": "Women Helpline (24/7)", "number": "181"},
            {"name": "Police Emergency", "number": "112"},
            {"name": "National Commission for Women", "number": "7827-170-170"},
        ],
        filing_body="Nearest Police Station (FIR) + Magistrate Court (Protection Order under DV Act)",
        filing_fee="Free — no fee for FIR or DV Act complaint",
        time_limit="No statutory limitation for filing FIR; DV Act protection order can be sought anytime during subsistence of domestic relationship",
        steps=[
            "Call Women Helpline 181 for immediate assistance and guidance",
            "File an FIR at the nearest police station under BNS Section 85/86",
            "If police refuse to register FIR, approach the Superintendent of Police or Magistrate under BNS Section 175(3)",
            "File an application before the Magistrate under the Protection of Women from Domestic Violence Act, 2005",
            "Seek a Protection Order (Section 18 DV Act) to restrain the abuser",
            "Apply for Residence Order (Section 19), Monetary Relief (Section 20), and Custody Order (Section 21) as needed",
            "Contact the nearest Legal Services Authority for free legal aid if unable to afford a lawyer",
        ],
        what_to_prove=[
            "Evidence of physical violence, threats, or emotional abuse (medical reports, photos, witness statements)",
            "Domestic relationship with the respondent (marriage certificate, shared residence proof)",
            "Pattern of cruelty or acts of violence (diary entries, messages, call records)",
        ],
        what_to_expect="Protection Order can be obtained within 3 days of application in urgent cases. Magistrate must dispose of the application within 60 days. Criminal prosecution under BNS 85/86 can result in imprisonment up to 3 years and fine.",
    ),
    ("criminal", "assault"): ActionPlan(
        domain="criminal",
        situation_type="assault",
        title="Physical Assault / Hurt / Grievous Hurt",
        applicable_laws=[
            "BNS Section 115 — Voluntarily causing hurt",
            "BNS Section 117 — Voluntarily causing grievous hurt",
            "BNS Section 118 — Voluntarily causing hurt or grievous hurt by dangerous weapons or means",
            "BNS Section 351 — Criminal intimidation",
        ],
        helplines=[
            {"name": "Police Emergency", "number": "112"},
            {"name": "Ambulance", "number": "108"},
        ],
        filing_body="Nearest Police Station (FIR)",
        filing_fee="Free — no fee for FIR",
        time_limit="File FIR as soon as possible; no strict limitation for cognizable offences but delay weakens the case",
        steps=[
            "Seek immediate medical attention and get a medico-legal certificate (MLC)",
            "Call Police Emergency 112 or go to the nearest police station",
            "File an FIR — police must register it for cognizable offences (BNS 115-118)",
            "Preserve evidence: medical reports, photos of injuries, CCTV footage, witness details",
            "If police refuse to register FIR, send a written complaint to the SP or approach the Magistrate",
        ],
        what_to_prove=[
            "Medical evidence of injury (MLC / hospital records)",
            "Identity of the attacker (witness statements, CCTV)",
            "Circumstances of the assault (time, place, provocation if any)",
        ],
        what_to_expect="Simple hurt (BNS 115): up to 1 year imprisonment or fine up to ₹10,000 or both. Grievous hurt (BNS 117): up to 7 years imprisonment and fine. Investigation by police, followed by chargesheet and trial.",
    ),
    ("criminal", "theft"): ActionPlan(
        domain="criminal",
        situation_type="theft",
        title="Theft / Robbery / Burglary",
        applicable_laws=[
            "BNS Section 303 — Theft",
            "BNS Section 305 — Robbery",
            "BNS Section 306 — Dacoity",
            "BNS Section 329 — Criminal breach of trust",
        ],
        helplines=[
            {"name": "Police Emergency", "number": "112"},
            {"name": "Cyber Crime (if online)", "number": "1930"},
        ],
        filing_body="Nearest Police Station (FIR)",
        filing_fee="Free",
        time_limit="Report immediately; no strict limitation for cognizable offences",
        steps=[
            "File an FIR at the nearest police station immediately",
            "Provide details: what was stolen, approximate value, time, location, suspects",
            "If online theft/fraud, also report at cybercrime.gov.in or call 1930",
            "Preserve any evidence: receipts, CCTV footage, transaction records",
            "Follow up with the Investigating Officer for case progress",
        ],
        what_to_prove=[
            "Ownership or possession of the stolen property",
            "That property was moved without consent",
            "Identity of the accused (if known)",
        ],
        what_to_expect="Theft (BNS 303): up to 3 years imprisonment and fine. Robbery (BNS 305): up to 10 years rigorous imprisonment and fine. Police will investigate and file a chargesheet if evidence supports.",
    ),
    ("criminal", "fraud"): ActionPlan(
        domain="criminal",
        situation_type="fraud",
        title="Fraud / Cheating / Forgery",
        applicable_laws=[
            "BNS Section 318 — Cheating",
            "BNS Section 319 — Cheating by personation",
            "BNS Section 336 — Forgery",
            "BNS Section 340 — Forgery for purpose of cheating",
            "Information Technology Act, 2000 Section 66D — Cheating by personation using computer resources",
        ],
        helplines=[
            {"name": "Police Emergency", "number": "112"},
            {"name": "Cyber Crime Helpline", "number": "1930"},
            {"name": "Cyber Crime Portal", "number": "cybercrime.gov.in"},
        ],
        filing_body="Nearest Police Station + Cyber Crime Cell (if online)",
        filing_fee="Free",
        time_limit="Report immediately; for cheating, limitation is 3 years from date of offence",
        steps=[
            "File an FIR at the nearest police station",
            "If online fraud: also file a complaint at cybercrime.gov.in or call 1930",
            "Gather all evidence: transaction records, screenshots, communications, contracts",
            "If bank fraud: immediately inform your bank to freeze/reverse the transaction",
            "For UPI fraud: report within 3 days for better chances of reversal",
        ],
        what_to_prove=[
            "Deception or false representation by the accused",
            "Inducement to deliver property or valuable security",
            "Financial loss or damage suffered",
            "Documentary evidence (agreements, messages, bank statements)",
        ],
        what_to_expect="Cheating (BNS 318): up to 3 years imprisonment and fine. With forgery: up to 7 years. Cyber fraud cases are investigated by Cyber Crime Cell. If amount > ₹1 lakh, case is prioritized.",
    ),
    ("criminal", "sexual_offence"): ActionPlan(
        domain="criminal",
        situation_type="sexual_offence",
        title="Sexual Offences / Harassment / Stalking",
        applicable_laws=[
            "BNS Section 63 — Rape",
            "BNS Section 74 — Assault or criminal force to woman with intent to outrage her modesty",
            "BNS Section 75 — Sexual harassment",
            "BNS Section 78 — Stalking",
            "BNS Section 79 — Voyeurism",
        ],
        helplines=[
            {"name": "Women Helpline (24/7)", "number": "181"},
            {"name": "Police Emergency", "number": "112"},
            {"name": "NCW WhatsApp", "number": "7827-170-170"},
        ],
        filing_body="Nearest Police Station — police MUST register FIR (zero FIR provision applies)",
        filing_fee="Free",
        time_limit="No limitation period for rape; report as soon as possible for other offences",
        steps=[
            "Call Women Helpline 181 or Police 112 immediately",
            "File an FIR — ANY police station must accept it (Zero FIR provision)",
            "Get a medical examination at a government hospital within 72 hours if applicable",
            "The statement must be recorded by a woman police officer (mandatory for sexual offences)",
            "Victim's identity is protected by law and cannot be disclosed",
            "Contact nearest Legal Services Authority for free legal representation",
        ],
        what_to_prove=[
            "Victim's statement is crucial and can be sole basis for conviction",
            "Medical evidence (if applicable)",
            "Any corroborating evidence: messages, CCTV, witnesses",
        ],
        what_to_expect="Rape (BNS 63): minimum 10 years to life imprisonment. Sexual harassment (BNS 75): up to 3 years. Stalking (BNS 78): up to 3 years for first offence, 5 years for repeat. Trial in fast-track court.",
    ),
    ("criminal", "criminal_general"): ActionPlan(
        domain="criminal",
        situation_type="criminal_general",
        title="General Criminal Matter / Filing FIR",
        applicable_laws=[
            "Bharatiya Nyaya Sanhita (BNS), 2023",
            "Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023 — Criminal Procedure",
            "Constitution of India, Article 21 — Right to life and liberty",
        ],
        helplines=[
            {"name": "Police Emergency", "number": "112"},
            {"name": "NALSA (Free Legal Aid)", "number": "15100"},
        ],
        filing_body="Nearest Police Station",
        filing_fee="Free",
        time_limit="Varies by offence; report as soon as possible",
        steps=[
            "Go to the nearest police station and file a First Information Report (FIR)",
            "You can file an FIR at ANY police station in India (Zero FIR provision under BNSS)",
            "Provide details: what happened, when, where, who was involved",
            "Get a copy of the FIR — this is your legal right",
            "If police refuse to register FIR, send a written complaint to the Superintendent of Police",
            "Alternatively, approach the Magistrate under BNSS Section 175(3) to direct FIR registration",
        ],
        what_to_prove=[
            "Facts of the offence (what happened)",
            "Any evidence available (documents, photos, witnesses)",
            "Identity of the accused (if known)",
        ],
        what_to_expect="Police must register FIR for cognizable offences. Investigation follows, leading to chargesheet if evidence supports. Non-cognizable offences require Magistrate's order for investigation.",
    ),
    # -----------------------------------------------------------------------
    # CONSUMER
    # -----------------------------------------------------------------------
    ("consumer", "defective_product"): ActionPlan(
        domain="consumer",
        situation_type="defective_product",
        title="Defective Product / Goods Complaint",
        applicable_laws=[
            "Consumer Protection Act, 2019 — Section 2(6) (Defect in goods)",
            "Consumer Protection Act, 2019 — Section 35 (Jurisdiction of District Commission)",
            "Consumer Protection Act, 2019 — Section 38 (Manner of filing complaint)",
            "Consumer Protection Act, 2019 — Section 39 (Admissibility of complaint)",
        ],
        helplines=[
            {"name": "National Consumer Helpline", "number": "1800-11-4000"},
            {"name": "NCH SMS", "number": "8800-001-915"},
        ],
        filing_body="District Consumer Disputes Redressal Commission",
        filing_fee="₹100 for claims up to ₹5 lakh; ₹200 for ₹5L–₹10L; ₹400 for ₹10L–₹20L; ₹500 for ₹20L–₹50L",
        time_limit="Within 2 years from the date of cause of action",
        steps=[
            "Send a written complaint (email/registered post) to the seller/manufacturer demanding resolution",
            "Keep proof of communication (delivery receipt, email confirmation)",
            "If unresolved within 15 days, file a consumer complaint",
            "File online at edaakhil.nic.in (e-Daakhil portal) or in person at the District Consumer Commission",
            "Attach: proof of purchase (bill/invoice), product photos, communication with seller, warranty card",
            "Call National Consumer Helpline 1800-11-4000 (toll-free) for guidance",
        ],
        what_to_prove=[
            "Proof of purchase (bill, invoice, online order confirmation)",
            "Nature of defect (photos, videos, expert opinion if needed)",
            "Communication with seller showing their failure to resolve",
            "Loss or damage suffered due to the defect",
        ],
        what_to_expect="District Commission must admit or reject within 21 days. Hearing within 3-5 months typically. Remedies include: replacement, repair, refund, compensation for mental agony. Orders are enforceable like a decree.",
        online_portals=["edaakhil.nic.in", "consumerhelpline.gov.in"],
    ),
    ("consumer", "service_deficiency"): ActionPlan(
        domain="consumer",
        situation_type="service_deficiency",
        title="Deficiency in Service",
        applicable_laws=[
            "Consumer Protection Act, 2019 — Section 2(11) (Deficiency in service)",
            "Consumer Protection Act, 2019 — Section 35-38",
        ],
        helplines=[
            {"name": "National Consumer Helpline", "number": "1800-11-4000"},
        ],
        filing_body="District Consumer Disputes Redressal Commission",
        filing_fee="₹100 for claims up to ₹5 lakh",
        time_limit="Within 2 years from the date of cause of action",
        steps=[
            "Document the service deficiency with evidence (bills, correspondence, photos)",
            "Send formal written complaint to the service provider",
            "If unresolved, file complaint at edaakhil.nic.in or District Consumer Commission",
            "You can also file at consumerhelpline.gov.in for mediation",
        ],
        what_to_prove=[
            "Contractual or implied obligation to provide the service",
            "Nature and extent of the deficiency",
            "Financial loss or inconvenience caused",
        ],
        what_to_expect="Similar to defective product complaints. Commission can award compensation, direct performance, or refund. Typical resolution: 3-6 months.",
        online_portals=["edaakhil.nic.in", "consumerhelpline.gov.in"],
    ),
    ("consumer", "unfair_trade"): ActionPlan(
        domain="consumer",
        situation_type="unfair_trade",
        title="Unfair Trade Practice / Misleading Advertisement",
        applicable_laws=[
            "Consumer Protection Act, 2019 — Section 2(47) (Unfair trade practice)",
            "Consumer Protection Act, 2019 — Section 18 (Central Consumer Protection Authority)",
            "Consumer Protection Act, 2019 — Section 21 (Power to issue directions against false advertisements)",
        ],
        helplines=[
            {"name": "National Consumer Helpline", "number": "1800-11-4000"},
        ],
        filing_body="District Consumer Commission or CCPA (Central Consumer Protection Authority)",
        filing_fee="₹100 for claims up to ₹5 lakh",
        time_limit="Within 2 years from the date of cause of action",
        steps=[
            "Collect evidence of the misleading claims (screenshots, brochures, ads)",
            "File complaint at District Consumer Commission or edaakhil.nic.in",
            "For systematic unfair practices, also complain to CCPA at ccpa.gov.in",
            "Report false ads to Advertising Standards Council of India (ASCI) at ascionline.in",
        ],
        what_to_prove=[
            "False or misleading representation made by the trader/advertiser",
            "Reliance on the representation in purchasing decision",
            "Loss or damage suffered",
        ],
        what_to_expect="CCPA can impose penalties up to ₹10 lakh on manufacturer and ₹50 lakh for repeat offence. Consumer Commission can award compensation. False endorsers can also be held liable.",
        online_portals=["edaakhil.nic.in", "ccpa.gov.in", "ascionline.in"],
    ),
    # -----------------------------------------------------------------------
    # CONSTITUTIONAL
    # -----------------------------------------------------------------------
    ("constitutional", "rti_denial"): ActionPlan(
        domain="constitutional",
        situation_type="rti_denial",
        title="Right to Information (RTI) Application / Denial",
        applicable_laws=[
            "Right to Information Act, 2005",
            "Constitution of India, Article 19(1)(a) — Right to freedom of speech and expression",
        ],
        helplines=[
            {"name": "Central Information Commission", "number": "011-2658-2898"},
        ],
        filing_body="Public Information Officer (PIO) of the concerned government office",
        filing_fee="₹10 by postal order, demand draft, or court fee stamp (₹0 for BPL applicants)",
        time_limit="PIO must respond within 30 days (48 hours if life/liberty involved)",
        steps=[
            "Write an RTI application addressed to the PIO of the concerned government office",
            "Clearly state what information you need — be specific and precise",
            "Attach ₹10 fee via postal order, DD, or court fee stamp (BPL applicants are exempt)",
            "Send via registered post or submit in person (get acknowledgment receipt)",
            "PIO must respond within 30 days",
            "If denied or no response: file First Appeal to the First Appellate Authority within 30 days",
            "If First Appeal fails: file Second Appeal to Central/State Information Commission within 90 days",
            "Online RTI can be filed for Central Government at rtionline.gov.in",
        ],
        what_to_prove=[
            "The information sought is held by a public authority",
            "You have filed a valid application with the required fee",
            "The PIO has failed to respond or denied without valid exemption",
        ],
        what_to_expect="PIO must respond within 30 days. First Appeal decided within 30-45 days. Second Appeal by Information Commission can impose a penalty of ₹250/day on errant PIO, up to ₹25,000. Most RTI applications are resolved within 2-3 months.",
        online_portals=["rtionline.gov.in"],
    ),
    ("constitutional", "fundamental_rights"): ActionPlan(
        domain="constitutional",
        situation_type="fundamental_rights",
        title="Violation of Fundamental Rights",
        applicable_laws=[
            "Constitution of India, Part III — Fundamental Rights (Articles 14-32)",
            "Constitution of India, Article 32 — Right to Constitutional Remedies (Supreme Court)",
            "Constitution of India, Article 226 — Power of High Courts to issue writs",
        ],
        helplines=[
            {"name": "NALSA (Free Legal Aid)", "number": "15100"},
            {"name": "National Human Rights Commission", "number": "011-2338-5368"},
        ],
        filing_body="High Court (Article 226) or Supreme Court (Article 32) via writ petition",
        filing_fee="Varies by court; free legal aid available through District Legal Services Authority",
        time_limit="No fixed limitation for writ petitions, but undue delay can be a ground for dismissal",
        steps=[
            "Document the rights violation with evidence",
            "Approach the District Legal Services Authority (DLSA) for free legal aid if needed — call NALSA 15100",
            "A lawyer can file a writ petition in the High Court under Article 226",
            "For fundamental rights violations by the State, Article 32 petition can be filed in Supreme Court",
            "Types of writs: Habeas Corpus (illegal detention), Mandamus (compel public duty), Certiorari, Prohibition, Quo Warranto",
            "National/State Human Rights Commission can also be approached for human rights violations",
        ],
        what_to_prove=[
            "State action or action by a public authority that violates fundamental rights",
            "The specific fundamental right affected",
            "How the action is arbitrary, unreasonable, or discriminatory",
        ],
        what_to_expect="High Court writ petitions can provide urgent interim relief. Final disposal may take months to years depending on complexity. Supreme Court Article 32 petitions treated as fundamental right itself and heard urgently.",
    ),
    # -----------------------------------------------------------------------
    # FAMILY
    # -----------------------------------------------------------------------
    ("family", "divorce"): ActionPlan(
        domain="family",
        situation_type="divorce",
        title="Divorce Proceedings",
        applicable_laws=[
            "Hindu Marriage Act, 1955 — Sections 13, 13B (divorce, mutual consent)",
            "Special Marriage Act, 1954 — Section 27 (divorce)",
            "Muslim Personal Law — Talaq, Khula, Mubarat",
            "Indian Divorce Act, 1869 (for Christians)",
        ],
        helplines=[
            {"name": "Women Helpline", "number": "181"},
            {"name": "NALSA (Free Legal Aid)", "number": "15100"},
        ],
        filing_body="Family Court or District Court",
        filing_fee="Varies by state; typically ₹500–₹2,000 for filing petition",
        time_limit="Mutual consent: 6 months cooling period (can be waived). Contested: varies by ground",
        steps=[
            "Consult a family lawyer to understand grounds applicable to your situation",
            "For mutual consent divorce: both parties file joint petition, 6-month cooling period, then second motion",
            "For contested divorce: file petition on grounds under applicable personal law (cruelty, desertion, adultery, etc.)",
            "Attempt mediation (court-mandated in many cases)",
            "If unable to afford a lawyer, contact DLSA for free legal aid — call 15100",
        ],
        what_to_prove=[
            "Valid marriage (marriage certificate or equivalent proof)",
            "Grounds for divorce as per applicable personal law",
            "For mutual consent: both parties agree, lived separately for 1+ year",
        ],
        what_to_expect="Mutual consent divorce: 6-18 months. Contested divorce: 1-5 years. Court encourages mediation and reconciliation before granting decree.",
    ),
    ("family", "maintenance"): ActionPlan(
        domain="family",
        situation_type="maintenance",
        title="Maintenance / Alimony",
        applicable_laws=[
            "BNSS Section 144 (earlier CrPC Section 125) — Maintenance of wives, children and parents",
            "Hindu Adoptions and Maintenance Act, 1956",
            "Protection of Women from Domestic Violence Act, 2005 — Section 20",
        ],
        helplines=[
            {"name": "Women Helpline", "number": "181"},
            {"name": "NALSA (Free Legal Aid)", "number": "15100"},
        ],
        filing_body="Magistrate's Court (Section 144 BNSS) or Family Court",
        filing_fee="Nominal; free for destitute applicants",
        time_limit="Can be filed anytime during subsistence of the right",
        steps=[
            "File application under BNSS Section 144 (maintenance for wife, children, or parents)",
            "Alternatively, claim maintenance under DV Act Section 20 along with protection order",
            "Provide evidence of spouse's income and your financial need",
            "Court can grant interim maintenance pending final hearing",
            "Contact DLSA (15100) for free legal aid if needed",
        ],
        what_to_prove=[
            "Marriage or relationship giving rise to maintenance obligation",
            "Inability to maintain yourself",
            "Spouse/relative has sufficient means",
            "Income proof of the respondent (salary slips, tax returns, bank statements)",
        ],
        what_to_expect="Magistrate must decide interim maintenance within 60 days. Final order within 6-12 months typically. Maintenance amount depends on spouse's income and applicant's needs. Non-payment can lead to imprisonment.",
    ),
    # -----------------------------------------------------------------------
    # LABOUR
    # -----------------------------------------------------------------------
    ("labour", "wrongful_termination"): ActionPlan(
        domain="labour",
        situation_type="wrongful_termination",
        title="Wrongful Termination / Illegal Dismissal",
        applicable_laws=[
            "Industrial Disputes Act, 1947 — Section 25F, 25N (retrenchment conditions)",
            "Standing Orders under Industrial Employment (Standing Orders) Act, 1946",
            "Shops and Establishments Act (state-specific)",
        ],
        helplines=[
            {"name": "Labour Commissioner Helpline", "number": "1800-202-0202"},
            {"name": "NALSA (Free Legal Aid)", "number": "15100"},
        ],
        filing_body="Labour Commissioner / Industrial Tribunal / Labour Court",
        filing_fee="Usually free or nominal for workers",
        time_limit="Raise industrial dispute within 3 years; approach Labour Court within 45 days of termination order",
        steps=[
            "Request a written termination letter with reasons from your employer",
            "File a complaint with the Labour Commissioner / Labour Department of your state",
            "The Labour Commissioner will first attempt conciliation between you and the employer",
            "If conciliation fails, the matter is referred to the Industrial Tribunal / Labour Court",
            "You can also approach the civil court for breach of employment contract",
            "Contact NALSA (15100) for free legal aid",
        ],
        what_to_prove=[
            "Employment relationship (offer letter, salary slips, ID card, emails)",
            "Terms of employment violated (notice period not given, no valid reason)",
            "If applicable: employer did not follow Industrial Disputes Act procedure for retrenchment",
        ],
        what_to_expect="Conciliation within 30-45 days. If employer terminated without following procedure (no notice, no retrenchment compensation), reinstatement with back wages may be ordered. Labour courts typically take 1-2 years.",
    ),
    ("labour", "unpaid_wages"): ActionPlan(
        domain="labour",
        situation_type="unpaid_wages",
        title="Unpaid Wages / Salary Dues",
        applicable_laws=[
            "Payment of Wages Act, 1936",
            "Minimum Wages Act, 1948",
            "Payment of Bonus Act, 1965",
            "Employees Provident Fund and Miscellaneous Provisions Act, 1952",
        ],
        helplines=[
            {"name": "Labour Commissioner Helpline", "number": "1800-202-0202"},
            {"name": "EPFO Helpline", "number": "1800-118-005"},
        ],
        filing_body="Labour Commissioner / Authority under Payment of Wages Act",
        filing_fee="Free",
        time_limit="Claim must be filed within 1 year of wage becoming due",
        steps=[
            "Send a formal written demand letter to your employer for pending wages",
            "File a complaint with the Labour Commissioner of your district",
            "For PF issues: file complaint at epfigms.gov.in or call EPFO helpline 1800-118-005",
            "For minimum wage violations: complaint to the Inspector under Minimum Wages Act",
            "If employer does not comply, the Labour Authority can impose penalties and direct payment",
        ],
        what_to_prove=[
            "Employment relationship and terms (offer letter, contract)",
            "Wages due and not paid (salary slips, bank statements showing non-payment)",
            "Period for which wages are due",
        ],
        what_to_expect="Labour Authority must decide within 90 days. Employer can be directed to pay wages with compensation (up to 10x the unpaid amount in penalties). Willful non-payment can lead to imprisonment up to 6 months.",
        online_portals=["epfigms.gov.in", "shramsuvidha.gov.in"],
    ),
    # -----------------------------------------------------------------------
    # PROPERTY
    # -----------------------------------------------------------------------
    ("property", "tenant_rights"): ActionPlan(
        domain="property",
        situation_type="tenant_rights",
        title="Tenant Rights / Illegal Eviction",
        applicable_laws=[
            "State-specific Rent Control Act (varies by state)",
            "Transfer of Property Act, 1882 — Section 106, 108",
            "Model Tenancy Act, 2021 (adopted by some states)",
        ],
        helplines=[
            {"name": "Police Emergency (for forceful eviction)", "number": "112"},
            {"name": "NALSA (Free Legal Aid)", "number": "15100"},
        ],
        filing_body="Rent Authority / Rent Court (as per state law) or Civil Court",
        filing_fee="Varies by state; typically ₹500–₹2,000",
        time_limit="Varies; generally respond within 15 days of receiving eviction notice",
        steps=[
            "Check if your state's Rent Control Act or Model Tenancy Act applies",
            "If receiving eviction notice: verify it follows legal procedure (written notice, valid grounds)",
            "Landlord CANNOT forcibly evict — must get court order. If forcibly evicted, call 112",
            "File response to eviction petition in the Rent Court within the specified time",
            "If no written agreement exists, collect evidence of tenancy (rent receipts, bank transfers, utility bills in your name)",
            "For disputes, approach the Rent Authority or Civil Court",
        ],
        what_to_prove=[
            "Tenancy/lease relationship (rental agreement, rent receipts, bank transfers)",
            "Compliance with rent payment obligations",
            "If challenging eviction: that proper legal procedure was not followed",
        ],
        what_to_expect="Landlord must follow due process — forcible eviction is illegal. Rent court proceedings may take 6-12 months. Security deposit must be returned within specified period (usually 1-2 months after vacating).",
    ),
    ("property", "land_dispute"): ActionPlan(
        domain="property",
        situation_type="land_dispute",
        title="Land / Property Dispute",
        applicable_laws=[
            "Registration Act, 1908",
            "Transfer of Property Act, 1882",
            "Indian Succession Act, 1925",
            "Hindu Succession Act, 1956 (if applicable)",
            "Specific Relief Act, 1963 — Section 34 (declaration of title)",
        ],
        helplines=[
            {"name": "NALSA (Free Legal Aid)", "number": "15100"},
        ],
        filing_body="Civil Court / Revenue Court (depending on nature of dispute)",
        filing_fee="Based on suit value (ad valorem court fees)",
        time_limit="Title suit: 12 years; Possession suit: 12 years; Specific performance: 3 years",
        steps=[
            "Verify property documents: sale deed, title deed, encumbrance certificate, mutation records",
            "Obtain Encumbrance Certificate from Sub-Registrar to check for existing claims",
            "If encroachment: file a complaint with the local revenue authority (Tehsildar/Patwari)",
            "For partition of property: file a partition suit in Civil Court",
            "For title dispute: file a declaration and injunction suit in Civil Court",
            "Consult a property lawyer before taking any action — documents review is essential",
        ],
        what_to_prove=[
            "Clear title to the property (chain of title documents)",
            "Possession history (utility bills, tax receipts, witnesses)",
            "Nature of the other party's claim and why it is invalid",
        ],
        what_to_expect="Civil suits can take 2-10 years depending on complexity. Interim injunctions can prevent further encroachment/transfer. Revenue court proceedings are faster (6-12 months).",
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_action_plan(domain: str, situation_type: str) -> ActionPlan | None:
    """Look up a deterministic action plan by domain and situation type."""
    return _ACTION_PLANS.get((domain, situation_type))


def get_all_plans() -> dict[tuple[str, str], ActionPlan]:
    """Return all action plans (for Delta Lake export in Phase 2)."""
    return dict(_ACTION_PLANS)


def format_action_context(plan: ActionPlan) -> str:
    """Format an action plan as context string to inject into LLM prompt."""
    helplines_str = "\n".join(
        f"  - {h['name']}: {h['number']}" for h in plan.helplines
    )
    steps_str = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan.steps))
    prove_str = "\n".join(f"  - {p}" for p in plan.what_to_prove)
    portals_str = ""
    if plan.online_portals:
        portals_str = "\nOnline portals: " + ", ".join(plan.online_portals)

    return (
        f"=== ACTION PLAN: {plan.title} ===\n"
        f"Applicable Laws: {'; '.join(plan.applicable_laws)}\n"
        f"Filing Body: {plan.filing_body}\n"
        f"Filing Fee: {plan.filing_fee}\n"
        f"Time Limit: {plan.time_limit}\n"
        f"Helplines:\n{helplines_str}\n"
        f"Steps:\n{steps_str}\n"
        f"What to prove/document:\n{prove_str}\n"
        f"What to expect: {plan.what_to_expect}"
        f"{portals_str}\n"
    )
