import os
import io
import json
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document

from openai import OpenAI


# =========================
# 1) File -> Text
# =========================
def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for p in reader.pages:
        pages.append((p.extract_text() or "").strip())
    return "\n".join([p for p in pages if p]).strip()


def read_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    paras = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paras).strip()


def file_to_text(uploaded_file) -> str:
    file_bytes = uploaded_file.read()
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return read_pdf(file_bytes)
    if name.endswith(".docx"):
        return read_docx(file_bytes)
    raise ValueError("Only PDF and DOCX are supported.")


# =========================
# 2) Basic Privacy Masking (recommended at booth)
# =========================
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")

def anonymize_text(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = URL_RE.sub("[URL]", text)
    return text


# =========================
# 3) Enterprise-style Criteria Presets + Keyword Bank
# =========================
@dataclass
class Weights:
    ats_keyword_alignment: int = 35
    role_requirements: int = 25
    impact_evidence: int = 15
    resume_completeness: int = 15
    timeline_clarity: int = 10

    def total(self) -> int:
        return (
            self.ats_keyword_alignment
            + self.role_requirements
            + self.impact_evidence
            + self.resume_completeness
            + self.timeline_clarity
        )


GENERAL_ACTION_VERBS = [
    "led", "managed", "owned", "improved", "increased", "reduced", "delivered", "designed",
    "developed", "implemented", "launched", "optimized", "streamlined", "automated",
    "analyzed", "collaborated", "negotiated", "executed", "aligned", "coordinated",
    "drove", "scaled", "supported"
]

# Generic “enterprise ATS” keyword clusters (broadly applicable)
ENTERPRISE_KEYWORD_BANK = {
    "business_core": [
        "stakeholder management", "cross-functional", "communication", "presentation",
        "problem solving", "process improvement", "continuous improvement",
        "operational excellence", "risk management", "project management",
        "change management", "strategy", "planning", "execution", "governance"
    ],
    "data_tools": [
        "excel", "power bi", "tableau", "sql", "python", "google sheets",
        "data analysis", "dashboard", "reporting", "analytics", "kpi", "okrs"
    ],
    "pm_tools": [
        "jira", "confluence", "asana", "trello", "clickup", "notion",
        "scrum", "agile", "kanban"
    ],
    "people_leadership": [
        "leadership", "coaching", "mentoring", "team management",
        "performance management", "hiring", "talent acquisition",
        "interviewing", "training"
    ],
    "customer_commercial": [
        "customer experience", "client management", "account management",
        "sales", "business development", "campaign", "marketing",
        "crm", "retention", "growth"
    ]
}

# Booth-friendly presets (choose quickly)
PRESETS = {
    "General Corporate (All-rounder)": {
        "must_have": ["stakeholder management", "cross-functional", "communication", "problem solving"],
        "nice_to_have": ["project management", "process improvement", "excel", "kpi"],
        "keyword_groups": ["business_core", "data_tools", "pm_tools"]
    },
    "Data / Analytics": {
        "must_have": ["data analysis", "sql", "dashboard", "reporting"],
        "nice_to_have": ["python", "power bi", "tableau", "statistics"],
        "keyword_groups": ["data_tools", "business_core"]
    },
    "Project / Product / Operations": {
        "must_have": ["project management", "cross-functional", "process improvement", "execution"],
        "nice_to_have": ["jira", "agile", "risk management", "kpi"],
        "keyword_groups": ["pm_tools", "business_core", "data_tools"]
    },
    "Marketing / Commercial": {
        "must_have": ["campaign", "communication", "stakeholder management"],
        "nice_to_have": ["crm", "growth", "analytics", "reporting"],
        "keyword_groups": ["customer_commercial", "business_core", "data_tools"]
    },
    "People / HR / TA": {
        "must_have": ["talent acquisition", "interviewing", "stakeholder management", "communication"],
        "nice_to_have": ["performance management", "training", "coaching", "project management"],
        "keyword_groups": ["people_leadership", "business_core", "pm_tools"]
    },
}


# =========================
# 4) Heuristics: Missing Sections, Metrics, Keyword Hits
# =========================
SECTION_PATTERNS = {
    "contact": re.compile(r"\b(email|phone|linkedin|address|contact)\b", re.IGNORECASE),
    "summary": re.compile(r"\b(summary|profile|about me|professional summary|objective)\b", re.IGNORECASE),
    "experience": re.compile(r"\b(experience|work history|employment|professional experience)\b", re.IGNORECASE),
    "education": re.compile(r"\b(education|academic|university|college|bachelor|master|phd)\b", re.IGNORECASE),
    "skills": re.compile(r"\b(skills|technical skills|competencies|tools)\b", re.IGNORECASE),
    "projects": re.compile(r"\b(projects|project experience)\b", re.IGNORECASE),
    "certifications": re.compile(r"\b(certification|certificate|licensed)\b", re.IGNORECASE),
}

METRIC_PATTERN = re.compile(r"(\b\d+%|\b\d+\s*(k|m|b)\b|\b\d{1,3}(,\d{3})+\b|\b\d+\b)", re.IGNORECASE)

def detect_missing_sections(resume_text: str) -> Tuple[List[str], Dict[str, bool]]:
    present = {name: bool(pat.search(resume_text)) for name, pat in SECTION_PATTERNS.items()}
    missing = [k for k, v in present.items() if not v]
    return missing, present

def count_metrics(resume_text: str) -> int:
    return len(METRIC_PATTERN.findall(resume_text))

def count_action_verbs(resume_text: str) -> int:
    text = resume_text.lower()
    return sum(text.count(v) for v in GENERAL_ACTION_VERBS)

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def keyword_hits(resume_text: str, keywords: List[str]) -> Dict[str, int]:
    text = normalize(resume_text)
    hits = {}
    for kw in keywords:
        k = normalize(kw)
        hits[kw] = text.count(k) if k else 0
    return hits


# =========================
# 5) OpenAI (Agentic) helpers
# =========================
def get_client() -> Optional[OpenAI]:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    return OpenAI(api_key=key)

def llm_json(client: OpenAI, model: str, instructions: str, user_input: str, schema_hint: str) -> Dict[str, Any]:
    prompt = f"""
Return STRICT JSON only. No markdown. No extra text.

Schema hint:
{schema_hint}

User input:
{user_input}
""".strip()

    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=prompt,
    )
    text = resp.output_text.strip()
    text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    return json.loads(text)

def agent_parse_resume(client: OpenAI, model: str, resume_text: str) -> Dict[str, Any]:
    instructions = (
        "You are an HR resume parser. Extract only what is supported by the resume text. "
        "If uncertain, use null or empty lists. Avoid protected/sensitive attributes."
    )
    schema_hint = """
{
  "candidate_summary": "string (2-4 lines)",
  "current_or_latest_title": "string|null",
  "years_experience_estimate": "number|null",
  "industries": ["string"],
  "skills": ["string"],
  "tools": ["string"],
  "work_history": [
    {"company":"string|null","title":"string|null","start":"string|null","end":"string|null","highlights":["string"]}
  ],
  "education": [{"degree":"string|null","school":"string|null","year":"string|null"}],
  "certifications": ["string"],
  "evidence_snippets": ["string (5-10 short snippets copied from resume)"]
}
""".strip()
    return llm_json(client, model, instructions, f"RESUME TEXT:\n{resume_text}", schema_hint)

def agent_score_and_gap(
    client: OpenAI,
    model: str,
    parsed: Dict[str, Any],
    jd_text: str,
    must_haves: List[str],
    nice_to_haves: List[str],
    weights: Weights,
    heuristics: Dict[str, Any],
) -> Dict[str, Any]:
    instructions = (
        "You are an unbiased resume screener. Score job-fit using only job-related evidence. "
        "Do NOT use age, gender, photo, address, or school prestige. "
        "Every score must include evidence from the resume snippets or extracted fields."
    )
    schema_hint = """
{
  "overall_fit_score_0_100": "number",
  "dimension_scores_0_100": {
    "ats_keyword_alignment": "number",
    "role_requirements": "number",
    "impact_evidence": "number",
    "resume_completeness": "number",
    "timeline_clarity": "number"
  },
  "must_have_match": [{"requirement":"string","score_0_5":"number","evidence":"string"}],
  "nice_to_have_match": [{"requirement":"string","score_0_5":"number","evidence":"string"}],
  "strengths": ["string"],
  "gaps": ["string"],
  "missing_sections": ["string"],
  "recommendation": {"decision":"Proceed|Hold|Reject","rationale":"string"}
}
""".strip()

    payload = {
        "job_description": jd_text,
        "weights_total_should_be_100": weights.total(),
        "weights": weights.__dict__,
        "must_haves": must_haves,
        "nice_to_haves": nice_to_haves,
        "parsed_resume": parsed,
        "heuristics": heuristics,
        "scoring_rules": [
            "Use evidence-based evaluation only.",
            "If no evidence is found for a requirement, say 'Not found in resume' and score low.",
            "Do not infer skills not stated."
        ]
    }
    return llm_json(client, model, instructions, json.dumps(payload, ensure_ascii=False), schema_hint)

def agent_interview_kit(client: OpenAI, model: str, jd_text: str, scoring: Dict[str, Any]) -> Dict[str, Any]:
    instructions = (
        "Create an interview kit to validate must-haves and close the identified gaps. "
        "Use professional, concise questions. Include what a good answer looks like."
    )
    schema_hint = """
{
  "interview_questions": [
    {"competency":"string","question":"string","what_good_looks_like":"string"}
  ],
  "quick_case_optional": {
    "title":"string|null",
    "prompt":"string|null",
    "timebox_minutes":"number|null",
    "evaluation_criteria":["string"]
  }
}
""".strip()
    return llm_json(
        client, model, instructions,
        json.dumps({"job_description": jd_text, "scoring": scoring}, ensure_ascii=False),
        schema_hint
    )


# =========================
# 6) Streamlit UI
# =========================
st.set_page_config(page_title="Job Fair Resume Agent", layout="wide")
st.title("🎪 Job Fair Resume Agent — Upload CV → ATS checks → Score → Gaps → Interview Kit")

with st.sidebar:
    st.header("Settings")
    model = st.text_input("Model", value="gpt-5")
    anonymize = st.checkbox("Anonymize email/phone/URLs before sending to AI", value=True)

    st.divider()
    preset_name = st.selectbox("Preset criteria", list(PRESETS.keys()), index=0)
    preset = PRESETS[preset_name]

    st.divider()
    st.subheader("Scoring Weights (sum should be 100)")
    w1 = st.slider("ATS Keyword Alignment", 0, 100, 35)
    w2 = st.slider("Role Requirements Match", 0, 100, 25)
    w3 = st.slider("Impact Evidence (metrics/results)", 0, 100, 15)
    w4 = st.slider("Resume Completeness (sections)", 0, 100, 15)
    w5 = st.slider("Timeline Clarity", 0, 100, 10)
    weights = Weights(w1, w2, w3, w4, w5)
    if weights.total() != 100:
        st.warning(f"Current weight sum = {weights.total()} (recommended 100)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1) Upload Resume")
    uploaded = st.file_uploader("PDF/DOCX", type=["pdf", "docx"])

    st.subheader("2) Paste Job Description")
    jd_text = st.text_area("Job Description", height=240, placeholder="Paste responsibilities + requirements here...")

with col2:
    st.subheader("3) Must-have / Nice-to-have (editable)")
    must_text = st.text_area("Must-have (1 per line)", value="\n".join(preset["must_have"]), height=140)
    nice_text = st.text_area("Nice-to-have (1 per line)", value="\n".join(preset["nice_to_have"]), height=140)

    st.subheader("4) Enterprise ATS Keyword Bank (auto-added)")
    groups = preset["keyword_groups"]
    st.caption("Keyword bank groups used: " + ", ".join(groups))

run = st.button("🚀 Run Agentic Screening", type="primary", use_container_width=True)

if run:
    if not uploaded or not jd_text.strip():
        st.error("Please upload a resume and paste a job description.")
        st.stop()

    client = get_client()
    if client is None:
        st.error("OPENAI_API_KEY is not set.")
        st.info("On Streamlit Cloud: App settings → Secrets → add OPENAI_API_KEY='your_key'")
        st.stop()

    try:
        resume_text = file_to_text(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    resume_text_for_ai = anonymize_text(resume_text) if anonymize else resume_text

    must_haves = [x.strip() for x in must_text.splitlines() if x.strip()]
    nice_to_haves = [x.strip() for x in nice_text.splitlines() if x.strip()]

    # Build keyword bank for heuristic matching
    bank_keywords = []
    for g in groups:
        bank_keywords.extend(ENTERPRISE_KEYWORD_BANK.get(g, []))

    # Heuristics
    missing, present_map = detect_missing_sections(resume_text)
    metrics_count = count_metrics(resume_text)
    verb_count = count_action_verbs(resume_text)

    must_hits = keyword_hits(resume_text, must_haves) if must_haves else {}
    nice_hits = keyword_hits(resume_text, nice_to_haves) if nice_to_haves else {}
    bank_hits = keyword_hits(resume_text, bank_keywords)

    heuristics = {
        "missing_sections_detected": missing,
        "sections_present_map": present_map,
        "metrics_count_estimate": metrics_count,
        "action_verb_count_estimate": verb_count,
        "must_have_keyword_hits": must_hits,
        "nice_to_have_keyword_hits": nice_hits,
        "enterprise_bank_top_hits": sorted(bank_hits.items(), key=lambda x: x[1], reverse=True)[:20],
        "notes": [
            "Keyword hits use simple substring matching for speed in booth settings.",
            "Recruiter should make final decisions; this is decision support."
        ]
    }

    # Agentic pipeline
    with st.spinner("Agent 1/3: Parsing resume..."):
        parsed = agent_parse_resume(client, model, resume_text_for_ai)

    with st.spinner("Agent 2/3: Scoring + gaps + missing sections..."):
        scoring = agent_score_and_gap(client, model, parsed, jd_text, must_haves, nice_to_haves, weights, heuristics)

    with st.spinner("Agent 3/3: Interview kit..."):
        kit = agent_interview_kit(client, model, jd_text, scoring)

    st.success("Done ✅")

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Parsed Resume (JSON)")
        st.json(parsed)

        st.subheader("ATS-style checks (heuristics)")
        st.write("**Missing sections:** " + (", ".join(missing) if missing else "None detected"))
        st.write(f"**Metrics count (approx):** {metrics_count}")
        st.write(f"**Action-verb count (approx):** {verb_count}")

        if must_hits:
            st.markdown("**Must-have keyword hits**")
            st.dataframe(pd.DataFrame([{"keyword": k, "hits": v} for k, v in must_hits.items()]), use_container_width=True)
        if nice_hits:
            st.markdown("**Nice-to-have keyword hits**")
            st.dataframe(pd.DataFrame([{"keyword": k, "hits": v} for k, v in nice_hits.items()]), use_container_width=True)

        st.markdown("**Enterprise keyword bank top hits**")
        st.dataframe(pd.DataFrame(heuristics["enterprise_bank_top_hits"], columns=["keyword", "hits"]), use_container_width=True)

    with right:
        st.subheader("Score & Recommendation")
        st.metric("Overall Fit Score (0-100)", scoring.get("overall_fit_score_0_100", "N/A"))

        dim = scoring.get("dimension_scores_0_100", {})
        if dim:
            st.markdown("**Dimension scores (0-100)**")
            st.dataframe(pd.DataFrame([dim]), use_container_width=True)

        rec = scoring.get("recommendation", {})
        st.write(f"**Decision:** {rec.get('decision','N/A')}")
        st.write(f"**Rationale:** {rec.get('rationale','')}")

        st.markdown("### Strengths")
        for s in scoring.get("strengths", [])[:10]:
            st.write(f"- {s}")

        st.markdown("### Gaps / Missing")
        for g in scoring.get("gaps", [])[:12]:
            st.write(f"- {g}")

        st.markdown("### Missing resume sections (model + heuristics)")
        for ms in scoring.get("missing_sections", [])[:12]:
            st.write(f"- {ms}")

        st.divider()
        st.subheader("Must-have Match (with evidence)")
        mh = scoring.get("must_have_match", [])
        if mh:
            st.dataframe(pd.DataFrame(mh), use_container_width=True)

        st.subheader("Nice-to-have Match (with evidence)")
        nh = scoring.get("nice_to_have_match", [])
        if nh:
            st.dataframe(pd.DataFrame(nh), use_container_width=True)

    st.divider()
    st.subheader("Interview Kit")
    qs = kit.get("interview_questions", [])
    if qs:
        for i, q in enumerate(qs[:12], start=1):
            st.write(f"**Q{i} — {q.get('competency','')}**: {q.get('question','')}")
            st.caption(f"What good looks like: {q.get('what_good_looks_like','')}")

    case = kit.get("quick_case_optional", {})
    if case and any(case.get(k) for k in ["title", "prompt"]):
        st.subheader("Optional quick case")
        st.write(f"**Title:** {case.get('title')}")
        st.write(f"**Timebox (minutes):** {case.get('timebox_minutes')}")
        st.write(case.get("prompt"))
        if case.get("evaluation_criteria"):
            for c in case["evaluation_criteria"]:
                st.write(f"- {c}")

    # Export
    export = {
        "file": uploaded.name,
        "preset": preset_name,
        "weights": weights.__dict__,
        "job_description": jd_text,
        "criteria": {"must_haves": must_haves, "nice_to_haves": nice_to_haves, "keyword_groups": groups},
        "heuristics": heuristics,
        "parsed_resume": parsed,
        "scoring": scoring,
        "interview_kit": kit
    }

    export_json = json.dumps(export, ensure_ascii=False, indent=2)
    st.download_button(
        "⬇️ Download Full Result (JSON)",
        data=export_json.encode("utf-8"),
        file_name=f"resume_screen_{os.path.splitext(uploaded.name)[0]}.json",
        mime="application/json",
        use_container_width=True
    )

    tracking_row = {
        "resume_file": uploaded.name,
        "preset": preset_name,
        "fit_score": scoring.get("overall_fit_score_0_100"),
        "decision": scoring.get("recommendation", {}).get("decision"),
        "top_strengths": " | ".join(scoring.get("strengths", [])[:3]),
        "top_gaps": " | ".join(scoring.get("gaps", [])[:3]),
        "missing_sections": " | ".join(missing[:6]),
        "metrics_count_est": metrics_count,
        "action_verb_count_est": verb_count
    }
    df = pd.DataFrame([tracking_row])
    st.download_button(
        "⬇️ Download Booth Tracking (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="jobfair_tracking.csv",
        mime="text/csv",
        use_container_width=True
    )

    with st.expander("Debug: Resume text preview (first 20k chars)"):
        st.text_area("Resume text", resume_text[:20000], height=260)
