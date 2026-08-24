import re

import requests
import streamlit as st
from pathlib import Path


API_URL = "http://127.0.0.1:8000/diagnose-multimodal"
TOP_K = 4
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

st.set_page_config(
    page_title="Skin Disease Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
:root {
    --bg: #F8FAFC;
    --sidebar-bg: #F1F5F9;
    --surface: #FFFFFF;
    --border: #E2E8F0;
    --text: #1A202C;
    --text-muted: #64748B;
    --accent: #2563EB;
    --accent-dark: #1D4ED8;
    --header-bg: #0F3D5C;
    --danger: #B91C1C;
    --warning: #92400E;
    --warning-bg: #FEF3C7;
    --warning-border: #FDE68A;
    --success: #166534;
    --radius: 8px;
    --shadow: 0 1px 3px rgba(15, 23, 42, 0.07);
}

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.stApp { background-color: var(--bg) !important; }
.block-container { padding-top: 3.5rem; padding-bottom: 3rem; max-width: 1180px; }
h1, h2, h3, h4, h5, h6 { color: var(--text) !important; font-weight: 700; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: var(--sidebar-bg) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label { color: var(--text); }
[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

.brand-title { font-weight: 700; color: var(--text); font-size: 1.05rem; }
.brand-tagline { color: var(--text-muted); font-size: 0.8rem; margin-top: 0.2rem; }

.sidebar-heading {
    font-size: 0.72rem; font-weight: 700; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.8rem;
}

.steps-list { list-style: none; padding: 0; margin: 0; }
.steps-list li {
    display: flex; gap: 0.75rem; padding: 0.55rem 0;
    border-bottom: 1px solid var(--border);
}
.steps-list li:last-child { border-bottom: none; }
.step-index { font-weight: 700; color: var(--text-muted); font-variant-numeric: tabular-nums; min-width: 1.3rem; font-size: 0.82rem; }
.step-text { color: var(--text); font-size: 0.85rem; line-height: 1.4; }

.disclaimer {
    border-left: 3px solid var(--warning);
    background: var(--warning-bg);
    border-radius: 4px;
    padding: 0.75rem 0.9rem;
    color: var(--warning);
    font-size: 0.8rem;
    line-height: 1.5;
}
.disclaimer strong { color: var(--warning); }

/* Page header */
.app-header {
    background: var(--header-bg);
    padding: 1.75rem 2.1rem;
    border-radius: var(--radius);
    margin-bottom: 2rem;
    box-shadow: var(--shadow);
}
.eyebrow { font-size: 0.72rem; font-weight: 700; color: rgba(255,255,255,0.65); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; }
.app-title { font-size: 1.9rem; font-weight: 800; color: #fff; letter-spacing: -0.01em; }
.app-subtitle { font-size: 0.92rem; color: rgba(255,255,255,0.82); margin-top: 0.3rem; }

/* Bordered containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow);
}

.panel-title { font-size: 1rem; font-weight: 700; color: var(--text); margin-bottom: 0.2rem; }
.panel-subtitle { color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1.1rem; }

/* Widgets */
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface);
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--text-muted) !important; }

[data-testid="stTextArea"] textarea {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    background: var(--surface);
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: none !important;
}

.stButton > button {
    border-radius: var(--radius) !important;
    font-weight: 600 !important;
    padding: 0.55rem 1rem !important;
    border: none !important;
}
.stButton > button[kind="primary"] { background: var(--accent) !important; color: #fff !important; }
.stButton > button[kind="primary"]:hover { background: var(--accent-dark) !important; }

.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { border-radius: 0; padding: 0.5rem 1rem; color: var(--text-muted); font-weight: 600; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }

[data-testid="stAlert"] { border-radius: var(--radius) !important; }

.stTextArea, .stFileUploader, .stButton { margin-bottom: 1rem; }

/* Results */
.summary { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); padding: 0.9rem 1.1rem; margin-bottom: 1.25rem; }
.summary-label { font-size: 0.72rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.35rem; }
.summary-text { font-size: 0.92rem; color: var(--text); line-height: 1.6; }

.risk-line { font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.9rem; }
.risk-value { font-weight: 700; }
.risk-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }

.confidence-row { display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.35rem; }
.confidence-value { font-weight: 700; color: var(--text); }
.progress-track { background: var(--border); height: 6px; border-radius: 3px; overflow: hidden; margin-bottom: 1.1rem; }
.progress-fill { background: var(--accent); height: 100%; }

.other-matches { font-size: 0.85rem; color: var(--text-muted); margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border); }
.other-matches-label { color: var(--text); font-weight: 600; }

.detail-section { padding: 0.9rem 0; border-bottom: 1px solid var(--border); }
.detail-section:last-child { border-bottom: none; }
.detail-label { font-size: 0.72rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.35rem; }
.detail-text { font-size: 0.92rem; color: var(--text); line-height: 1.6; }

.empty-state { max-width: 360px; margin: 0.5rem auto; text-align: center; padding: 1.8rem 1.4rem; }
.empty-state-title { font-weight: 700; color: var(--text); font-size: 0.98rem; margin-bottom: 0.35rem; }
.empty-state-text { color: var(--text-muted); font-size: 0.84rem; line-height: 1.5; }

.similar-caption { font-size: 0.82rem; color: var(--text-muted); margin-top: 0.4rem; }
.similar-name { font-weight: 600; color: var(--text); }

.footer-note { text-align: center; color: var(--text-muted); font-size: 0.8rem; margin-top: 2.5rem; padding-top: 1.2rem; border-top: 1px solid var(--border); }
"""

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = None


def parse_sections(full_text: str) -> dict:
    """Split a knowledge-base markdown document into '## Header' -> content sections."""
    sections = {}
    for match in re.finditer(r"##\s*(.+?)\s*\n(.*?)(?=\n##|\Z)", full_text or "", re.DOTALL):
        header = match.group(1).strip().lower()
        sections[header] = match.group(2).strip()
    return sections


def risk_color_label(risk_text: str):
    text = (risk_text or "").lower()
    if "high" in text:
        return "var(--danger)", "High risk"
    if "moderate" in text:
        return "var(--warning)", "Moderate risk"
    if "low" in text:
        return "var(--success)", "Low risk"
    return "var(--text-muted)", "Risk unspecified"


def detail_section(label: str, content: str) -> str:
    return f'<div class="detail-section"><div class="detail-label">{label}</div><div class="detail-text">{content}</div></div>'


def local_image_path(image_path):
    """Resolve paths produced by the backend's image index for local display."""
    if not image_path:
        return None
    candidate = Path(image_path)
    for path in (candidate, BACKEND_DIR / candidate, PROJECT_ROOT / candidate):
        resolved = path.resolve()
        if resolved.is_file():
            return resolved
    return None


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(
        '<div class="brand-title">Skin Disease Assistant</div>'
        '<div class="brand-tagline">Reference-based skin condition assistant</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown('<div class="sidebar-heading">How it works</div>', unsafe_allow_html=True)
    steps = [
        "Upload a photo of the skin concern",
        "Describe the symptoms",
        "The system retrieves similar reference cases",
        "Review the analysis and next steps",
    ]
    steps_html = "".join(
        f'<li><span class="step-index">{i:02d}</span><span class="step-text">{step}</span></li>'
        for i, step in enumerate(steps, start=1)
    )
    st.markdown(f'<ul class="steps-list">{steps_html}</ul>', unsafe_allow_html=True)

    st.divider()
    st.markdown(
        '<div class="disclaimer"><strong>Not a medical diagnosis.</strong> This tool is for '
        "informational purposes only. Always consult a qualified healthcare professional.</div>",
        unsafe_allow_html=True,
    )

# ---------- Page header ----------
st.markdown(
    '<div class="app-header">'
    '<div class="eyebrow">Research prototype</div>'
    '<div class="app-title">Skin Disease Assistant</div>'
    '<div class="app-subtitle">Multimodal retrieval-augmented diagnostic support</div>'
    "</div>",
    unsafe_allow_html=True,
)

# ---------- Main area ----------
left_col, right_col = st.columns([1, 1.3], gap="large")

with left_col:
    with st.container(border=True):
        st.markdown('<div class="panel-title">New analysis</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-subtitle">Upload an image and/or describe the symptoms, '
            "then run the analysis.</div>",
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader("Upload a skin image (optional)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded image", width=260)

        symptoms = st.text_area(
            "Describe the symptoms (optional)",
            placeholder="e.g. itchy red patches on the forearm, dry and scaly",
            height=120,
        )

        submitted = st.button("Analyze", type="primary", use_container_width=True)

        if submitted:
            has_text = bool(symptoms.strip())
            has_image = uploaded_file is not None

            if not has_text and not has_image:
                st.session_state.error = "Please provide symptoms, an image, or both."
                st.session_state.result = None
            else:
                form_data = {"symptoms": symptoms, "top_k": TOP_K}
                files = {"_multipart": (None, "")}
                if has_image:
                    files["file"] = (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)

                with st.spinner("Running retrieval pipeline..."):
                    try:
                        response = requests.post(API_URL, data=form_data, files=files, timeout=60)
                        data = response.json()
                        if response.status_code != 200:
                            st.session_state.error = f"Backend error: {response.text}"
                            st.session_state.result = None
                        elif "error" in data:
                            st.session_state.error = data["error"]
                            st.session_state.result = None
                        else:
                            st.session_state.result = data
                            st.session_state.error = None
                    except requests.RequestException as exc:
                        st.session_state.error = f"Could not reach the backend: {exc}"
                        st.session_state.result = None

with right_col:
    if st.session_state.error:
        st.error(st.session_state.error)

    result = st.session_state.result

    if result is None:
        with st.container(border=True):
            st.markdown(
                '<div class="empty-state"><div class="empty-state-title">No analysis yet</div>'
                '<div class="empty-state-text">Submit an image or describe symptoms on the left '
                "to see the diagnosis here.</div></div>",
                unsafe_allow_html=True,
            )
    else:
        candidates = result.get("candidates", [])
        top = candidates[0] if candidates else None

        tab_diagnosis, tab_details, tab_similar = st.tabs(["Diagnosis", "Details", "Similar Cases"])

        # ---- Diagnosis tab ----
        with tab_diagnosis:
            if result.get("final_explanation"):
                st.markdown(
                    '<div class="summary"><div class="summary-label">AI summary</div>'
                    f'<div class="summary-text">{result["final_explanation"]}</div></div>',
                    unsafe_allow_html=True,
                )

            if top is None:
                st.warning("No confident matches were found for the provided input.")
            else:
                sections = parse_sections(top["description"])
                overview = sections.get("overview", top["description"])
                pct = top["final_score"] * 100
                risk_color, risk_label = risk_color_label(top["risk_level"])

                st.markdown(f"## {top['name']}")
                st.markdown(
                    f'<div class="risk-line"><span class="risk-dot" style="background:{risk_color};">'
                    f'</span>Risk level: '
                    f'<span class="risk-value" style="color:{risk_color};">{risk_label}</span> '
                    f'&mdash; {top["risk_level"]}</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="confidence-row"><span>Match confidence</span>'
                    f'<span class="confidence-value">{pct:.0f}%</span></div>'
                    f'<div class="progress-track"><div class="progress-fill" '
                    f'style="width:{pct:.0f}%;"></div></div>',
                    unsafe_allow_html=True,
                )

                st.markdown(detail_section("Description", overview), unsafe_allow_html=True)

                if len(candidates) > 1:
                    others = ", ".join(
                        f"{c['name']} ({c['final_score'] * 100:.0f}%)" for c in candidates[1:]
                    )
                    st.markdown(
                        f'<div class="other-matches"><span class="other-matches-label">'
                        f"Other possible matches:</span> {others}</div>",
                        unsafe_allow_html=True,
                    )

        # ---- Details tab ----
        with tab_details:
            if top is None:
                st.info("No candidate details to show yet.")
            else:
                sections = parse_sections(top["description"])

                st.markdown(f"#### {top['name']}")
                st.markdown(
                    detail_section("Symptoms matched", sections.get("skin symptoms", "Not specified.")),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    detail_section("Possible causes", sections.get("possible causes", "Not specified.")),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    detail_section("Risk groups", sections.get("risk groups", "Not specified.")),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    detail_section("Recommended next steps", top["next_steps"]),
                    unsafe_allow_html=True,
                )

                if len(candidates) > 1:
                    st.markdown("###### Other possible matches")
                    for candidate in candidates[1:]:
                        with st.expander(f"{candidate['name']} — {candidate['final_score'] * 100:.0f}% match"):
                            other_sections = parse_sections(candidate["description"])
                            st.markdown(
                                detail_section(
                                    "Symptoms matched", other_sections.get("skin symptoms", "Not specified.")
                                ),
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                detail_section(
                                    "Possible causes", other_sections.get("possible causes", "Not specified.")
                                ),
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                detail_section(
                                    "Risk groups", other_sections.get("risk groups", "Not specified.")
                                ),
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                detail_section("Recommended next steps", candidate["next_steps"]),
                                unsafe_allow_html=True,
                            )

        # ---- Similar Cases tab ----
        with tab_similar:
            image_candidates = [c for c in candidates if local_image_path(c.get("similar_image_path"))]

            if not image_candidates:
                st.info(
                    "No reference images retrieved. Upload a skin image to see visually similar cases."
                )
            else:
                cols_per_row = 3
                for row_start in range(0, len(image_candidates), cols_per_row):
                    row_items = image_candidates[row_start : row_start + cols_per_row]
                    row_cols = st.columns(cols_per_row)
                    for col, candidate in zip(row_cols, row_items):
                        with col:
                            image_path = local_image_path(candidate["similar_image_path"])
                            st.image(str(image_path), use_container_width=True)
                            st.markdown(
                                f'<div class="similar-caption"><span class="similar-name">'
                                f'{candidate["name"]}</span> — {candidate["image_score"] * 100:.0f}% similar</div>',
                                unsafe_allow_html=True,
                            )

st.markdown(
    '<div class="footer-note">Skin Disease Assistant — for research and educational purposes only.</div>',
    unsafe_allow_html=True,
)
