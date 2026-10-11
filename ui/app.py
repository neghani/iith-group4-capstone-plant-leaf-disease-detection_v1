import os

import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Plant Disease Assistant",
    page_icon="🌿",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem; }
    .sub-header { color: #4a5568; margin-bottom: 1.5rem; }
    .badge { display: inline-block; padding: 0.25rem 0.6rem; border-radius: 999px;
             background: #edf2f7; font-size: 0.85rem; margin-right: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "session_context" not in st.session_state:
    st.session_state.session_context = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_report" not in st.session_state:
    st.session_state.last_report = None


def api_health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def render_report(report: dict):
    st.markdown(f'<span class="badge">{report.get("status_label", "Result")}</span>', unsafe_allow_html=True)
    st.subheader(report.get("diagnosis_name", "Diagnosis"))
    if report.get("crop"):
        st.caption(f"Crop: {report['crop']}")

    conf = report.get("confidence", 0)
    cat = report.get("confidence_category", "inconclusive")
    st.progress(min(max(conf, 0.0), 1.0), text=f"Confidence: {conf:.0%} ({cat})")
    st.write(report.get("confidence_explanation", ""))

    if report.get("expert_confirmation_required"):
        st.warning("Needs confirmation — consider local expert advice before major treatment.")

    st.markdown("**Summary**")
    st.write(report.get("description", ""))
    if report.get("initial_message"):
        st.info(report["initial_message"])

    symptoms = report.get("symptoms") or []
    if symptoms:
        st.markdown("**Signs that may fit**")
        for s in symptoms:
            st.markdown(f"- **{s.get('name')}**: {s.get('explanation')}")

    alts = report.get("alternatives") or []
    if alts:
        st.markdown("**Other possibilities**")
        st.write(", ".join(alts))

    actions = report.get("actions") or []
    if actions:
        st.markdown("**What you can do now**")
        for i, a in enumerate(actions, start=1):
            st.markdown(f"{i}. {a.get('step')}")
            if a.get("safety_note"):
                st.caption(f"Safety: {a['safety_note']}")

    if report.get("monitoring_guidance"):
        st.markdown("**Follow-up monitoring**")
        st.write(report["monitoring_guidance"])

    if report.get("diagnostic_limitations"):
        st.markdown("**Limitations**")
        st.caption(report["diagnostic_limitations"])


col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown('<p class="main-header">Plant Disease Detection</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Upload a leaf photo for guidance and ask follow-up questions.</p>',
        unsafe_allow_html=True,
    )
with col_status:
    health = api_health()
    if health:
        llm_note = "LLM on" if health.get("llm_configured") else "LLM off (template mode)"
        st.success(f"API connected · {llm_note}")
    else:
        st.error("API unreachable")

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("### Upload")
    uploaded = st.file_uploader(
        "Leaf image (JPEG, PNG, WebP)",
        type=["jpg", "jpeg", "png", "webp"],
    )
    crop = st.selectbox(
        "Crop (optional hint)",
        ["", "Tomato", "Potato", "Apple", "Corn"],
    )
    extra = st.text_area(
        "Extra symptoms you see (optional)",
        placeholder="e.g. spots spreading after rain",
        height=80,
    )
    analyze = st.button("Analyze leaf", type="primary", use_container_width=True)

    if uploaded:
        st.image(Image.open(uploaded), caption="Preview", use_container_width=True)

    if analyze:
        if not uploaded:
            st.warning("Please upload an image first.")
        else:
            with st.spinner("Running diagnosis..."):
                files = {
                    "file": (
                        uploaded.name,
                        uploaded.getvalue(),
                        uploaded.type or "image/jpeg",
                    )
                }
                data = {"crop": crop, "additional_symptoms": extra}
                try:
                    resp = requests.post(
                        f"{API_URL}/predict",
                        files=files,
                        data=data,
                        timeout=120,
                    )
                    if resp.status_code >= 400:
                        st.error(resp.json().get("detail", resp.text))
                    else:
                        payload = resp.json()
                        st.session_state.last_report = payload.get("report")
                        st.session_state.session_context = payload.get("session_context") or {}
                        st.session_state.chat_history = []
                        if not payload.get("quality", {}).get("ok", True):
                            st.warning(payload["quality"].get("message", "Quality issue"))
                        elif payload.get("low_confidence"):
                            st.warning("Low confidence — treat as a tentative guess.")
                except requests.RequestException as exc:
                    st.error(f"Request failed: {exc}")

with right:
    st.markdown("### Diagnosis")
    if st.session_state.last_report:
        render_report(st.session_state.last_report)
    else:
        st.write("Results will appear here after you analyze a leaf photo.")

st.divider()
st.markdown("### Follow-up chat")
st.caption("Agriculture questions only · tied to your latest diagnosis in this session")

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])

user_msg = st.chat_input("Ask about symptoms, care, or monitoring...")
if user_msg:
    if not st.session_state.session_context:
        st.warning("Analyze a leaf first so chat has diagnosis context.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        try:
            r = requests.post(
                f"{API_URL}/chat",
                json={
                    "message": user_msg,
                    "session_context": st.session_state.session_context,
                },
                timeout=60,
            )
            r.raise_for_status()
            reply = r.json().get("reply", "")
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
        except requests.RequestException as exc:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": f"Chat failed: {exc}"}
            )
        st.rerun()

st.caption(
    "Demo capstone — no login, images are not stored. "
    "Treatment guidance is general; confirm with local experts."
)
