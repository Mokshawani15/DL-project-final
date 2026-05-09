"""
app.py — AI-Based Image Answer Evaluator
Main Streamlit application. Orchestrates the full pipeline:
  1. Image input (upload or AI-generated)
  2. Caption generation (BLIP)
  3. Paragraph expansion (GPT-2)
  4. User answer input
  5. Semantic similarity scoring
  6. Feedback display
"""

import io
import time
import random
from pathlib import Path
import streamlit as st
from PIL import Image

# Groq API key (replace with your actual token)
GROQ_API_KEY = "YOUR_API_KEY_HERE"

# ── Page config (MUST be the very first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="ImageIQ — AI Answer Evaluator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #ffffff !important;
}
h1, h2, h3, p, span, label {
    color: #ffffff !important;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* ── Remove default padding ── */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* ── Hero header ── */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    margin-bottom: 1.5rem;
}
.hero-header h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -1px;
}
.hero-header p {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

/* ── Step badge ── */
.step-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(167,139,250,0.15);
    border: 1px solid rgba(167,139,250,0.35);
    border-radius: 999px;
    padding: 0.3rem 1rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: #a78bfa;
    margin-bottom: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Card / glass panel ── */
.glass-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.4rem;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}

/* ── Caption box ── */
.caption-box {
    background: rgba(96,165,250,0.1);
    border-left: 4px solid #60a5fa;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    color: #bfdbfe;
    font-style: italic;
    font-size: 1rem;
    margin-top: 0.6rem;
}

/* ── Paragraph box ── */
.para-box {
    background: rgba(52,211,153,0.08);
    border-left: 4px solid #34d399;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #a7f3d0;
    font-size: 0.93rem;
    line-height: 1.75;
    margin-top: 0.6rem;
    white-space: pre-wrap;
}

/* ── Score ring ── */
.score-ring-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1rem;
}
.score-number {
    font-size: 4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.score-label {
    color: #ffffff;
    font-size: 0.9rem;
    margin-top: 0.3rem;
}

/* ── Grade pill ── */
.grade-pill {
    display: inline-block;
    padding: 0.4rem 1.4rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

/* ── Feedback box ── */
.feedback-box {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    color: #ffffff;
    font-size: 0.95rem;
    line-height: 1.7;
    margin-top: 0.8rem;
}

/* ── Progress bar override ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #a78bfa, #60a5fa) !important;
    border-radius: 999px !important;
}
.stProgress {
    margin-top: 0.5rem;
}

/* ── Buttons ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(124,58,237,0.4) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Text areas & inputs ── */
.stTextArea textarea {
    background: rgba(30, 41, 59, 0.7) !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.25) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 1.5rem 0 !important;
}

/* ── Section titles ── */
.section-title {
    color: #e2e8f0;
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
.section-sub {
    color: #64748b;
    font-size: 0.85rem;
    margin-bottom: 0.8rem;
}

/* ── Word count badge ── */
.word-count {
    font-size: 0.78rem;
    color: #64748b;
    text-align: right;
    margin-top: 0.3rem;
}

/* ── Spinner override ── */
.stSpinner > div {
    border-top-color: #7c3aed !important;
}

/* ── Info / success / error overrides ── */
.stAlert {
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ── imports (avoids loading DL models on startup before needed) ──────────
@st.cache_resource(show_spinner=False)
def load_caption_module():
    from caption import generate_caption
    return generate_caption

@st.cache_resource(show_spinner=False)
def load_paragraph_module():
    from paragraph import expand_caption
    return expand_caption

@st.cache_resource(show_spinner=False)
def load_similarity_module():
    from similarity import compute_similarity, similarity_to_score
    return compute_similarity, similarity_to_score

from feedback import generate_feedback


# ── Helper: render step badge ─────────────────────────────────────────────────
def step_badge(icon: str, label: str):
    st.markdown(f'<div class="step-badge">{icon} {label}</div>', unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <h1>🧠 ImageIQ</h1>
  <p>AI-powered image description evaluator — see how well you can describe what you see</p>
</div>
""", unsafe_allow_html=True)


# ── Project Significance Section ──────────────────────────────────────────────
st.markdown("""
<div class="glass-card">
  <div style="text-align:center;padding:1rem 0;">
    <h2 style="font-size:1.8rem;font-weight:700;
        background: linear-gradient(90deg,#34d399,#60a5fa,#a78bfa);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        margin-bottom:0.5rem;">
        How ImageIQ will help you
    </h2>
    <p style="color:#94a3b8;font-size:1rem;margin-bottom:1rem;">
       ImageIQ isn’t just tech — it’s a creative learning tool.
    </p>
  </div>

  <ul style="color:#e2e8f0;font-size:0.95rem;line-height:1.7;">
    <li>🔍 <b>Observation Skills</b> — notice details you might miss.</li>
    <li>📝 <b>Descriptive Ability</b> — learn to write richer, precise language.</li>
    <li>💡 <b>Creative Thinking</b> — generate your own interpretations.</li>
    <li>🎯 <b>Constructive Feedback</b> — reflect and improve continuously.</li>
    <li>🤝 <b>AI + Human Creativity</b> — AI guides, you imagine.</li>
  </ul>

  <hr style="border-color:rgba(255,255,255,0.08);margin:1rem 0;" />

  <p style="color:#a7f3d0;font-size:0.9rem;line-height:1.6;text-align:center;">
    🚀 By practicing repeatedly with different images, learners develop a habit of 
    <b>creative exploration</b> — essential for problem‑solving and innovation.
  </p>
</div>
""", unsafe_allow_html=True)

# ── User Guide Section ─────────────────────────────────────────────────────────
st.markdown("""
<div class="glass-card">
  <div style="text-align:center;padding:1rem 0;">
    <h2 style="font-size:1.6rem;font-weight:700;
        background: linear-gradient(90deg,#7c3aed,#2563eb);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        margin-bottom:0.8rem;">
        🚀 How This Project Works
    </h2>
    <p style="color:#94a3b8;font-size:1rem;margin-bottom:1rem;">
       Follow these steps to evaluate your descriptive skills:
    </p>
  </div>

  <div style="display:flex;justify-content:space-around;align-items:center;
              flex-wrap:wrap;color:#e2e8f0;font-size:0.9rem;">
    <div style="text-align:center;">
      <span style="font-size:2rem;">📸</span><br>Image<br>
      <small>Upload your picture </small>
      <small>or Randomly generate one</small>
    </div>
    <div style="font-size:1.5rem;">➡️</div>
    <div style="text-align:center;">
      <span style="font-size:2rem;">💬</span><br>Caption<br>
      <small>AI will generate a short caption</small>
    </div>
    <div style="font-size:1.5rem;">➡️</div>
    <div style="text-align:center;">
      <span style="font-size:2rem;">📝</span><br>Paragraph<br>
      <small>AI will wait for your response</small>
    </div>
    <div style="font-size:1.5rem;">➡️</div>
    <div style="text-align:center;">
      <span style="font-size:2rem;">✍️</span><br>Your Answer<br>
      <small>Write your own description</small>
    </div>
    <div style="font-size:1.5rem;">➡️</div>
    <div style="text-align:center;">
      <span style="font-size:2rem;">📊</span><br>Score<br>
      <small>Now the system will perform Similarity evaluation</small>
    </div>
    <div style="font-size:1.5rem;">➡️</div>
    <div style="text-align:center;">
      <span style="font-size:2rem;">🎯</span><br>Feedback<br>
      <small>Get tips to improve and see the ideal answer</small>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# # ── Horizontal pipeline indicator ────────────────────────────────────────────
# pipeline_steps = [
#     ("📸", "Image"),
#     ("💬", "Caption"),
#     ("📝", "Paragraph"),
#     ("✍️", "Your Answer"),
#     ("📊", "Score"),
#     ("🎯", "Feedback"),
# ]
# cols = st.columns(len(pipeline_steps))
# for col, (icon, label) in zip(cols, pipeline_steps):
#     col.markdown(
#         f"<div style='text-align:center;color:#94a3b8;font-size:0.78rem;'>"
#         f"<span style='font-size:1.4rem;display:block;margin-bottom:2px'>{icon}</span>{label}</div>",
#         unsafe_allow_html=True,
#     )
# st.markdown("---")

# ── Session state initialisation ──────────────────────────────────────────────
for key, default in {
    "image": None,
    "image_name": None,          # filename shown under the image
    "caption": None,
    "paragraph": None,
    "score": None,
    "feedback": None,
    "stage": "upload",          # upload → caption → paragraph → answer → result
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Image Input (Upload or Random Flickr8k)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
step_badge("01", "Image Input")
st.markdown('<div class="section-title">Provide an Image</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Choose to upload your own image or let the system pick one from the Flickr8k dataset.</div>',
    unsafe_allow_html=True,
)

tab_upload, tab_random = st.tabs(["📤 Upload Your Image", "🎲 Random Flickr8k Image"])

# ── Tab 1: Upload ──────────────────────────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader(
        "Drop an image here",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
        key="file_uploader",
    )
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.session_state.image      = img
        st.session_state.image_name = uploaded.name
        # Reset downstream state
        for key in ["caption", "paragraph", "score", "feedback"]:
            st.session_state[key] = None
        st.session_state.stage = "caption"

# ── Tab 2: Random Flickr8k ────────────────────────────────────────────────────
with tab_random:
    flickr_path = Path("/Users/moksha_wani/DL project final/data/Images")  # adjust this path to your dataset location
    if st.button("🎲 Pick a Random Flickr8k Image"):
        valid_exts = {".jpg", ".jpeg", ".png"}
        all_images = [p for p in flickr_path.iterdir() if p.suffix.lower() in valid_exts]
        if not all_images:
            st.error("No images found in Flickr8k folder. Check your dataset path.")
        else:
            chosen = random.choice(all_images)
            img = Image.open(chosen).convert("RGB")
            st.session_state.image      = img
            st.session_state.image_name = chosen.name
            # Reset downstream state
            for key in ["caption", "paragraph", "score", "feedback"]:
                st.session_state[key] = None
            st.session_state.stage = "caption"
            st.success(f"✅ Randomly selected: **{chosen.name}**")
            st.rerun()

# ── Show selected image ───────────────────────────────────────────────────────
if st.session_state.image:
    img_col, _ = st.columns([1, 1])
    with img_col:
        caption_label = st.session_state.image_name or "Input Image"
        st.image(st.session_state.image, caption=caption_label, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# STEP 2: USER ANSWER
# ════════════════════════════════════════════════════════
if st.session_state.image:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    step_badge("02", "Your Answer")

    user_answer = st.text_area(
        "Describe the image",
        key="user_answer_input",
        height=150
    )

    st.markdown('</div>', unsafe_allow_html=True)

from ideal_answer import call_api_generate_paragraph

# STEP 3: Generate Ideal Answer via Groq API
if st.session_state.image and not st.session_state.paragraph:
    if st.button("🔍 Generate Ideal Answer", key="btn_caption"):
        if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY":
            st.error("❌ Please set your `GROQ_API_KEY` in `app.py` first.")
        else:
            with st.spinner("Analyzing image with Groq AI..."):
                ideal_paragraph = call_api_generate_paragraph(st.session_state.image, GROQ_API_KEY)
                st.session_state.paragraph = ideal_paragraph
                st.session_state.stage = "answer"
            st.rerun()

# STEP 4: Display Ideal Answer
if st.session_state.paragraph:
    word_count = len(st.session_state.paragraph.split())
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    step_badge("04", "Ideal Answer (AI Reference)")
    st.markdown(f'<div class="para-box">{st.session_state.paragraph}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="word-count">📝 {word_count} words</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# STEP 5: EVALUATION
# ════════════════════════════════════════════════════════
user_answer = st.session_state.get("user_answer_input", "")

if st.session_state.paragraph:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    step_badge("05", "Evaluation")

    if st.button("🚀 Evaluate My Answer"):

        if not user_answer.strip():
            st.warning("Please write your answer first.")
        else:
            compute_similarity, similarity_to_score = load_similarity_module()

            similarity = compute_similarity(
            st.session_state.paragraph,
            user_answer.strip()
)


            score = similarity_to_score(similarity)
            feedback = generate_feedback(score)

            st.session_state.score = score
            st.session_state.feedback = feedback
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# RESULT
# ════════════════════════════════════════════════════════

if st.session_state.score is not None:
    fb = st.session_state.feedback
    score = st.session_state.score

    st.markdown("---")
    st.markdown(f"## 📊 Score: {score:.1f}%")
    st.markdown(f"### {fb.emoji} {fb.grade}")
    st.write(fb.detailed_feedback)

    st.markdown("### 📌 Ideal Answer")
    st.write(st.session_state.paragraph)   # ← use the paragraph already generated

    if st.button("🔄 Try Again"):
        for key in ["image", "caption", "paragraph", "score", "feedback"]:
            st.session_state[key] = None
        st.rerun()

    # Performance breakdown...
    # (keep the rest of your metrics code unchanged)


    # ── Score breakdown bar ──────────────────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title" style="margin-bottom:1rem">Performance Breakdown</div>',
        unsafe_allow_html=True,
    )
    metrics = {
        "🎯 Overall Accuracy":  score,
        "🏅 Grade":             {"Excellent": 100, "Good": 80, "Fair": 60, "Poor": 30}[fb.grade],
    }
    m1, m2, m3 = st.columns(3)
    m1.metric("Score", f"{score:.1f}%")
    m2.metric("Grade", f"{fb.emoji} {fb.grade}")
    m3.metric("Status", "✅ Pass" if score >= 50 else "❌ Needs Work")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Try again ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Try with a Different Image", key="btn_reset"):
        for key in ["image", "image_name", "caption", "paragraph", "score", "feedback"]:
            st.session_state[key] = None
        st.session_state.stage = "upload"
        st.rerun()


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#334155;font-size:0.8rem;padding-bottom:1rem;">'
    'ImageIQ • BLIP + GPT-2 + Sentence-BERT • Deep Learning Project'
    '</div>',
    unsafe_allow_html=True,
)
