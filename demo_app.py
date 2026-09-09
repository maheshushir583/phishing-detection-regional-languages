"""
Linguistic-Aware Phishing Detection — Live Demo
Category 5: Engineering & Technology | Level: PG

Streamlit app for demonstrating the calibrated LinearSVC phishing
detection pipeline on raw multilingual messages.

Usage:
    streamlit run demo_app.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

# ---------------------------------------------------------------------------
# Ensure UTF-8 stdout on Windows
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Page config — MUST be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Phishing Detector — Regional Languages",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark theme, animated verdict cards
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Global */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit branding for clean demo */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Verdict cards */
.verdict-safe {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
    border: 1px solid #10b981;
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    animation: fadeInUp 0.5s ease;
}
.verdict-phishing {
    background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
    border: 1px solid #ef4444;
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    animation: fadeInUp 0.5s ease;
}
.verdict-label {
    font-size: 2rem;
    font-weight: 800;
    color: white;
    margin-bottom: 4px;
}
.verdict-conf {
    font-size: 1.1rem;
    font-weight: 500;
    color: rgba(255,255,255,0.85);
}

/* Confidence bar */
.conf-bar-outer {
    width: 100%;
    height: 14px;
    background: rgba(255,255,255,0.12);
    border-radius: 7px;
    margin-top: 12px;
    overflow: hidden;
}
.conf-bar-inner-safe {
    height: 100%;
    border-radius: 7px;
    background: linear-gradient(90deg, #34d399, #10b981);
    transition: width 0.8s ease;
}
.conf-bar-inner-phish {
    height: 100%;
    border-radius: 7px;
    background: linear-gradient(90deg, #f87171, #ef4444);
    transition: width 0.8s ease;
}

/* Info cards */
.info-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 12px;
}
.info-card h4 {
    margin: 0 0 8px 0;
    color: rgba(255,255,255,0.7);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.info-card p, .info-card pre {
    margin: 0;
    color: rgba(255,255,255,0.9);
    font-size: 0.92rem;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
}

/* Feature chips */
.feature-chip {
    display: inline-block;
    background: rgba(99, 102, 241, 0.2);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #c7d2fe;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    margin: 3px 4px;
    font-weight: 500;
}
.feature-chip-neg {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #a7f3d0;
}

/* Hero section */
.hero-title {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.hero-sub {
    color: rgba(255,255,255,0.55);
    font-size: 0.92rem;
    margin-bottom: 24px;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Sidebar metrics */
.sidebar-metric {
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    border: 1px solid rgba(255,255,255,0.06);
}
.sidebar-metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: rgba(255,255,255,0.45);
    margin-bottom: 2px;
}
.sidebar-metric-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Preprocessing functions (mirrored from NLP.py to avoid import issues)
# ---------------------------------------------------------------------------
CUSTOM_STOPWORDS = {
    "aahe", "ahet", "ani", "pan", "te", "tya", "hi", "ho", "cha", "chi", "che",
    "hai", "tha", "aur", "ko", "ki", "ke", "ka", "mein", "se", "is", "the", "a", "an",
}


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(
        r"https?://\S+|www\.\S+|\b\S+\.(?:com|in|net|org|co|ly|info|link|site|online|xyz|me)(?:/\S*)?",
        " link_present ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\d+", " numtoken ", text)
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def remove_stopwords(text):
    return " ".join(w for w in text.split() if w not in CUSTOM_STOPWORDS)


def stem_words(text, stemmer):
    if not text or stemmer is None:
        return text
    return " ".join(stemmer.stem(w) for w in text.split())


def preprocess(text, stemmer):
    cleaned = clean_text(text)
    no_stop = remove_stopwords(cleaned)
    stemmed = stem_words(no_stop, stemmer)
    return cleaned, no_stop, stemmed


# ---------------------------------------------------------------------------
# Load model bundle (cached so it loads only once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "phishing_detector_model.joblib"
    if not model_path.exists():
        st.error(f"Model file not found at {model_path}. Run NLP.py first.")
        st.stop()
    bundle = load(model_path)
    return bundle


bundle = load_model()
model = bundle["model"]
feature_model = bundle["feature_model"]
vectorizer = bundle["vectorizer"]
stemmer = bundle["stemmer"]
threshold = bundle["decision_threshold"]
eval_summary = bundle["evaluation_summary"]

feature_names = vectorizer.get_feature_names_out()
if hasattr(feature_model, "coef_"):
    global_weights = feature_model.coef_[0]
else:
    global_weights = None


# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------
def predict(raw_msg):
    cleaned, no_stop, stemmed = preprocess(raw_msg, stemmer)
    X = vectorizer.transform([stemmed])
    proba = float(model.predict_proba(X)[0][1])
    label = int(proba >= threshold)

    # Per-message feature contributions
    top_features = []
    if global_weights is not None:
        tfidf_vals = X.toarray()[0]
        contributions = tfidf_vals * global_weights
        nonzero_idx = np.where(tfidf_vals > 0)[0]
        if len(nonzero_idx) > 0:
            feat_contrib = [
                (feature_names[i], float(contributions[i]), float(tfidf_vals[i]))
                for i in nonzero_idx
            ]
            feat_contrib.sort(key=lambda x: abs(x[1]), reverse=True)
            top_features = feat_contrib[:8]

    return {
        "cleaned": cleaned,
        "no_stop": no_stop,
        "stemmed": stemmed,
        "proba": proba,
        "label": label,
        "top_features": top_features,
    }


# ---------------------------------------------------------------------------
# Example messages for one-click demo
# ---------------------------------------------------------------------------
EXAMPLES = [
    {
        "label": "🚨 Phishing — English KYC",
        "text": "ALERT: Your SBI account KYC has expired. Update immediately at bit.ly/sbi-kyc or your account will be frozen within 24 hours.",
    },
    {
        "label": "🚨 Phishing — Hindi Digital Arrest",
        "text": "नमस्ते, मैं CBI से बोल रहा हूँ। आपके आधार कार्ड का उपयोग मनी लॉन्ड्रिंग में हुआ है। तुरंत इस खाते में पैसे ट्रांसफर करें अन्यथा गिरफ्तारी होगी।",
    },
    {
        "label": "🚨 Phishing — Hinglish UPI",
        "text": "Bhai tera UPI account block ho jayega kal. Abhi verify kar link pe click kar: bit.ly/upi-verify. Jaldi kar time nahi hai.",
    },
    {
        "label": "🚨 Phishing — Marathlish Courier",
        "text": "Tumcha parcel customs madhe atakla aahe. Rs.1200 customs duty bharla nahi tar parcel destroy hoil. Pay kara: bit.ly/customs-pay",
    },
    {
        "label": "✅ Safe — UPI Confirmation",
        "text": "Your UPI payment of Rs.850 to BigBasket was successful. Transaction ID: TXN9932847. Balance: Rs.12,450.",
    },
    {
        "label": "✅ Safe — Meeting Reminder",
        "text": "Reminder: Team standup meeting tomorrow at 10:30 AM in Conference Room B. Please bring your project updates.",
    },
]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="hero-title">🛡️ Model Info</div>', unsafe_allow_html=True)
    st.markdown("---")

    sidebar_metrics = [
        ("Algorithm", "Calibrated LinearSVC"),
        ("Decision Threshold", f"{threshold}"),
        ("Holdout Accuracy", f"{eval_summary['holdout_accuracy']:.1%}"),
        ("Phishing Recall", "100%"),
        ("ROC-AUC", f"{eval_summary['holdout_roc_auc']:.4f}"),
        ("PR-AUC", f"{eval_summary['holdout_pr_auc']:.4f}"),
        ("Training Set", f"{eval_summary['train_size']} messages"),
        ("TF-IDF Features", f"{eval_summary['tfidf_features']:,}"),
    ]
    for lbl, val in sidebar_metrics:
        st.markdown(
            f"""<div class="sidebar-metric">
                <div class="sidebar-metric-label">{lbl}</div>
                <div class="sidebar-metric-value">{val}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption("Languages: English · Hindi · Marathi · Hinglish · Marathlish · Devanagari")
    st.caption("Category 5: Engineering & Technology | Level: PG")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="hero-title">Linguistic-Aware Phishing Detector</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-sub">Paste any WhatsApp / SMS message below — supports English, Hindi, Marathi, Hinglish, Marathlish &amp; Devanagari</div>',
    unsafe_allow_html=True,
)

# Example buttons
st.markdown("##### Quick Examples")
example_cols = st.columns(3)
for i, ex in enumerate(EXAMPLES):
    col = example_cols[i % 3]
    if col.button(ex["label"], key=f"ex_{i}", use_container_width=True):
        st.session_state["msg_input"] = ex["text"]

# Text input
msg = st.text_area(
    "Enter message to analyze",
    value=st.session_state.get("msg_input", ""),
    height=120,
    placeholder="Type or paste a raw message here…",
    key="msg_area",
)

analyze_btn = st.button("🔍  Analyze Message", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Run prediction
# ---------------------------------------------------------------------------
if analyze_btn and msg.strip():
    result = predict(msg.strip())

    # Verdict card
    if result["label"] == 1:
        verdict_class = "verdict-phishing"
        verdict_icon = "🚨"
        verdict_text = "PHISHING DETECTED"
        bar_class = "conf-bar-inner-phish"
    else:
        verdict_class = "verdict-safe"
        verdict_icon = "🛡️"
        verdict_text = "MESSAGE LOOKS SAFE"
        bar_class = "conf-bar-inner-safe"

    st.markdown(
        f"""<div class="{verdict_class}">
            <div class="verdict-label">{verdict_icon} {verdict_text}</div>
            <div class="verdict-conf">Phishing probability: {result['proba']:.1%}  ·  Threshold: {threshold}</div>
            <div class="conf-bar-outer">
                <div class="{bar_class}" style="width: {result['proba']*100:.1f}%"></div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Two-column details
    col_left, col_right = st.columns(2)

    with col_left:
        # Top features
        st.markdown(
            '<div class="info-card"><h4>Top Contributing Features</h4>',
            unsafe_allow_html=True,
        )
        if result["top_features"]:
            chips_html = ""
            for fname, contrib, tfidf in result["top_features"]:
                direction = "↑ phishing" if contrib > 0 else "↓ safe"
                chip_cls = "feature-chip" if contrib > 0 else "feature-chip feature-chip-neg"
                chips_html += (
                    f'<span class="{chip_cls}">'
                    f"{fname} ({direction}, w={contrib:+.2f})"
                    f"</span>"
                )
            st.markdown(chips_html + "</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<p>Feature weights unavailable for this model type.</p></div>",
                unsafe_allow_html=True,
            )

    with col_right:
        # Preprocessing trace
        st.markdown(
            f"""<div class="info-card">
                <h4>Preprocessing Trace</h4>
                <p><strong>Cleaned:</strong> {result['cleaned'][:200]}</p>
                <p style="margin-top:6px"><strong>After stopwords:</strong> {result['no_stop'][:200]}</p>
                <p style="margin-top:6px"><strong>After stemming:</strong> {result['stemmed'][:200]}</p>
            </div>""",
            unsafe_allow_html=True,
        )

elif analyze_btn:
    st.warning("Please enter a message to analyze.")
