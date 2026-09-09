"""
Real-World Case Study Validation Script
Runs 10 out-of-sample messages through the trained phishing detector
and produces a poster-ready results table.

Usage:
    python run_case_study.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Preprocessing (mirrored from NLP.py for standalone execution)
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
# Main
# ---------------------------------------------------------------------------
def main():
    project_dir = Path(__file__).parent

    # Load model
    bundle = load(project_dir / "phishing_detector_model.joblib")
    model = bundle["model"]
    feature_model = bundle["feature_model"]
    vectorizer = bundle["vectorizer"]
    stemmer = bundle["stemmer"]
    threshold = bundle["decision_threshold"]

    feature_names = vectorizer.get_feature_names_out()
    global_weights = feature_model.coef_[0] if hasattr(feature_model, "coef_") else None

    # Load case study messages
    cs_df = pd.read_csv(project_dir / "case_study_messages.csv", encoding="utf-8-sig")

    results = []
    for idx, row in cs_df.iterrows():
        raw = row["message_text"]
        cleaned, no_stop, stemmed = preprocess(raw, stemmer)
        X = vectorizer.transform([stemmed])
        proba = float(model.predict_proba(X)[0][1])
        pred_label = int(proba >= threshold)
        true_label = int(row["true_label"])
        correct = pred_label == true_label

        # Top 3 features for this message
        top_feats = ""
        if global_weights is not None:
            tfidf_vals = X.toarray()[0]
            contributions = tfidf_vals * global_weights
            nonzero_idx = np.where(tfidf_vals > 0)[0]
            if len(nonzero_idx) > 0:
                feat_contrib = [
                    (feature_names[i], float(contributions[i]))
                    for i in nonzero_idx
                ]
                feat_contrib.sort(key=lambda x: abs(x[1]), reverse=True)
                top_feats = ", ".join(
                    f"{f[0]} ({f[1]:+.2f})" for f in feat_contrib[:3]
                )

        # Truncate message for display
        display_msg = raw[:80] + "…" if len(raw) > 80 else raw

        results.append({
            "msg_id": idx + 1,
            "message": display_msg,
            "language": row["language"],
            "category": row["scam_category"],
            "true_label": "Phishing" if true_label == 1 else "Safe",
            "prediction": "Phishing" if pred_label == 1 else "Safe",
            "confidence": f"{proba:.1%}",
            "correct": "✅" if correct else "❌",
            "top_features": top_feats,
        })

    results_df = pd.DataFrame(results)

    # Save CSV
    results_df.to_csv(project_dir / "case_study_results.csv", index=False, encoding="utf-8-sig")

    # Print results
    total = len(results_df)
    correct_count = sum(1 for r in results if r["correct"] == "✅")
    accuracy = correct_count / total

    print("=" * 90)
    print("REAL-WORLD CASE STUDY VALIDATION RESULTS")
    print("=" * 90)
    print(f"\nMessages tested: {total} (out-of-sample, not in training set)")
    print(f"Accuracy: {correct_count}/{total} = {accuracy:.0%}")
    print()

    # Markdown table
    print("### Poster-Ready Table (Markdown)\n")
    print("| # | Message (truncated) | Language | Category | True Label | Verdict | Confidence | Correct |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        # Escape pipe characters in message
        safe_msg = r["message"].replace("|", "\\|")
        print(
            f"| {r['msg_id']} | {safe_msg} | {r['language']} | {r['category']} | "
            f"{r['true_label']} | {r['prediction']} | {r['confidence']} | {r['correct']} |"
        )

    print(f"\n**Overall Case Study Accuracy: {accuracy:.0%} ({correct_count}/{total})**")

    # Also print feature analysis
    print("\n\n### Key Feature Contributions per Message\n")
    for r in results:
        print(f"**Msg {r['msg_id']}** ({r['language']}, {r['category']}): {r['top_features']}")

    return results_df, accuracy


if __name__ == "__main__":
    main()
