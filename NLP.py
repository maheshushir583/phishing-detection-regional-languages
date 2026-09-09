import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


try:
    from nltk.stem import PorterStemmer
except ModuleNotFoundError:
    PorterStemmer = None


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


CUSTOM_STOPWORDS = [
    "aahe", "ahet", "ani", "pan", "te", "tya", "hi", "ho", "cha", "chi", "che",
    "hai", "tha", "aur", "ko", "ki", "ke", "ka", "mein", "se", "is", "the", "a", "an",
]

DEVANAGARI_DIGIT_TRANSLATION = str.maketrans("०१२३४५६७८९", "0123456789")

SAFE_WORD_HINTS = {
    "otp", "transaction", "successful", "recharge", "validity", "meeting",
    "birthday", "order", "tracking", "deliver", "courier", "teams", "bank",
}

PHISHING_WORD_HINTS = {
    "urgent", "blocked", "freeze", "kyc", "lottery", "parcel", "seized",
    "click", "suspend", "account", "bill", "helpline", "call", "link_present",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a phishing detector and test raw WhatsApp messages."
    )
    parser.add_argument(
        "--dataset",
        default="phishing_dataset_400_fixed.csv",
        help="CSV dataset path. Default: phishing_dataset_400_fixed.csv",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for phishing probability. Default: 0.5",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Folder where research artifacts are saved. Default: current folder.",
    )
    parser.add_argument(
        "--model-output",
        default="phishing_detector_model.joblib",
        help="Saved model filename. Use with --output-dir. Default: phishing_detector_model.joblib",
    )
    parser.add_argument(
        "--final-model",
        choices=["auto", "MultinomialNB", "LogisticRegression", "LinearSVC"],
        default="auto",
        help="Final model to train after benchmarking. Default: auto selects the top CV model.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip PNG chart generation.",
    )
    parser.add_argument(
        "--message",
        help="Blind-test one raw WhatsApp message against the trained model.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for a raw WhatsApp message after model training.",
    )
    args = parser.parse_args()
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be between 0 and 1.")
    return args


def load_dataset(file_path):
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lower()

    required_columns = {"message_text", "is_phishing"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise KeyError(
            f"Missing required columns: {sorted(missing_columns)}. "
            f"Available columns: {df.columns.tolist()}"
        )

    return df


def normalize_labels(series):
    normalized = (
        series.astype(str)
        .str.strip()
        .str.translate(DEVANAGARI_DIGIT_TRANSLATION)
    )

    if not normalized.isin(["0", "1"]).all():
        invalid_values = sorted(normalized[~normalized.isin(["0", "1"])].unique().tolist())
        raise ValueError(
            f"Unsupported values found in 'is_phishing': {invalid_values}. "
            "Use only 0/1 or ०/१."
        )

    return normalized.astype(int)


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
    words = text.split()
    filtered_words = [word for word in words if word not in CUSTOM_STOPWORDS]
    return " ".join(filtered_words)


def stem_words(text, stemmer):
    if not text:
        return ""

    if stemmer is None:
        return text

    return " ".join(stemmer.stem(word) for word in text.split())


def preprocess_text(text, stemmer):
    clean_message = clean_text(text)
    processed_text = remove_stopwords(clean_message)
    stemmed_text = stem_words(processed_text, stemmer)
    return clean_message, processed_text, stemmed_text


def preprocess_dataframe(df):
    stemmer = PorterStemmer() if PorterStemmer is not None else None

    df = df.copy()
    df["is_phishing"] = normalize_labels(df["is_phishing"])
    processed_columns = df["message_text"].apply(
        lambda text: pd.Series(preprocess_text(text, stemmer))
    )
    processed_columns.columns = ["clean_message", "final_processed_text", "stemmed_text"]
    df[processed_columns.columns] = processed_columns

    return df, stemmer


def build_vectorizer():
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_df=0.95,
        max_features=3000,
    )


def build_candidate_models():
    return {
        "MultinomialNB": MultinomialNB(alpha=0.1),
        "LogisticRegression": LogisticRegression(
            max_iter=3000,
            C=3.0,
            class_weight="balanced",
        ),
        "LinearSVC": LinearSVC(
            C=1.5,
            class_weight="balanced",
        ),
    }


def benchmark_models(df):
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(df["stemmed_text"])
    y = df["is_phishing"].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    benchmark_rows = []
    for model_name, model in build_candidate_models().items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "f1": "f1",
                "precision": "precision",
                "recall": "recall",
                "roc_auc": "roc_auc",
                "pr_auc": "average_precision",
            },
            n_jobs=1,
        )
        benchmark_rows.append(
            {
                "model": model_name,
                "cv_accuracy": scores["test_accuracy"].mean(),
                "cv_f1": scores["test_f1"].mean(),
                "cv_precision": scores["test_precision"].mean(),
                "cv_recall": scores["test_recall"].mean(),
                "cv_roc_auc": scores["test_roc_auc"].mean(),
                "cv_pr_auc": scores["test_pr_auc"].mean(),
            }
        )

    benchmark_df = pd.DataFrame(benchmark_rows).sort_values(
        by=["cv_accuracy", "cv_f1"],
        ascending=False,
    )
    best_model_name = benchmark_df.iloc[0]["model"]
    return benchmark_df, best_model_name


def extract_linear_feature_weights(model, vectorizer, top_n=15):
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    weights_df = pd.DataFrame({"feature": feature_names, "weight": coefficients})

    phishing_features = weights_df.sort_values(by="weight", ascending=False).head(top_n)
    legitimate_features = weights_df.sort_values(by="weight", ascending=True).head(top_n)
    bigram_features = weights_df[weights_df["feature"].str.contains(" ", regex=False)]
    bigram_features = bigram_features.sort_values(by="weight", ascending=False).head(top_n)

    return phishing_features, legitimate_features, bigram_features


def extract_nb_feature_weights(model, vectorizer, top_n=15):
    feature_names = vectorizer.get_feature_names_out()
    score_difference = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    weights_df = pd.DataFrame({"feature": feature_names, "weight": score_difference})

    phishing_features = weights_df.sort_values(by="weight", ascending=False).head(top_n)
    legitimate_features = weights_df.sort_values(by="weight", ascending=True).head(top_n)
    bigram_features = weights_df[weights_df["feature"].str.contains(" ", regex=False)]
    bigram_features = bigram_features.sort_values(by="weight", ascending=False).head(top_n)

    return phishing_features, legitimate_features, bigram_features


def explain_error(processed_text, predicted_label):
    tokens = processed_text.split()
    token_set = set(tokens)
    safe_overlap = sorted(token_set.intersection(SAFE_WORD_HINTS))
    phishing_overlap = sorted(token_set.intersection(PHISHING_WORD_HINTS))
    unique_tokens = [token for token in tokens if len(token) > 7]

    if predicted_label == 0 and safe_overlap:
        return (
            "Likely false negative: the message contains neutral or service-style terms "
            f"({', '.join(safe_overlap[:3])}) that may hide the phishing intent."
        )

    if predicted_label == 0 and unique_tokens:
        return (
            "Likely false negative: the message uses rare or highly specific tokens "
            f"({', '.join(unique_tokens[:3])}) that are underrepresented in training data."
        )

    if predicted_label == 1 and phishing_overlap:
        return (
            "Likely false positive: the model focused on strong phishing-style cues "
            f"({', '.join(phishing_overlap[:3])}) even though the message may be safe."
        )

    if predicted_label == 1:
        return (
            "Likely false positive: short urgent wording or account-related vocabulary "
            "made the message resemble phishing examples."
        )

    return (
        "Likely error cause: the mixed Marathi/Hinglish phrasing is too unique for the "
        "current dataset size."
    )


def analyze_misclassifications(df_test, y_test, y_pred, y_proba):
    analysis_df = df_test.copy()
    analysis_df["actual_label"] = y_test
    analysis_df["predicted_label"] = y_pred
    analysis_df["phishing_probability"] = y_proba
    analysis_df = analysis_df[analysis_df["actual_label"] != analysis_df["predicted_label"]].copy()

    analysis_df["error_type"] = analysis_df.apply(
        lambda row: (
            "False Negative" if row["actual_label"] == 1 and row["predicted_label"] == 0
            else "False Positive"
        ),
        axis=1,
    )
    analysis_df["reason"] = analysis_df.apply(
        lambda row: explain_error(row["final_processed_text"], row["predicted_label"]),
        axis=1,
    )

    false_negatives = analysis_df[analysis_df["error_type"] == "False Negative"]
    other_errors = analysis_df[analysis_df["error_type"] != "False Negative"]
    selected_errors = pd.concat([false_negatives, other_errors], ignore_index=True).head(3)

    return analysis_df, selected_errors


def build_threshold_analysis(y_true, y_proba):
    rows = []
    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
            }
        )

    return pd.DataFrame(rows)


def build_evaluation_summary(results, best_model_name, decision_threshold):
    tn, fp, fn, tp = results["confusion_matrix"].ravel()
    return pd.DataFrame(
        [
            {
                "selected_model": best_model_name,
                "decision_threshold": decision_threshold,
                "train_size": results["train_size"],
                "test_size": results["test_size"],
                "tfidf_features": results["X_shape"][1],
                "holdout_accuracy": results["accuracy"],
                "holdout_roc_auc": results["roc_auc"],
                "holdout_pr_auc": results["pr_auc"],
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
            }
        ]
    )


def build_confusion_matrix_df(matrix):
    return pd.DataFrame(
        matrix,
        index=["Actual Safe (0)", "Actual Phishing (1)"],
        columns=["Predicted Safe (0)", "Predicted Phishing (1)"],
    )


def train_final_model(df, best_model_name, decision_threshold=0.5):
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["is_phishing"],
    )

    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(train_df["stemmed_text"])
    X_test = vectorizer.transform(test_df["stemmed_text"])
    y_train = train_df["is_phishing"].values
    y_test = test_df["is_phishing"].values

    if best_model_name == "LinearSVC":
        feature_model = LinearSVC(C=1.5, class_weight="balanced")
        prediction_model = CalibratedClassifierCV(
            estimator=LinearSVC(C=1.5, class_weight="balanced"),
            method="sigmoid",
            cv=3,
        )
    elif best_model_name == "LogisticRegression":
        feature_model = LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced")
        prediction_model = LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced")
    else:
        feature_model = MultinomialNB(alpha=0.1)
        prediction_model = MultinomialNB(alpha=0.1)

    feature_model.fit(X_train, y_train)
    prediction_model.fit(X_train, y_train)

    if hasattr(prediction_model, "predict_proba"):
        y_proba = prediction_model.predict_proba(X_test)[:, 1]
    else:
        y_proba = prediction_model.decision_function(X_test)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
    y_pred = (y_proba >= decision_threshold).astype(int)

    phishing_features = legitimate_features = bigram_features = pd.DataFrame()
    if hasattr(feature_model, "coef_"):
        phishing_features, legitimate_features, bigram_features = extract_linear_feature_weights(
            feature_model,
            vectorizer,
            top_n=15,
        )
    elif hasattr(feature_model, "feature_log_prob_"):
        phishing_features, legitimate_features, bigram_features = extract_nb_feature_weights(
            feature_model,
            vectorizer,
            top_n=15,
        )

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    threshold_analysis_df = build_threshold_analysis(y_test, y_proba)
    error_analysis_df, selected_errors = analyze_misclassifications(
        test_df,
        y_test,
        y_pred,
        y_proba,
    )

    return {
        "feature_model": feature_model,
        "prediction_model": prediction_model,
        "vectorizer": vectorizer,
        "accuracy": accuracy,
        "report": report,
        "report_dict": report_dict,
        "confusion_matrix": matrix,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold_analysis_df": threshold_analysis_df,
        "error_analysis_df": error_analysis_df,
        "selected_errors": selected_errors,
        "phishing_features": phishing_features,
        "legitimate_features": legitimate_features,
        "bigram_features": bigram_features,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "X_shape": vectorizer.transform(df["stemmed_text"]).shape,
    }


def print_benchmark_results(benchmark_df):
    formatted_df = benchmark_df.copy()
    for column in [
        "cv_accuracy",
        "cv_f1",
        "cv_precision",
        "cv_recall",
        "cv_roc_auc",
        "cv_pr_auc",
    ]:
        formatted_df[column] = formatted_df[column].map(lambda value: round(value, 4))

    print("\nCross-Validation Benchmark:")
    print(formatted_df.to_string(index=False))


def print_confusion_matrix(matrix):
    matrix_df = build_confusion_matrix_df(matrix)
    print("\nConfusion Matrix:")
    print(matrix_df.to_string())


def print_probability_metrics(results):
    print("\nProbability-Based Holdout Metrics:")
    print("ROC-AUC:", round(results["roc_auc"], 4))
    print("PR-AUC:", round(results["pr_auc"], 4))


def print_threshold_analysis(threshold_analysis_df):
    printable_df = threshold_analysis_df.copy()
    for column in ["accuracy", "precision", "recall", "f1"]:
        printable_df[column] = printable_df[column].map(lambda value: round(value, 4))

    print("\nThreshold Analysis:")
    print(printable_df.to_string(index=False))


def print_feature_table(title, features_df):
    print(f"\n{title}")
    if features_df.empty:
        print("No interpretable feature weights were available for this model.")
        return

    printable_df = features_df.copy()
    printable_df["weight"] = printable_df["weight"].map(lambda value: round(value, 4))
    print(printable_df.to_string(index=False))


def print_selected_errors(selected_errors):
    print("\nThree Misclassified Messages for Error Analysis:")
    if selected_errors.empty:
        print("No misclassified messages were found in this split.")
        return

    for idx, (_, row) in enumerate(selected_errors.iterrows(), start=1):
        print(f"\n{idx}. {row['error_type']}")
        print(f"Original Message: {row['message_text']}")
        print(f"Processed Text: {row['final_processed_text']}")
        print(f"Actual Label: {row['actual_label']} | Predicted Label: {row['predicted_label']}")
        print(f"Phishing Probability: {row['phishing_probability']:.2%}")
        print(f"Why It Was Wrong: {row['reason']}")


def plot_confusion_matrix(matrix, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_title("Holdout Confusion Matrix")
    ax.set_xticks([0, 1], labels=["Safe", "Phishing"])
    ax.set_yticks([0, 1], labels=["Safe", "Phishing"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, matrix[row, col], ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_model_benchmark(benchmark_df, output_path):
    plot_df = benchmark_df.set_index("model")[["cv_accuracy", "cv_f1", "cv_roc_auc", "cv_pr_auc"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title("Cross-Validation Model Benchmark")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_threshold_analysis(threshold_analysis_df, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threshold_analysis_df["threshold"], threshold_analysis_df["precision"], marker="o", label="Precision")
    ax.plot(threshold_analysis_df["threshold"], threshold_analysis_df["recall"], marker="o", label="Recall")
    ax.plot(threshold_analysis_df["threshold"], threshold_analysis_df["f1"], marker="o", label="F1")
    ax.set_title("Decision Threshold Analysis")
    ax.set_xlabel("Phishing Probability Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_research_artifacts(
    output_dir,
    model_output,
    df,
    benchmark_df,
    results,
    stemmer,
    best_model_name,
    dataset_path,
    decision_threshold,
    skip_plots=False,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    evaluation_summary_df = build_evaluation_summary(
        results,
        best_model_name,
        decision_threshold,
    )
    classification_report_df = pd.DataFrame(results["report_dict"]).transpose()

    df.to_csv(output_path / "cleaned_phishing_data.csv", index=False, encoding="utf-8-sig")
    benchmark_df.to_csv(output_path / "model_benchmark.csv", index=False, encoding="utf-8-sig")
    results["error_analysis_df"].to_csv(output_path / "error_analysis.csv", index=False, encoding="utf-8-sig")
    results["phishing_features"].to_csv(output_path / "top_phishing_indicators.csv", index=False, encoding="utf-8-sig")
    results["legitimate_features"].to_csv(output_path / "top_legitimate_indicators.csv", index=False, encoding="utf-8-sig")
    results["bigram_features"].to_csv(output_path / "top_bigram_indicators.csv", index=False, encoding="utf-8-sig")
    results["threshold_analysis_df"].to_csv(output_path / "threshold_analysis.csv", index=False, encoding="utf-8-sig")
    evaluation_summary_df.to_csv(output_path / "evaluation_summary.csv", index=False, encoding="utf-8-sig")
    classification_report_df.to_csv(output_path / "classification_report.csv", encoding="utf-8-sig")
    build_confusion_matrix_df(results["confusion_matrix"]).to_csv(
        output_path / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    model_bundle = {
        "model": results["prediction_model"],
        "feature_model": results["feature_model"],
        "vectorizer": results["vectorizer"],
        "stemmer": stemmer,
        "selected_model": best_model_name,
        "dataset_path": dataset_path,
        "decision_threshold": decision_threshold,
        "evaluation_summary": evaluation_summary_df.iloc[0].to_dict(),
    }
    dump(model_bundle, output_path / model_output)

    if not skip_plots:
        plot_confusion_matrix(results["confusion_matrix"], output_path / "confusion_matrix.png")
        plot_model_benchmark(benchmark_df, output_path / "model_benchmark.png")
        plot_threshold_analysis(results["threshold_analysis_df"], output_path / "threshold_analysis.png")

    return evaluation_summary_df


def predict_message(raw_message, stemmer, vectorizer, model, decision_threshold=0.5):
    clean_message, processed_text, stemmed_text = preprocess_text(raw_message, stemmer)
    X_new = vectorizer.transform([stemmed_text])
    phishing_probability = float(model.predict_proba(X_new)[0][1])
    predicted_label = int(phishing_probability >= decision_threshold)

    return {
        "raw_message": raw_message,
        "clean_message": clean_message,
        "processed_text": processed_text,
        "stemmed_text": stemmed_text,
        "predicted_label": predicted_label,
        "phishing_probability": phishing_probability,
        "decision_threshold": decision_threshold,
    }


def print_prediction(prediction):
    print("\nBlind Test Result:")
    print("Raw Message:", prediction["raw_message"])
    print("Cleaned Text:", prediction["clean_message"])
    print("Processed Text:", prediction["processed_text"])
    print("Decision Threshold:", prediction["decision_threshold"])
    print("Predicted Label:", prediction["predicted_label"])
    print("Phishing Probability:", f"{prediction['phishing_probability']:.2%}")
    print(
        "Verdict:",
        "Potential phishing message" if prediction["predicted_label"] == 1
        else "Looks legitimate according to the model",
    )


def maybe_run_blind_test(args, stemmer, vectorizer, model):
    if args.message:
        prediction = predict_message(
            args.message,
            stemmer,
            vectorizer,
            model,
            args.threshold,
        )
        print_prediction(prediction)
        return

    if args.interactive:
        raw_message = input("\nPaste a raw WhatsApp message for blind testing: ").strip()
        if raw_message:
            prediction = predict_message(
                raw_message,
                stemmer,
                vectorizer,
                model,
                args.threshold,
            )
            print_prediction(prediction)


def main():
    args = parse_args()
    df = load_dataset(args.dataset)
    df, stemmer = preprocess_dataframe(df)
    benchmark_df, benchmark_best_model_name = benchmark_models(df)
    final_model_name = (
        benchmark_best_model_name if args.final_model == "auto" else args.final_model
    )
    results = train_final_model(df, final_model_name, args.threshold)

    print("Original Text Sample:", df["message_text"].iloc[0])
    print("Cleaned Text Sample:", df["clean_message"].iloc[0])
    print("After Stopword Removal:", df["final_processed_text"].iloc[0])
    print("After Stemming:", df["stemmed_text"].iloc[0])
    print("Shape of X (Matrix):", results["X_shape"])
    print("Best Benchmark Model:", benchmark_best_model_name)
    print("Selected Final Model:", final_model_name)
    print("Decision Threshold:", args.threshold)
    print(f"Train/Test Split: {results['train_size']}/{results['test_size']}")
    print_benchmark_results(benchmark_df)
    print("\nHoldout Accuracy:", round(results["accuracy"], 4))
    print_probability_metrics(results)
    print("\nDetailed Research Report:")
    print(results["report"])
    print_confusion_matrix(results["confusion_matrix"])
    print_threshold_analysis(results["threshold_analysis_df"])
    print_feature_table("Top 15 Phishing Indicators:", results["phishing_features"])
    print_feature_table("Top 15 Legitimate Indicators:", results["legitimate_features"])
    print_feature_table("Top Bigram Indicators:", results["bigram_features"])
    print_selected_errors(results["selected_errors"])

    if PorterStemmer is None:
        print("\nNote: nltk is not installed, so stemming was skipped.")

    save_research_artifacts(
        args.output_dir,
        args.model_output,
        df,
        benchmark_df,
        results,
        stemmer,
        final_model_name,
        args.dataset,
        args.threshold,
        args.skip_plots,
    )
    maybe_run_blind_test(args, stemmer, results["vectorizer"], results["prediction_model"])


if __name__ == "__main__":
    main()
