from NLP import (
    benchmark_models,
    load_dataset,
    predict_message,
    preprocess_dataframe,
    print_prediction,
    train_final_model,
)


def main():
    df = load_dataset("phishing_dataset_400_fixed.csv")
    df, stemmer = preprocess_dataframe(df)
    _, _ = benchmark_models(df)
    decision_threshold = 0.7
    results = train_final_model(df, "LinearSVC", decision_threshold)

    raw_message = input("Paste a raw WhatsApp message: ").strip()
    if not raw_message:
        print("No message provided.")
        return

    prediction = predict_message(
        raw_message,
        stemmer,
        results["vectorizer"],
        results["prediction_model"],
        decision_threshold,
    )
    print_prediction(prediction)


if __name__ == "__main__":
    main()
