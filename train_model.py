import argparse
import joblib
import pandas as pd

from data_loader import load_transactions
from rules import rule_based_label
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def bootstrap_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the rule_based_label function to generate a 'label' column.
    """
    df["label"] = df["text"].apply(rule_based_label)
    return df

def train_and_evaluate(input_csv: str, model_path: str):
    """
    1. Load data and bootstrap labels
    2. Split into train/test
    3. Build TF‑IDF + LogisticRegression pipeline
    4. Train, evaluate, and save the model
    """
    # 1) Load + bootstrap
    df = load_transactions(input_csv)
    df = bootstrap_labels(df)

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    # 3) Define the pipeline
    pipeline = Pipeline([
        # Convert text to TF‑IDF features (unigrams + bigrams)
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2))),
        # Logistic Regression classifier
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # 4) Train
    pipeline.fit(X_train, y_train)

    # Evaluate on hold‑out
    y_pred = pipeline.predict(X_test)
    print("=== EVALUATION ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Show top “deductible” and “non‑deductible” indicators
    feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
    coefs = pipeline.named_steps["clf"].coef_[0]
    top_pos = sorted(zip(coefs, feature_names), reverse=True)[:10]
    top_neg = sorted(zip(coefs, feature_names))[:10]
    print("Top deductible indicators:", top_pos)
    print("Top non-deductible indicators:", top_neg)

    # 5) Save the trained pipeline for later use
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the tax‑deductible classifier")
    parser.add_argument(
        "--input", default="data/sample_transactions-2.csv",
        help="Path to the raw transactions CSV"
    )
    parser.add_argument(
        "--output", default="models/tax_deductible_clf.joblib",
        help="Where to save the trained model"
    )
    args = parser.parse_args()
    train_and_evaluate(args.input, args.output)
