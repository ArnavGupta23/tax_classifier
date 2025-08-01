import argparse, joblib, pandas as pd

from data_loader import load_transactions
from rules import rule_based_label

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def bootstrap_labels(df):
    """
    Apply rule_based_label and fill missing (None) with 0
    so we can train a first-pass model.
    """
    labels = []
    for txt in df["text"]:
        lbl, _ = rule_based_label(txt)   # import from rules.py
        # Treat None (no clear rule) as 0 for this bootstrap stage
        labels.append(0 if lbl is None else lbl)
    df["label"] = labels
    return df


def train_and_evaluate(input_csv, model_path):
    # Load + weak labels
    df = load_transactions(input_csv)
    df = bootstrap_labels(df)

    # Split
    X = df[["text", "merchant"]]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Preprocessing: text TF-IDF + merchant One-Hot
    preprocessor = ColumnTransformer([
        ("text", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1,3),
            sublinear_tf=True,
            min_df=2
        ), "text"),
        ("merchant", OneHotEncoder(handle_unknown="ignore"), ["merchant"])
    ])

    # Pipeline with hyperparameter tuning
    pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    param_grid = {
        "clf__C": [0.1, 1, 10],
        "clf__penalty": ["l2"]
    }
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    # Train
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print("Best params:", grid.best_params_)

    # Evaluate
    y_pred = best.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save
    joblib.dump(best, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/sample_transactions-2.csv")
    p.add_argument("--output", default="models/tax_deductible_clf.joblib")
    args = p.parse_args()
    train_and_evaluate(args.input, args.output)
