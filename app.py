import streamlit as st
import pandas as pd
import json

from data_loader import load_transactions
from rules import rule_based_label
from classifier import _pipeline  # your trained ML pipeline

def main():
    st.title("Tax-Deductible Transaction Classifier")
    st.markdown(
        "Upload your own transactions or click **Load Sample Transactions** "
        "to see the model in action. Toggle between rule-based and ML-based."
    )

    # ─── Sidebar Controls ──────────────────────────────────────────────────────
    use_ml = st.sidebar.checkbox("Use ML model", value=True)
    threshold = 0.5
    if st.sidebar.checkbox("Edit Threshold", value=False):
        threshold = st.sidebar.slider(
            "Deductible probability threshold",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05
        )

    # ─── Data Loading Options ──────────────────────────────────────────────────
    df = None
    if st.button("Load Sample Transactions"):
        df = load_transactions("data/sample_transactions-2.csv")
        st.success("Sample data loaded!")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        df = load_transactions(uploaded)
        st.success("Your file has been uploaded!")

    if df is None:
        st.info("🔍 Upload a CSV or click **Load Sample Transactions** to begin.")
        return

    # ─── Classification ────────────────────────────────────────────────────────
    records = []  # Store final classification results

    for _, row in df.iterrows():
        text = row["text"]
        merchant = row["merchant"]

        # Try rule-based classification first
        rule_label, rule_reason = rule_based_label(text)

        if rule_label is not None:
            final_label = bool(rule_label)
            final_reason = rule_reason
        elif use_ml:
            # Fallback to ML if no rule matched and ML is enabled
            X = pd.DataFrame([{"text": text, "merchant": merchant}])
            prob = _pipeline.predict_proba(X)[0, 1]
            final_label = bool(prob >= threshold)
            final_reason = f"{rule_reason}; ML (p={prob:.2f})"
        else:
            # Default to non-deductible if no rule and ML disabled
            final_label = False
            final_reason = rule_reason

        # Append result to output list
        records.append({
            "date": row["date"],
            "merchant": merchant,
            "description": row["description"],
            "deductible": bool(final_label),
            "reason": final_reason
        })


    # ─── Display ───────────────────────────────────────────────────────────────
    out_df = pd.DataFrame(records)
    st.subheader("Classification Results")
    st.dataframe(out_df, use_container_width=True)

    # ─── Downloads ─────────────────────────────────────────────────────────────
    csv_data = out_df.to_csv(index=False)
    st.download_button("Download results as CSV", csv_data, "predictions.csv", "text/csv")

    json_data = json.dumps(records, indent=2)
    st.download_button("Download results as JSON", json_data, "predictions.json", "application/json")

if __name__ == "__main__":
    main()
