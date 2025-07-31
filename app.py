import streamlit as st
import pandas as pd
import json

from data_loader import load_transactions
from classifier import classify_text

def main():
    # ─── Header ────────────────────────────────────────────────────────────────
    st.title("Tax-Deductible Transaction Classifier")
    st.markdown(
        "Upload your own transactions or click **Load Sample Transactions** "
        "to see the model in action. Toggle between rule-based and ML-based."
    )

    # ─── Sidebar Controls ──────────────────────────────────────────────────────
    use_ml = st.sidebar.checkbox("Use ML model", value=True)
    threshold = st.sidebar.slider(
        "Deductible probability threshold",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )

    # ─── Data Loading Options ──────────────────────────────────────────────────
    df = None

    # 1) Button to load the provided sample CSV
    if st.button("Load Sample Transactions"):
        df = load_transactions("data/sample_transactions-2.csv")
        st.success("Sample data loaded!")

    # 2) File uploader for custom CSVs
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        df = load_transactions(uploaded)
        st.success("Your file has been uploaded!")

    # If neither sample nor upload, prompt the user and exit
    if df is None:
        st.info("🔍 Upload a CSV or click **Load Sample Transactions** to begin.")
        return

    # ─── Classification ────────────────────────────────────────────────────────
    # For each transaction row, run ML or rule-based classification
    labels, reasons = [], []
    for _, row in df.iterrows():
        # classify_text returns (0 or 1, reason string)
        label, reason = classify_text(row["text"], use_ml=use_ml)

        # If ML is selected, re-apply the threshold from the sidebar
        if use_ml:
            # Extract the probability from the reason string
            # e.g. reason="ML (p=0.87)" → prob=0.87
            prob = float(reason.split("p=")[1].rstrip(")"))
            label = int(prob >= threshold)
            reason = f"ML (p={prob:.2f})"

        labels.append(label)
        reasons.append(reason)

    # Attach results back to the DataFrame
    df["deductible"] = labels
    df["reason"]      = reasons

    # Convert numeric label to boolean for JSON output
    df["deductible"] = df["deductible"].astype(bool)

    # ─── Display ───────────────────────────────────────────────────────────────
    st.subheader("🚀 Classification Results")
    st.dataframe(
        df[["date", "merchant", "description", "deductible", "reason"]],
        use_container_width=True
    )

    # ─── Downloads ─────────────────────────────────────────────────────────────
    # 1) CSV
    csv_data = df.to_csv(index=False)
    st.download_button(
        "Download results as CSV",
        csv_data,
        "predictions.csv",
        "text/csv"
    )

    # 2) JSON (matching your schema)
    records = df[["date", "merchant", "description", "deductible", "reason"]] \
        .to_dict(orient="records")
    json_data = json.dumps(records, indent=2)
    st.download_button(
        "Download results as JSON",
        json_data,
        "predictions.json",
        "application/json"
    )

if __name__ == "__main__":
    main()
