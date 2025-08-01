# 🧾 Tax-Deductible Transaction Classifier

This project classifies bank transactions as either **tax-deductible** or **non-deductible** using a hybrid system that combines **rule-based logic** and a **machine learning model**.

---

## 📁 Project Structure

```
├── app.py                # Streamlit web app interface
├── classifier.py         # ML prediction logic
├── data_loader.py        # Preprocessing and CSV loading
├── rules.py              # Rule matching function
├── rules_config.py       # Regex-based tax rules
├── train_model.py        # Training pipeline for ML model
├── requirements.txt      # Python dependencies
``` 

---

## ⚙️ How It Works

### Hybrid Classification Logic

1. **Rule-Based Classifier**: Uses regex patterns (e.g., business travel, meals, equipment) to assign labels and explanations.
2. **ML Classifier**: Falls back to a trained logistic regression model when no rule matches, using TF-IDF on text and one-hot encoding on merchant name.

The model outputs a clear explanation for every prediction.

---

## 🧪 Input Format

The input CSV must contain:
- `date` – e.g., `2024-05-10`
- `amount` – e.g., `125.75`
- `merchant` – e.g., `Delta Airlines`
- `description` – e.g., `Flight to NYC for business conference`

---

## 📊 Output Format

The output is available in both CSV and JSON formats. Each record contains:
- `date`
- `merchant`
- `description`
- `deductible` – `true` or `false`
- `reason` – explanation (e.g., "Business travel", or "ML (p=0.84)")

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Retrain model
```bash
python train_model.py --input data/sample_transactions-2.csv --output models/tax_deductible_clf.joblib
```

### 3. Launch web app
```bash
streamlit run app.py
```

You can upload your own transaction CSV or load the sample.

---

## 🎯 Features

- ✅ Explainable rule-based deductions
- ✅ ML fallback with threshold tuning
- ✅ Streamlit UI with file upload and downloads
- ✅ JSON + CSV export support
- ✅ Modular and extensible codebase

---

## 📄 Assumptions

- Transactions that don’t match any rule default to ML classification.
- The threshold for ML confidence is set to 0.5 by default (user-adjustable).
- Rule matching is prioritized by order — first match wins.

---

## 🏁 Sample JSON Output

```json
[
  {
    "date": "2025-07-25",
    "merchant": "Apple",
    "description": "MacBook purchase for work",
    "deductible": true,
    "reason": "Business equipment purchase"
  }
]
```

---

## 🙋‍♂️ Author

**Arnav Gupta**  
AI/ML Internship Candidate  
[arnavgupta.info](https://arnavgupta.info)

---

source .venv/bin/activate