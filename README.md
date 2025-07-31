# nexa_tax_classifier

##How to run

cd /path/to/nexa_tax_classifier
source .venv/bin/activate

python train_model.py --input data/sample_transactions-2.csv --output models/tax_deductible_clf.joblib
