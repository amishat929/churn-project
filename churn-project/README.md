# Customer Churn Prediction: end-to-end data analytics project

Predict which telecom customers are about to leave, explain **why**, and turn the prediction into a
**money number** that a business can act on. Includes a Streamlit app you can run (or deploy) in one command.

| | |
|---|---|
| **Data** | IBM Telco Customer Churn, 7,043 customers, 20 features, 26.5% churn |
| **Models** | Logistic Regression, Random Forest, XGBoost (compared with 5-fold CV) |
| **Imbalance** | SMOTE inside the pipeline (applied only to training folds, no leakage) |
| **Explainability** | SHAP: global drivers + per-customer reasons |
| **App** | Streamlit: predict, insights, model report, business-impact calculator |

## Results

| Model | CV ROC-AUC | Test ROC-AUC | Recall @0.5 | F1 @0.5 |
|---|---|---|---|---|
| Logistic Regression | 0.8445 | 0.8399 | 0.786 | 0.614 |
| Random Forest | 0.8433 | 0.8395 | 0.679 | 0.609 |
| **XGBoost** (selected) | **0.8453** | 0.8424 | 0.623 | 0.601 |

All three models score almost the same (ROC-AUC about 0.84). That is typical for this dataset: the
information is in the features, not in model complexity. The model is chosen on **cross-validated** AUC,
not on the test set.

**Top churn drivers (SHAP):** month-to-month contract, short tenure, no online security, fibre-optic
internet, no tech support, paying by electronic check.

**Business result:** with the default assumptions ($20 per offer, $500 per saved customer, 30% of
offers succeed) the profit-maximising rule is to contact customers whose score is >= 0.13. On the test
set this earns **$35,850 vs $27,920** for sending an offer to everyone. The threshold is picked on
out-of-fold training predictions, so the test number is an honest estimate.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train.py            # ~20 s: trains models, saves model + charts to models/ and reports/
streamlit run app.py       # opens http://localhost:8501
python tests/test_app.py   # optional smoke tests
```

If you skip `train.py`, the app trains the model automatically on first launch.

## Project structure

```
churn-project/
├── app.py                  # Streamlit app (4 tabs)
├── train.py                # training, evaluation, threshold tuning, SHAP, charts
├── requirements.txt
├── data/telco_raw.csv      # dataset
├── src/
│   ├── preprocess.py       # cleaning + feature engineering
│   ├── business.py         # cost-benefit maths
│   ├── explain.py          # SHAP helpers
│   └── theme.py            # shared chart colours
├── models/                 # churn_model.joblib, metrics.json, comparison table
├── reports/                # ROC, confusion matrix, SHAP, net-benefit PNGs
└── tests/test_app.py       # smoke tests
```

## What was done (pipeline)

1. **Cleaning:** 11 blank `TotalCharges` values (new customers, tenure 0) set to 0; target mapped to 0/1.
2. **Feature engineering:** `NumServices` (count of add-on services), `AvgChargePerMonth`, `TenureGroup`.
3. **Modelling:** one-hot encoding + scaling + SMOTE + classifier in a single `imblearn` pipeline;
   stratified 80/20 split; 5-fold stratified CV.
4. **Threshold tuning:** instead of the default 0.5, the threshold maximises expected profit
   (`saved churners x success rate x value - offers sent x offer cost`), chosen on out-of-fold predictions.
5. **Explainability:** SHAP values give global importance and the "why this score" chart per customer.
6. **App:** rule-based retention actions driven by the customer's risk factors.

## Limitations (good to mention in an interview)

- SMOTE shifts predicted probabilities upward, so the app shows a **risk score** (good for ranking),
  not a calibrated probability. Calibration (`CalibratedClassifierCV`) would be a natural next step.
- Offer cost, customer value and success rate are **assumptions**; real values would come from the
  business (ideally from an A/B test of retention offers).
- The dataset is a snapshot with no timestamps, so there is no time-based validation.
- Total charges in the app are estimated as tenure x monthly charges.

## Ideas to extend

- Calibrate probabilities and compare SMOTE vs `class_weight`
- Hyper-parameter tuning with Optuna
- Customer lifetime value (CLV) segments to prioritise offers
- Deploy free on Streamlit Community Cloud (push to GitHub, then connect the repo)

*Dataset: IBM sample data, "Telco Customer Churn".*
