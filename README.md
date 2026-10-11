# Customer Churn Prediction and Analytics System

**Course:** Python for Data Science (BE05000231)  
**Project:** Customer Churn Prediction and Retention Analytics

## Overview
This project analyzes telecom customer records, creates derived features, explores churn patterns, runs selected statistical tests, and compares three classification models: Logistic Regression, Decision Tree, and Random Forest.

## Dataset
Place the IBM Telco Customer Churn CSV in this folder and name it:

`telecom_customer_churn.csv`

The implementation expects either:
- `Customer Status` with values such as `Stayed`, `Churned`, and `Joined`; or
- a binary `Churn` column with values such as `Yes` and `No`.

The script uses the exact feature names described in the project presentation when available, including `Tenure in Months`, `Monthly Charge`, `Total Revenue`, `Online Security`, `Online Backup`, `Device Protection Plan`, `Premium Tech Support`, and `Contract`. Missing optional feature columns are skipped.

**Important:** Dataset versions differ. Check the actual CSV column names before running the script. The CSV is not included in this starter package.

## Features
- CSV loading and initial inspection
- Missing-value handling through scikit-learn pipelines
- Feature engineering: tenure group, service count, and revenue per tenure month
- EDA charts and optional correlation heatmap
- Excludes Customer ID and post-churn fields (`Churn Category`, `Churn Reason`) from model inputs to reduce data leakage
- Welch's t-test for Monthly Charge by churn class when available
- Chi-square test for Contract vs churn when available
- Train/test split and preprocessing
- Logistic Regression, Decision Tree, and Random Forest comparison
- Accuracy, precision, recall, F1 score, ROC AUC (when available)
- Confusion matrix and feature importance plots
- Exported model metrics and test-set predictions

## Installation
Use Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
```

## Run
Place `telecom_customer_churn.csv` next to `churn_prediction.py`, then run:

```bash
python churn_prediction.py
```

Generated charts and CSV outputs are written to `outputs/`.

## Target definition
When `Customer Status` is used, `Churned` is the positive class and `Stayed`/`Joined` are grouped as `Not Churned`. This is a binary classification choice for this implementation and should be explained in the project report.

## Limitations
- Model performance depends on the exact dataset and its columns.
- A random train/test split provides an estimate, not a guarantee of real-world performance.
- Feature importance does not establish causation.
- This is an educational prototype, not a deployed retention system.
