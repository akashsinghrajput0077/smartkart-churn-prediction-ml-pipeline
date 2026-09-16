# SmartKart — Customer Churn Prediction ML Pipeline

A complete, end-to-end machine learning pipeline that predicts customer churn for **SmartKart**, a fictional e-commerce business, using **Logistic Regression**. Built as a BBA AI/ML coursework project (Chitkara Business School, CLO02), this notebook takes a deliberately messy, real-world-style dataset through all 15 standard stages of an ML pipeline — from raw CSV to a business-ready churn risk report.

## Business Problem

SmartKart wants to know **which customers are likely to churn (leave)** so the retention team can proactively intervene (e.g., send offers, resolve complaints) before losing them. The goal is not just a working model, but an actionable list of at-risk customers ranked by churn probability.

## Dataset

`SmartKart_dirty_100_rows.csv` — 100 customer records, **intentionally messy**, containing:
- Duplicate rows
- Missing values
- Invalid entries (e.g., `Age` stored as text like `"thirty"`, whitespace, negative ages)
- Outliers (e.g., `Monthly_Spend = 99999`, `Complaints = 50`)

**Columns:** `Customer_ID`, `Age`, `Monthly_Spend`, `Complaints`, `Churn` (target: 1 = churned, 0 = stayed)

## Pipeline Overview (15 Steps)

| Step | Stage | What Happens |
|------|-------|---------------|
| 1 | Data Collection | Upload and load the raw CSV |
| 2 | Data Understanding | Inspect shape, dtypes, missing values, duplicates |
| 3 | Data Cleaning | Strip whitespace, remove duplicates, fix invalid types, impute missing values with median |
| 4 | Outlier Detection & Treatment | IQR method to cap extreme values in `Monthly_Spend` and `Complaints` |
| 5 | Feature Selection | Keep `Age`, `Monthly_Spend`, `Complaints`; drop `Customer_ID` |
| 6 | Define Target Variable | `Churn` (binary: 0/1) |
| 7 | Encode Target Variable | Verify `Churn` is already numeric |
| 8 | Train-Test Split | 80/20 stratified split |
| 9 | Feature Standardisation | `StandardScaler` fit on train, applied to test |
| 10 | Model Building | Instantiate `LogisticRegression` |
| 11 | Model Training | Fit model on scaled training data |
| 12 | Prediction | Predict classes and probabilities on the test set |
| 13 | Model Evaluation | Confusion matrix, accuracy, precision, recall, F1-score |
| 14 | Model Interpretation | Examine coefficients to explain *why* customers churn |
| 15 | Final Output | Generate `smartkart_churn_risk_report.csv` — a ranked, business-ready action list |

## Key Results

- **Accuracy:** ~89–95%
- **Recall:** ~100% (the model catches essentially every real churner in the test set — critical for a churn-prevention use case, since missing a churner is costlier than a false alarm)
- **Precision:** ~83–91%
- **Top churn drivers:**
  - Higher `Monthly_Spend` → **lower** churn risk (high-value customers are more loyal)
  - More `Complaints` → **higher** churn risk (the clearest actionable lever for the business)

## Deliverable

`smartkart_churn_risk_report.csv` — a table of test customers labeled **"Likely to Churn"** or **"Not Likely to Churn"**, ranked by churn probability, ready for the retention/marketing team to act on.

## Tech Stack

- Python 3
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn (`LogisticRegression`, `StandardScaler`, `train_test_split`, classification metrics)
- Google Colab (recommended environment)

## Getting Started

1. Open `SmartKart_Churn_Prediction_ML_Pipeline.ipynb` in Google Colab or Jupyter.
2. Run Step 1 and upload `SmartKart_dirty_100_rows.csv` when prompted.
3. Run all remaining cells sequentially — each step includes markdown explanations of *what* and *why*.
4. Download the generated `smartkart_churn_risk_report.csv` at the end.

## Repository Structure

```
├── SmartKart_Churn_Prediction_ML_Pipeline.ipynb   # Main pipeline notebook
├── SmartKart_dirty_100_rows.csv                   # Raw input dataset
├── smartkart_churn_risk_report.csv                # Generated output (business report)
└── README.md                                      # Project documentation
```

## Learning Objectives

- Apply the full data preprocessing lifecycle: inspection, cleaning, outlier treatment.
- Perform feature selection and target encoding for a supervised classification problem.
- Correctly split and scale data to avoid data leakage.
- Train and evaluate a Logistic Regression model using standard classification metrics.
- Translate model coefficients into business-relevant insights.
- Produce a decision-ready output for a non-technical business team.

## License

This project is intended for educational purposes as part of a BBA AI/ML coursework practical.
