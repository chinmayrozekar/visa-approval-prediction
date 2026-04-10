# Visa Approval Prediction — Advanced Ensemble ML (EasyVisa)

Ensemble classification models to predict US visa application outcomes (Certified vs Denied) on 25,480 OFLC applications. Helps consultancies automate initial screening and route borderline cases to specialist reviewers.

## Business Problem
The OFLC processed 775,979 employer applications in FY 2016 — a 9% YoY increase. Manual review of every case is unsustainable. This project builds a model to automate initial classification, freeing reviewers for ambiguous applications.

## Dataset
- `EasyVisa.csv` — 25,480 rows × 12 columns
- Target: `case_status` (Certified / Denied)
- Key features: applicant education, region of employment, prior job experience, employer size, number of employees, prevailing wage

## Approach
1. EDA — fixed negative values in employee count; confirmed Northeast region dominates (>50% of cases); 88%+ of applicants had prior experience
2. Feature engineering — one-hot encoding for categorical variables; analysed education and experience distributions
3. Class imbalance — applied SMOTE oversampling for minority class
4. Modeling — AdaBoost, Random Forest, Gradient Boosting (GBM), XGBoost; hyperparameter tuning via RandomizedSearchCV
5. Model selection — GBM selected as best balance of recall and precision on held-out test data

## Results

| Model | Accuracy | Recall | Precision | F1 |
|-------|----------|--------|-----------|-----|
| AdaBoost (tuned) | 73.5% | 84.0% | 78.0% | 80.9% |
| Random Forest (tuned) | 70.6% | 72.1% | 81.7% | 76.6% |
| **GBM (tuned) — selected** | **75.2%** | **86.5%** | **78.5%** | **82.3%** |
| XGBoost (tuned) | 71.2% | 94.9% | 71.4% | 81.5% |

**Selected model:** Gradient Boosting — best overall F1 (82.3%) and accuracy (75.2%) on test data.
Use XGBoost variant if the business cost of a missed Denied case outweighs false positives (recall: 94.9%).

## Key Findings
- Bachelor's and Master's degree holders have higher certification rates; education is a meaningful predictor
- Northeast region accounts for >50% of employment cases; regional imbalance is important for sampling
- 88%+ of applicants had prior job experience — experience alone is not a strong differentiator
- Ensemble models substantially outperform logistic regression baselines on this dataset

## Recommendations
- Deploy GBM as a screening layer to auto-approve high-confidence Certified applications
- Route likely-Denied applications to a specialist team; use XGBoost for higher-recall risk detection
- Retrain quarterly as OFLC policy and labour market conditions evolve

## Technologies
Python · Pandas · NumPy · Scikit-learn · XGBoost · imbalanced-learn (SMOTE) · Matplotlib · Seaborn · Jupyter Notebook

## Code
Notebook: [`EasyVisa_Approval_Prediction.ipynb`](EasyVisa_Approval_Prediction.ipynb)

---
*Author: Chinmay Rozekar*
