# Loan Approval Risk Classification

**Statistical Software II — Final Assignment**  
Berat Ercan · Marmara University, Department of Statistics

---

## Overview

This project applies and benchmarks **9 classification algorithms** on the [Financial Risk for Loan Approval](https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval) dataset from Kaggle. The full pipeline covers data cleaning, multicollinearity and data leakage detection, class imbalance handling, model training with hyperparameter optimization, and final comparison by Accuracy and F1-Score.

---

## Dataset

| Property | Value |
|---|---|
| Source | Kaggle — Financial Risk for Loan Approval |
| Rows | 20,000 |
| Original Features | 36 |
| Target | `LoanApproved` (binary: 0 / 1) |
| Class Distribution | ~75% rejected, ~25% approved |

---

## Project Structure

```
├── Statistical_Software_II_Final_Berat_Ercan.ipynb   # Main notebook
├── Loan.csv                                            # Dataset (add manually)
└── README.md
```

> **Note:** `Loan.csv` is not included in this repository. Download it from the Kaggle link above and place it in the root directory before running the notebook.

---

## Pipeline

### 1. Data Preprocessing
- Dropped `ApplicationDate` (datetime, non-predictive)
- Correlation matrix analysis → identified multicollinearity candidates
- VIF analysis → removed features with VIF > 10
- Data leakage check → dropped `InterestRate`, `MonthlyLoanPayment`, `RiskScore`, `TotalDebtToIncomeRatio`
- Additional drops for semantic redundancy: `MonthlyIncome` (≈ `AnnualIncome`), `NetWorth` (≈ `TotalAssets`), `Experience` (≈ `Age`)

### 2. Class Imbalance
- ~75/25 split detected
- Handled via `class_weight="balanced"` on applicable models

### 3. Models & Optimization

| # | Model | Optimization |
|---|---|---|
| 1 | Logistic Regression | `class_weight="balanced"` |
| 2 | Naive Bayes (GaussianNB) | — |
| 3 | K-Nearest Neighbors | GridSearchCV on `k` |
| 4 | Linear SVM | GridSearchCV on `C` |
| 5 | Non-Linear SVM (RBF) | GridSearchCV on `C`, `gamma` |
| 6 | Artificial Neural Network (MLP) | GridSearchCV on layers, activation |
| 7 | Decision Tree (CART) | GridSearchCV on `max_depth` |
| 8 | Random Forest | RandomizedSearchCV |
| 9 | Gradient Boosting | GridSearchCV |
| 10 | XGBoost | GridSearchCV |

---

## Key Results

| Model | Accuracy | F1-Score |
|---|---|---|
| Logistic Regression | ~0.85 | ~0.85 |
| Naive Bayes | ~0.86 | ~0.86 |
| KNN (optimized) | ~0.87 | ~0.87 |
| **Linear SVM (optimized)** | **best** | **best** |
| Non-Linear SVM | slight drop after optimization | — |
| ANN | ~marginal gain after optimization | — |
| CART (optimized) | notable improvement | — |
| Random Forest | strong, minimal drop after optimization | — |
| Gradient Boosting | stable | — |
| XGBoost | small improvement after optimization | — |

**Winner: Linear SVM (post-optimization)** — best combined Accuracy and F1-Score across all tested models.

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

---

## Usage

```bash
git clone https://github.com/<your-username>/loan-approval-classification.git
cd loan-approval-classification
# Place Loan.csv in this directory
jupyter notebook Statistical_Software_II_Final_Berat_Ercan.ipynb
```
