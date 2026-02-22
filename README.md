# 💳 Credit Risk Analysis

A machine learning project to predict loan default risk using classification models, domain-driven feature engineering, and explainable AI — simulating real-world banking credit scoring systems with a real-time customer loan decision system.

---

## 📌 Problem Statement

Banks face significant financial losses when loans are approved for customers who later default. This project predicts which customers are likely to default (bad credit risk) so that banks can make informed loan approval decisions and minimize financial losses.

> **Key Focus:** Minimizing False Negatives — approving a bad customer causes direct financial loss to the bank.

---

## 📂 Dataset

- **Source:** [Credit Risk Customers — Kaggle](https://www.kaggle.com/datasets/ppb00x/credit-risk-customers)
- **Size:** 1,000 customers, 21 features
- **Target Variable:** `class` (Good / Bad)
- **Class Distribution:** 70% Good (700), 30% Bad (300) — Imbalanced
- **No missing values, No duplicates**

### Features:
| Feature | Description |
|---|---|
| `checking_status` | Status of existing checking account |
| `duration` | Loan duration in months |
| `credit_history` | Past credit history |
| `purpose` | Purpose of loan (car, education, business, etc.) |
| `credit_amount` | Loan amount requested |
| `savings_status` | Savings account status |
| `employment` | Employment duration |
| `installment_commitment` | Installment rate as % of income |
| `personal_status` | Personal status and gender |
| `age` | Customer age |
| `housing` | Housing type (own/free/rent) |
| `job` | Job type |
| `foreign_worker` | Whether customer is a foreign worker |

---

## 🔍 Exploratory Data Analysis

Key findings from EDA:

- Customers with **higher credit amount** are more likely to default
- Customers with **lower savings status** show higher default rates
- **Loan purpose** significantly impacts default risk
- **Housing type** affects risk — free housing customers show higher default
- Duration and credit amount are strongly correlated (0.62)

---

## ⚙️ Feature Engineering

Created 4 new domain-driven features:

| Feature | Formula | Business Insight |
|---|---|---|
| `monthly_burden` | credit_amount / duration | Higher monthly repayment = Higher risk |
| `credit_per_age` | credit_amount / age | Young customers with large loans = Risky |
| `high_credit` | credit_amount > median (0/1) | Flag for above-average loan amounts |
| `long_duration` | duration > median (0/1) | Flag for long-term loans |

> Domain-driven feature engineering provides stronger signals to the model beyond raw features.

---

## ⚙️ Data Preprocessing

- **No missing values** — dataset was clean
- **Label Encoding** with separate encoders per column (13 categorical columns)
- **Standard Scaling** on numerical features: age, credit_amount, duration, monthly_burden, credit_per_age
- **Stratified Train-Test Split** (80/20) — maintains 70/30 class ratio in both sets
- **SMOTE** — balanced training data from 240 → 560 bad samples

---

## 🤖 Models & Results

### ROC-AUC Comparison:

| Model | ROC-AUC | Accuracy |
|---|---|---|
| Logistic Regression | 0.755 | 71% |
| Random Forest | 0.788 | 73% |
| **XGBoost** | **0.793** 🏆 | 73% |

### False Negatives Comparison (Most Critical Metric):

| Model | False Negatives |
|---|---|
| Logistic Regression | 23 |
| Random Forest | 29 |
| XGBoost Original | 27 |
| **XGBoost + Threshold Tuning** | **17** 🏆 |

### XGBoost Improvement Strategy:

Adjusted decision threshold from **0.5 → 0.3** — model becomes more sensitive to bad customers:

- False Negatives reduced: **27 → 17** ✅
- Outperformed Logistic Regression (23) by catching 6 additional bad customers!

> **Why False Negatives matter more than Accuracy?**
> Predicting a bad customer as good = bank approves loan = direct financial loss.

---

## 📊 Visualizations

- Credit Risk Distribution (Count Plot)
- Age vs Credit Risk (Box Plot)
- Credit Amount vs Risk (Box Plot)
- Loan Purpose vs Risk (Count Plot)
- Housing vs Credit Risk (Count Plot)
- Saving Accounts vs Credit Risk (Count Plot)
- Monthly Burden vs Risk (Box Plot)
- Age Group vs Credit Risk (Count Plot)
- Credit per Age vs Risk (Box Plot)
- Correlation Heatmap
- Confusion Matrix — All 3 Models
- ROC Curve Comparison
- Model Comparison Bar Chart (ROC-AUC)

---

## 🔎 Model Explainability — SHAP

Used **SHAP (SHapley Additive Explanations)** with XGBoost to interpret predictions:

- **Summary Bar Plot** — Top features driving default globally
- **Beeswarm Plot** — Feature impact distribution across all customers
- **Waterfall Plot** — Individual customer prediction explanation
- **Force Plot** — Push/pull of features for single customer prediction
- **Dependence Plot** — How credit_amount affects default probability

> SHAP explainability is critical in banking — regulators require banks to explain why a loan was rejected. This makes the model production-ready and compliant.

---

## 🏦 Real-Time Loan Decision System

Built a `predict_customer()` function that takes a new customer's details and outputs an instant loan decision:

```
=============================================
🏦 BANK LOAN DECISION SYSTEM
=============================================
Customer Age           : 52
Credit Amount          : ₹2,000
Duration               : 12 months
Monthly Burden         : ₹167/month
---------------------------------------------
Default Probability    : 11.1%
Risk Level             : LOW RISK
Decision               : ✅ LOAN APPROVE — Low Default Risk!
=============================================
```

---

## 💡 Business Insights

- Customers with **little/no savings + high credit amount** = highest risk segment
- **Threshold tuning** (0.5 → 0.3) caught 10 more bad customers than original XGBoost
- **Monthly burden** feature was a strong default predictor — domain knowledge pays off!
- Model can be used as a **real-time loan pre-screening tool** before manual review

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn, SHAP, Matplotlib, Seaborn
- **Environment:** Google Colab

---

## 📁 Project Structure

```
credit-risk-analysis/
│
├── data/
│   └── credit_customers.csv
├── notebook/
│   └── Credit_Risk_Analysis.ipynb
└── README.md
```

---

## 🚀 How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/credit-risk-analysis.git
   ```
2. Install required libraries
   ```bash
   pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn
   ```
3. Open the notebook
   ```bash
   jupyter notebook Credit_Risk_Analysis.ipynb
   ```

---

## 📈 Conclusion

This project built an end-to-end credit risk prediction pipeline with strong business focus. XGBoost achieved the best ROC-AUC of 0.793. By tuning the decision threshold from 0.5 to 0.3, false negatives were reduced from 27 to 17 — outperforming all baseline models. Domain-driven feature engineering and SHAP explainability make this model interpretable and production-ready for real-world banking applications.

---

## 📌 GitHub Repository Description

```
Credit risk prediction system using XGBoost with threshold tuning & SMOTE. 
Reduced false negatives to 17 with domain-driven feature engineering and SHAP explainability. 
Includes real-time customer loan decision system. ROC-AUC: 0.793
```

## 🏷️ Topics/Tags

```
credit-risk, loan-default, machine-learning, xgboost, shap, smote,
feature-engineering, python, scikit-learn, finance, banking, classification
```
