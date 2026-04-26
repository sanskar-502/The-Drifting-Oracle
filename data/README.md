# 📂 Data Directory — Download Instructions

Place the following files in this directory:

## 1. Home Credit Default Risk (Training Distribution)
- **Source:** https://www.kaggle.com/c/home-credit-default-risk/data
- **File needed:** `application_train.csv`
- **Expected:** ~307K rows, 122 columns
- **Rename to:** `application_train.csv` (keep original name)

## 2. German Credit Dataset (Drift / Post-Inflation Batch)
- **Source:** https://www.kaggle.com/datasets/uciml/german-credit
- **File needed:** `german_credit_data.csv`
- **Expected:** 1,000 rows, ~10 columns
- **Rename to:** `german_credit_data.csv`

## 3. RBI/SEBI Policy Text (RAG Ground Truth)
- **File:** `rbi_sebi_policy.txt`
- **Status:** ✅ Already created by project setup

## Final Structure:
```
data/
├── application_train.csv      ← Download from Kaggle
├── german_credit_data.csv     ← Download from Kaggle
├── rbi_sebi_policy.txt        ← Already here
└── README.md                  ← This file
```
