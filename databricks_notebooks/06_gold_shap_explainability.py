# Databricks notebook source
# MAGIC %pip install xgboost shap -q

"""
============================================================
  DRIFTING ORACLE — NOTEBOOK 06: SHAP EXPLAINABILITY
============================================================
  Phase 6: Interpretable Machine Learning (Gold Layer)
  
  Deploys SHAP (SHapley Additive exPlanations) to crack open
  the black-box machine learning model (XGBoost).
  
  Extracts:
    1. Global Feature Importance (Average impact across all)
    2. Local Explanation (Why was Applicant X rejected/approved?)
       - Top 3 positive drivers (Pushing Risk UP)
       - Top 3 negative drivers (Pushing Risk DOWN)
       
  Serializes results to: gold_shap_explanations
============================================================
"""

import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import shap

# ── Databricks Pipeline Constants ───────────────────────────
CATALOG_NAME = "workspace"
SCHEMA_NAME = "default"

SILVER_GERMAN_CREDIT = f"{CATALOG_NAME}.{SCHEMA_NAME}.silver_german_credit"
PROXY_FEATURES = ["age", "duration_months", "credit_amount", "income_proxy", "employment_years"]
MODEL_NAME = "credit_risk_model"
GOLD_SHAP_EXPLANATIONS = f"{CATALOG_NAME}.{SCHEMA_NAME}.gold_shap_explanations"

print("\n" + "=" * 70)
print("  🔮 DRIFTING ORACLE — Phase 6: Interpretable ML (SHAP)")
print("=" * 70)

print(f"  ✅ Using Native Databricks SparkSession")
GOLD_SHAP_EXPLANATIONS = "gold_shap_explanations"

# ─────────────────────────────────────────────────────────────
#  STEP 1: Load Data & Champion Model
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  📥 STEP 1: Fetching Active @Champion & Inference Data")
print("─" * 60)

# Fetch latest German data (Newest Applicants) to explain their decisions
df_german_spark = spark.read.table(SILVER_GERMAN_CREDIT)

print("  ⏳ Converting Spark DataFrame to Pandas for SHAP analysis...")
df_applicants = df_german_spark.select(*PROXY_FEATURES).toPandas().dropna()
df_applicants.reset_index(drop=True, inplace=True)

# Fetch current MLflow Champion Model
mlflow.set_registry_uri("databricks-uc")
full_model_uri = f"models:/{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}@Champion"

print(f"  ⏳ Downloading @Champion Model: {full_model_uri}")

# Load model dynamically — Champion could be XGBoost, RandomForest, or LogisticRegression
# Try XGBoost first (most common winner), fall back to sklearn
try:
    model = mlflow.xgboost.load_model(full_model_uri)
    model_flavor = "xgboost"
except Exception:
    model = mlflow.sklearn.load_model(full_model_uri)
    model_flavor = "sklearn"

print(f"  ✅ Loaded Champion ({model_flavor} flavor)")

# Generate baseline predictions for these applicants
y_prob = model.predict_proba(df_applicants)[:, 1]

# ─────────────────────────────────────────────────────────────
#  STEP 2: Global SHAP Explainer (The Macro View)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  🌍 STEP 2: Calculating Global Feature Importance")
print("─" * 60)

# Dynamically select SHAP explainer based on model type
model_type_str = str(type(model))
if "LogisticRegression" in model_type_str:
    print("  ⏳ Fitting SHAP LinearExplainer to the Logistic Regression Model...")
    explainer = shap.LinearExplainer(model, df_applicants)
else:
    print("  ⏳ Fitting SHAP TreeExplainer to the Tree-Based Model...")
    explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(df_applicants)

# Global importance is the mean absolute SHAP value across all applicants
global_shap_abs_mean = np.abs(shap_values).mean(axis=0)
global_importance = dict(zip(PROXY_FEATURES, global_shap_abs_mean))

print("  📊 GLOBAL SHAP IMPORTANCE (What drives the system macroscopically?):")
sorted_global = sorted(global_importance.items(), key=lambda x: -x[1])
for i, (feat, imp) in enumerate(sorted_global, 1):
    print(f"     #{i}. {feat:<20} | Impact Magnitude: {imp:.4f}")

# ─────────────────────────────────────────────────────────────
#  STEP 3: Local Explanation (The Micro View)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  🔍 STEP 3: Local Explanation (Applicant Zero)")
print("─" * 60)

# We will audit Applicant index 0
APPLICANT_ID = 0
applicant_data = df_applicants.iloc[APPLICANT_ID]
applicant_risk = y_prob[APPLICANT_ID]
applicant_shap = shap_values[APPLICANT_ID]

# Create a dictionary of {feature: shap_value} for this specific person
local_factors = dict(zip(PROXY_FEATURES, applicant_shap))

# Sort to find the Top 3 pushes UP (Risk drivers) and DOWN (Safety drivers)
sorted_local = sorted(local_factors.items(), key=lambda x: x[1])

top_3_negative = sorted_local[:3]   # Largest negative values (Pushing Risk toward 0)
top_3_positive = sorted_local[-3:]  # Largest positive values (Pushing Risk toward 1)
top_3_positive.reverse()            # Highest risk drivers first

print(f"  👤 Auditing Applicant {APPLICANT_ID} | ML Output Risk Score: {applicant_risk:.1%}")

print(f"\n  📈 TOP 3 RISK DRIVERS (Why might we reject them?):")
for feat, s_val in top_3_positive:
    actual_val = applicant_data[feat]
    print(f"     ➕ {feat:<18} (Value = {actual_val:<8.1f}) adds +{s_val:.3f} to Risk")

print(f"\n  📉 TOP 3 SAFETY DRIVERS (Why might we approve them?):")
for feat, s_val in top_3_negative:
    actual_val = applicant_data[feat]
    print(f"     ➖ {feat:<18} (Value = {actual_val:<8.1f}) drops {s_val:.3f} from Risk")


# ─────────────────────────────────────────────────────────────
#  STEP 4: Serialize to Gold Layer Database
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  💾 STEP 4: Archiving SHAP Audit to Gold Layer")
print("─" * 60)

# Build the record for this specific applicant's audit
explanation_summary = f"Risk {applicant_risk:.1%} primarily driven UP by {top_3_positive[0][0]} and driven DOWN by {top_3_negative[0][0]}."

audit_record = [{
    "applicant_id": int(APPLICANT_ID),
    "model_version": full_model_uri,
    "total_risk_score": float(applicant_risk),
    "top_positive_feature": str(top_3_positive[0][0]),
    "top_positive_shap": float(top_3_positive[0][1]),
    "top_negative_feature": str(top_3_negative[0][0]),
    "top_negative_shap": float(top_3_negative[0][1]),
    "explanation_summary": explanation_summary,
    "audit_timestamp": pd.Timestamp.now().isoformat()
}]

df_gold_shap = pd.DataFrame(audit_record)

spark_shap = spark.createDataFrame(df_gold_shap)
spark_shap.write.format("delta").mode("append").saveAsTable(GOLD_SHAP_EXPLANATIONS)

print(f"  ✅ SHAP Audit precisely anchored into Gold Delta table: {GOLD_SHAP_EXPLANATIONS}")
print("\n🏁 Phase 6 Complete. Model is now 100% Mathematically Transparent.\n")
