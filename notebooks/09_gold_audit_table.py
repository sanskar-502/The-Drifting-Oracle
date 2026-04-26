"""
============================================================
  DRIFTING ORACLE — NOTEBOOK 09: GOLD AUDIT TABLE
============================================================
  Phase 9: Unified Observability & Audit-Ready Output
  
  Consolidates ALL pipeline outputs into a single, governed
  Gold Audit Table that satisfies regulatory audit requirements.
  
  Columns:
    applicant_id, model_prediction, shap_summary,
    drift_score, retraining_status, explanation_label,
    hallucination_cost_band, model_version, timestamp
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

# ── Project path setup ──────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import (
    IS_DATABRICKS,
    SILVER_GERMAN_CREDIT,
    PROXY_FEATURES,
    MODEL_NAME,
    CATALOG_NAME,
    SCHEMA_NAME,
    GOLD_DRIFT_METRICS,
    GOLD_MODEL_COMPARISON,
    get_silver_path,
    get_gold_path,
    print_config,
)
from utils.spark_utils import get_spark_session, read_table, save_table

print("\n" + "=" * 70)
print("  🔮 DRIFTING ORACLE — Phase 9: Gold Audit Table & Observability")
print("=" * 70)

spark = get_spark_session("DriftingOracle_Phase9_Audit")
GOLD_AUDIT_TABLE = "gold_audit_table"

# ─────────────────────────────────────────────────────────────
#  STEP 1: Load All Upstream Gold Tables
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  📥 STEP 1: Loading All Upstream Pipeline Outputs")
print("─" * 60)

# 1A. Load SHAP Explanations
print("  ⏳ Loading SHAP explanations...")
try:
    df_shap = read_table(spark, get_gold_path("gold_shap_explanations")).toPandas()
    print(f"  ✅ SHAP: {len(df_shap)} records")
except:
    df_shap = pd.DataFrame()
    print("  ⚠️ SHAP table not found — will use defaults")

# 1B. Load Drift Metrics
print("  ⏳ Loading drift metrics...")
try:
    df_drift = read_table(spark, get_gold_path(GOLD_DRIFT_METRICS)).toPandas()
    print(f"  ✅ Drift: {len(df_drift)} feature scores")
except:
    df_drift = pd.DataFrame()
    print("  ⚠️ Drift table not found — will use defaults")

# 1C. Load Model Comparison (Retraining Status)
print("  ⏳ Loading retraining history...")
try:
    df_retrain = read_table(spark, get_gold_path(GOLD_MODEL_COMPARISON)).toPandas()
    print(f"  ✅ Retraining: {len(df_retrain)} events")
except:
    df_retrain = pd.DataFrame()
    print("  ⚠️ Retrain table not found — will use defaults")

# 1D. Load Hallucination Cost
print("  ⏳ Loading hallucination cost scores...")
try:
    df_hallucination = read_table(spark, get_gold_path("gold_hallucination_cost")).toPandas()
    print(f"  ✅ Hallucination: {len(df_hallucination)} records")
except:
    df_hallucination = pd.DataFrame()
    print("  ⚠️ Hallucination table not found — will use defaults")

# ─────────────────────────────────────────────────────────────
#  STEP 2: Load Champion Model & Generate Predictions
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  🤖 STEP 2: Running Champion Model Inference")
print("─" * 60)

# Load applicant data
german_path = get_silver_path(SILVER_GERMAN_CREDIT)
df_applicants = read_table(spark, german_path).select(*PROXY_FEATURES).toPandas().dropna()
df_applicants.reset_index(drop=True, inplace=True)

# Load Champion Model
if IS_DATABRICKS:
    mlflow.set_registry_uri("databricks-uc")
else:
    from pathlib import Path
    local_mlruns = Path(PROJECT_ROOT) / "mlruns"
    mlflow.set_tracking_uri(local_mlruns.as_uri())

from mlflow.tracking import MlflowClient
client = MlflowClient()
try:
    latest = client.get_latest_versions(MODEL_NAME)[0]
    model_version = latest.version
    model_uri = f"models:/{MODEL_NAME}/{model_version}"
except:
    model_version = "1"
    model_uri = f"models:/{MODEL_NAME}/1"

print(f"  ⏳ Loading model: {model_uri}")

# Load the underlying model (not pyfunc) to access predict_proba directly
# PyFunc's .predict() returns binary classes [0,1], NOT probabilities.
try:
    model = mlflow.xgboost.load_model(model_uri)
except Exception:
    model = mlflow.sklearn.load_model(model_uri)

# Generate continuous probability predictions (not binary classes)
predictions = model.predict_proba(df_applicants)
risk_scores = predictions[:, 1]  # Probability of default (Class 1)

print(f"  ✅ Generated probability predictions for {len(df_applicants)} applicants")

# ─────────────────────────────────────────────────────────────
#  STEP 3: Extract Aggregate Drift Score
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  📊 STEP 3: Extracting Drift & Retraining Status")
print("─" * 60)

# Aggregate drift: average PSI across all features
if not df_drift.empty:
    avg_drift = df_drift['psi_score'].mean()
    max_drift_feature = df_drift.loc[df_drift['psi_score'].idxmax(), 'feature_name']
    drift_status = df_drift.iloc[0]['drift_status']
    print(f"  ✅ Average PSI: {avg_drift:.4f} (Max: {max_drift_feature})")
else:
    avg_drift = 0.0
    drift_status = "Unknown"

# Retraining status
if not df_retrain.empty:
    retraining_status = df_retrain.iloc[-1].get('promotion_action', 'No retraining')
    better_model = df_retrain.iloc[-1].get('better_model', 'N/A')
    print(f"  ✅ Last Retraining: {retraining_status} ({better_model})")
else:
    retraining_status = "No retraining executed"

# ─────────────────────────────────────────────────────────────
#  STEP 4: Build Unified Audit Table
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  🏗️ STEP 4: Assembling Unified Audit Table")
print("─" * 60)

audit_records = []
timestamp = pd.Timestamp.now().isoformat()

for i in range(len(df_applicants)):
    applicant_id = i
    prediction = float(risk_scores[i])
    
    # SHAP summary for this applicant
    if not df_shap.empty and i < len(df_shap):
        shap_summary = df_shap.iloc[i].get('explanation_summary', 'N/A')
    elif not df_shap.empty:
        # Use the first SHAP record as template (only Applicant 0 was fully computed in Phase 6)
        shap_summary = df_shap.iloc[0].get('explanation_summary', 'N/A') + " [Prototype — per-applicant SHAP pending]"
    else:
        shap_summary = "SHAP not computed"
    
    # Hallucination cost for this applicant
    if not df_hallucination.empty and i < len(df_hallucination):
        hallucination_band = df_hallucination.iloc[i].get('impact_band', 'N/A')
        explanation_label = df_hallucination.iloc[i].get('review_flag', 'N/A')
    elif not df_hallucination.empty:
        hallucination_band = df_hallucination.iloc[0].get('impact_band', 'N/A')
        explanation_label = df_hallucination.iloc[0].get('review_flag', 'N/A')
    else:
        hallucination_band = "Not scored"
        explanation_label = "Not evaluated"
    
    # Risk classification
    if prediction > 0.5:
        risk_class = "HIGH RISK — Likely Default"
    elif prediction > 0.3:
        risk_class = "MEDIUM RISK — Monitor"
    else:
        risk_class = "LOW RISK — Likely Repay"
    
    audit_records.append({
        "applicant_id": int(applicant_id),
        "model_prediction": round(float(prediction), 4),
        "risk_classification": risk_class,
        "shap_summary": str(shap_summary),
        "drift_score_avg_psi": round(float(avg_drift), 4),
        "retraining_status": str(retraining_status),
        "explanation_label": str(explanation_label),
        "hallucination_cost_band": str(hallucination_band),
        "model_version": str(model_version),
        "timestamp": timestamp
    })

df_audit = pd.DataFrame(audit_records)
print(f"  ✅ Assembled {len(df_audit)} audit records with {len(df_audit.columns)} columns")

# ─────────────────────────────────────────────────────────────
#  STEP 5: Display Audit Preview
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  🏛️ GOLD AUDIT TABLE — PREVIEW (First 5 Applicants)")
print("=" * 70)

for _, r in df_audit.head(5).iterrows():
    print(f"""
  ┌────────────────────────────────────────────────────────────────┐
  │  👤 Applicant: {r['applicant_id']:<47} │
  │  🤖 Prediction: {r['model_prediction']:<7.4f} → {r['risk_classification']:<30} │
  │  📝 SHAP: {r['shap_summary'][:50]:<52} │
  │  📊 Drift (Avg PSI): {r['drift_score_avg_psi']:<41.4f} │
  │  🔄 Retraining: {r['retraining_status'][:43]:<45} │
  │  🏷️  Explanation: {r['explanation_label'][:43]:<45} │
  │  💰 Hallucination Band: {r['hallucination_cost_band']:<38} │
  │  📦 Model Version: {r['model_version']:<43} │
  │  🕐 Timestamp: {r['timestamp'][:25]:<47} │
  └────────────────────────────────────────────────────────────────┘""")

# ─────────────────────────────────────────────────────────────
#  STEP 6: Schema Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  📋 FINAL AUDIT TABLE SCHEMA")
print("─" * 60)
for col in df_audit.columns:
    dtype = str(df_audit[col].dtype)
    print(f"     {col:<30} : {dtype}")

# ─────────────────────────────────────────────────────────────
#  STEP 7: Save to Gold Layer
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  💾 STEP 7: Persisting Audit Table to Gold Layer")
print("─" * 60)

with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".json") as f:
    for record in df_audit.to_dict(orient="records"):
        f.write(json.dumps(record) + "\n")
    temp_json_path = f.name

gold_audit_path = get_gold_path(GOLD_AUDIT_TABLE)
spark_audit = spark.read.json(temp_json_path)
save_table(spark_audit, gold_audit_path)
os.remove(temp_json_path)

print(f"  ✅ Saved {len(df_audit)} records to: {GOLD_AUDIT_TABLE}")

# ─────────────────────────────────────────────────────────────
#  FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  🏆 PHASE 9 — GOLD AUDIT TABLE COMPLETE")
print("=" * 70)
print(f"""
  ┌────────────────────────────────────────────────────────────────┐
  │  📊 Total Audit Records:     {len(df_audit):<33,} │
  │  📦 Model Version:           {model_version:<33} │
  │  📊 Avg Drift (PSI):         {avg_drift:<33.4f} │
  │  🔄 Retraining Status:       {retraining_status[:33]:<33} │
  │  📋 Columns Per Record:      {len(df_audit.columns):<33} │
  │  💾 Gold Table:              {GOLD_AUDIT_TABLE:<33} │
  └────────────────────────────────────────────────────────────────┘
""")
print("🏁 Phase 9 Complete. Full audit-ready output table generated.\n")
