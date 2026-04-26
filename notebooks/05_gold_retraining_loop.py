"""
============================================================
  DRIFTING ORACLE — NOTEBOOK 05: RETRAINING LOOP
============================================================
  Phase 5: Champion vs. Challenger Retraining
  
  Simulates the arrival of delayed ground-truth labels for 
  the drifted German Credit batch. It trains a Challenger 
  model on the new data, tests it competitively against the 
  incumbent Champion, and handles MLflow promotion.
============================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ── Project path setup ──────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import (
    IS_DATABRICKS,
    SILVER_GERMAN_CREDIT,
    PROXY_FEATURES,
    TARGET_COLUMN,
    XGB_PARAMS,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_NAME,
    CATALOG_NAME,
    SCHEMA_NAME,
    GOLD_MODEL_COMPARISON,
    get_silver_path,
    get_gold_path,
    print_config,
)
from utils.spark_utils import get_spark_session, read_table, save_table

print("\n" + "=" * 70)
print("  🔮 DRIFTING ORACLE — Phase 5: Champion-Challenger Auto-Retraining")
print("=" * 70)

spark = get_spark_session("DriftingOracle_Phase5_Retrain")
print_config()

# ─────────────────────────────────────────────────────────────
#  STEP 1: Load Drifted Batch & Simulate Delayed Labels
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  📥 STEP 1: Loading Drifted Batch & Resolving Labels")
print("─" * 60)

german_path = get_silver_path(SILVER_GERMAN_CREDIT)
df_german_spark = read_table(spark, german_path)
required_cols = PROXY_FEATURES + [TARGET_COLUMN]

df_german = df_german_spark.select(*required_cols).toPandas().dropna()

# ⚠️ HACKATHON RATIONALE: Credit Risk has "delayed ground truth"
# We ingested German data without `Risk` labels to calculate Drift natively.
# Now, to retrain, we simulate 90-days passing and underwriting returning the actual outcomes.
# We inject synthetic labels mathematically biased so the new data carries a different signal.
print("  ⏳ Simulating ground-truth maturity (90-day delay)...")
np.random.seed(42)
# Create a synthetic boundary: low income & young = high risk natively
synthetic_risk_prob = np.clip(
    (df_german["credit_amount"] / (df_german["income_proxy"] + 1)) * 0.1 
    - (df_german["age"] * 0.005) + 0.3, 
    0, 1
)
# Binomial flip probabilities into 1 (Bad) and 0 (Good)
df_german[TARGET_COLUMN] = np.random.binomial(n=1, p=synthetic_risk_prob)

print(f"  ✅ Matured Batch Acquired: {len(df_german):,} rows")
print("  📊 New Ground Truth Distribution (Target Drifted):")
print(df_german[TARGET_COLUMN].value_counts(normalize=True).round(3) * 100)

# ─────────────────────────────────────────────────────────────
#  STEP 2: Prepare Evaluation Protocol
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  ⚙️ STEP 2: Splitting Retraining Data")
print("─" * 60)

X = df_german[PROXY_FEATURES]
y = df_german[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=101, stratify=y
)

# ─────────────────────────────────────────────────────────────
#  STEP 3: Load Champion Model
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  🏆 STEP 3: Evaluating Incumbent @Champion")
print("─" * 60)

# Local vs Databricks MLflow Setup
if IS_DATABRICKS:
    mlflow.set_registry_uri("databricks-uc")
    full_model_uri = f"models:/{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}@Champion"
else:
    from pathlib import Path
    local_mlruns = Path(PROJECT_ROOT) / "mlruns"
    mlflow.set_tracking_uri(local_mlruns.as_uri())
    
    # In local MLflow, fetching by alias isn't fully supported without UC. 
    # Fetch the latest registered version.
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    try:
        latest_version_info = client.get_latest_versions(MODEL_NAME)[0]
        full_model_uri = f"models:/{MODEL_NAME}/{latest_version_info.version}"
    except Exception:
        full_model_uri = f"models:/{MODEL_NAME}/1" # Fallback to version 1

print(f"  ⏳ Downloading @Champion from Registry: {full_model_uri}")
try:
    # Use generic pyfunc loader since it could be XGBoost or RandomForest or LR
    champion_model = mlflow.pyfunc.load_model(full_model_uri)
    y_prob_champ = champion_model.predict(X_test)
    y_pred_champ = (y_prob_champ > 0.5).astype(int)
    
    champ_auc = roc_auc_score(y_test, y_prob_champ)
    champ_f1 = f1_score(y_test, y_pred_champ)
    champ_acc = accuracy_score(y_test, y_pred_champ)
    
    print(f"  ✅ Champion Evaluation on Drifted Data:")
    print(f"     Accuracy: {champ_acc:.4f} | F1: {champ_f1:.4f} | ROC-AUC: {champ_auc:.4f}")
except Exception as e:
    print(f"  ⚠️ Error loading Champion (Did Phase 3 run?): {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
#  STEP 4: Train Challenger Model
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  ⚔️ STEP 4: Training The @Challenger Engine")
print("─" * 60)

if IS_DATABRICKS:
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME + "_Retraining")
else:
    mlflow.set_experiment("DriftingOracle_Local_Retraining")

with mlflow.start_run(run_name="Challenger_AutoRetrain_XGBoost"):
    
    # Scale positive weight dynamically for new dataset
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    challenger_params = XGB_PARAMS.copy()
    challenger_params["scale_pos_weight"] = round(spw, 2)
    
    mlflow.log_params(challenger_params)
    
    print("  ⏳ Training Challenger XGBoost on New Distro...")
    challenger_model = XGBClassifier(**challenger_params)
    challenger_model.fit(X_train, y_train)
    
    y_prob_challenger = challenger_model.predict_proba(X_test)[:, 1]
    y_pred_challenger = (y_prob_challenger > 0.5).astype(int)
    
    chall_auc = roc_auc_score(y_test, y_prob_challenger)
    chall_f1 = f1_score(y_test, y_pred_challenger)
    chall_acc = accuracy_score(y_test, y_pred_challenger)
    
    mlflow.log_metrics({"challenger_auc": chall_auc, "challenger_f1": chall_f1})
    
    sig = mlflow.models.signature.infer_signature(X_train, y_pred_challenger)
    mlflow.xgboost.log_model(challenger_model, "challenger_xgboost", signature=sig)
    
    # Preserve Run ID
    challenger_run_id = mlflow.active_run().info.run_id
    
    print(f"  ✅ Challenger Evaluation:")
    print(f"     Accuracy: {chall_acc:.4f} | F1: {chall_f1:.4f} | ROC-AUC: {chall_auc:.4f}")

# ─────────────────────────────────────────────────────────────
#  STEP 5: Compare & Promote (The MLOps Gateway)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  ⚖️ STEP 5: PROMOTION GATEWAY")
print("=" * 70)

print(f"  🏆 Champion ROC-AUC:   {champ_auc:.4f}")
print(f"  ⚔️ Challenger ROC-AUC: {chall_auc:.4f}")

promotion_decision = "REJECTED"

if chall_auc > champ_auc:
    promotion_decision = "PROMOTED to @Champion"
    print("\n  ✅ RESULT: CHALLENGER DEFEATED CHAMPION.")
    print("  The new distribution degraded incumbent performance. Challenger adapts better.")
    
    # Promote Challenger to Champion
    if IS_DATABRICKS:
        from mlflow import MlflowClient
        client = MlflowClient()
        new_version_req = mlflow.register_model(
            f"runs:/{challenger_run_id}/challenger_xgboost", 
            f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}"
        )
        client.set_registered_model_alias(f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}", "Champion", new_version_req.version)
        print(f"  ✅ UC Registry Updated: Challenger promoted to Version {new_version_req.version} @Champion")
    else:
        new_version_req = mlflow.register_model(
             f"runs:/{challenger_run_id}/challenger_xgboost", 
             MODEL_NAME
        )
        # Using built in MLFlow file local syntax usually requires explicit client passing, simulating success
        print(f"  ✅ Local MLflow Registry Updated: Model Version {new_version_req.version} mapped to Challenger.")

else:
    print("\n  ❌ RESULT: CHALLENGER REJECTED.")
    print("  Incumbent Champion remains more mathematically robust. Overfitting prevented.")

# ─────────────────────────────────────────────────────────────
#  STEP 6: Gold Layer Logging
# ─────────────────────────────────────────────────────────────
comparison_record = [{
    "timestamp": pd.Timestamp.now().isoformat(),
    "champion_auc": float(champ_auc),
    "champion_f1": float(champ_f1),
    "challenger_auc": float(chall_auc),
    "challenger_f1": float(chall_f1),
    "better_model": "Challenger" if chall_auc > champ_auc else "Champion",
    "promotion_action": promotion_decision
}]

df_gold_comparison = pd.DataFrame(comparison_record)

import json
import tempfile
with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".json") as f:
    for record in df_gold_comparison.to_dict(orient="records"):
        f.write(json.dumps(record) + "\n")
    temp_json_path = f.name

gold_comparison_path = get_gold_path(GOLD_MODEL_COMPARISON)
spark_comparison = spark.read.json(temp_json_path)
save_table(spark_comparison, gold_comparison_path)
os.remove(temp_json_path)

print(f"\n  💾 Logged Retraining Action to Gold Table: {GOLD_MODEL_COMPARISON}")
print("\n🏁 Phase 5 Complete. System autonomously evolved.\n")
