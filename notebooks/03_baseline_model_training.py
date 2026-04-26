"""
============================================================
  DRIFTING ORACLE — NOTEBOOK 03: BASELINE MODEL TRAINING
============================================================
  Phase 3: Multi-Model Comparison & Champion Selection
  
  Trains 3 classifiers on the Silver Home Credit dataset,
  compares them head-to-head, and registers the best
  performer as the @Champion in MLflow.
  
  Models Evaluated:
    1. Logistic Regression  — Regulatory baseline (interpretable)
    2. Random Forest         — Strong tabular (explainable)
    3. XGBoost               — Best-in-class gradient boosting
  
  Selection Criteria: ROC-AUC (primary), then F1-Score
============================================================
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# ── Project path setup ──────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import (
    IS_DATABRICKS,
    SILVER_HOME_CREDIT,
    PROXY_FEATURES,
    TARGET_COLUMN,
    RF_PARAMS,
    LR_PARAMS,
    XGB_PARAMS,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_NAME,
    CATALOG_NAME,
    SCHEMA_NAME,
    get_silver_path,
    print_config,
)
from utils.spark_utils import get_spark_session, read_table

print("\n" + "=" * 70)
print("  🔮 DRIFTING ORACLE — Phase 3: Multi-Model Training & Comparison")
print("=" * 70)

spark = get_spark_session("DriftingOracle_Phase3_Model")
print_config()

# ─────────────────────────────────────────────────────────────
#  STEP 1: Load Silver Data & Prepare Features
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  📥 STEP 1: Loading Silver Layer Data (Home Credit)")
print("─" * 60)

silver_home_path = get_silver_path(SILVER_HOME_CREDIT)
df_home_spark = read_table(spark, silver_home_path)

required_cols = PROXY_FEATURES + [TARGET_COLUMN]

print("  ⏳ Converting Spark DataFrame to Pandas...")
df_home = df_home_spark.select(*required_cols).toPandas()

# Handle any remaining NaN/inf values
df_home = df_home.replace([np.inf, -np.inf], np.nan).dropna()

print(f"  ✅ Data Loaded: {df_home.shape[0]:,} rows × {df_home.shape[1]} columns")
print(f"  🎯 Features: {', '.join(PROXY_FEATURES)}")
print(f"  📊 Target Distribution:")
target_dist = df_home[TARGET_COLUMN].value_counts()
for val, cnt in target_dist.items():
    pct = cnt / len(df_home) * 100
    label = "Repaid (Good)" if val == 0 else "Defaulted (Bad)"
    print(f"     {val} ({label}): {cnt:,} ({pct:.1f}%)")

# ─────────────────────────────────────────────────────────────
#  STEP 2: Train/Test Split
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  ⚙️ STEP 2: Stratified Train/Test Split")
print("─" * 60)

X = df_home[PROXY_FEATURES]
y = df_home[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# NOTE: No StandardScaler — all models train on raw features so that
# downstream notebooks (05, 09) can pass raw data uniformly without
# needing to know which model type is the Champion.

print(f"  ✅ Training Set: {X_train.shape[0]:,} samples")
print(f"  ✅ Testing Set:  {X_test.shape[0]:,} samples")

# ─────────────────────────────────────────────────────────────
#  STEP 3: Configure MLflow
# ─────────────────────────────────────────────────────────────
if IS_DATABRICKS:
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    full_model_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}"
else:
    from pathlib import Path
    local_mlruns = Path(PROJECT_ROOT) / "mlruns"
    local_mlruns.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(local_mlruns.as_uri())
    mlflow.set_experiment("DriftingOracle_Local_Experiment")
    full_model_name = MODEL_NAME


# ─────────────────────────────────────────────────────────────
#  STEP 4: Define & Train All 3 Models
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  🚀 STEP 3: Training 3 Models Head-to-Head")
print("=" * 70)

# Store results for comparison
results = []

# ═══════════════════════════════════════════════════════════
#  MODEL 1: Logistic Regression (Regulatory Baseline)
# ═══════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  📐 MODEL 1: Logistic Regression (Regulatory Baseline)")
print("─" * 60)

with mlflow.start_run(run_name="Model1_LogisticRegression"):
    mlflow.log_params({f"lr_{k}": v for k, v in LR_PARAMS.items()})
    mlflow.set_tag("model_type", "LogisticRegression")
    mlflow.set_tag("purpose", "regulatory_baseline")

    t0 = time.time()
    lr_model = LogisticRegression(**LR_PARAMS)
    lr_model.fit(X_train, y_train)
    lr_train_time = time.time() - t0

    y_pred_lr = lr_model.predict(X_test)
    y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

    lr_acc = accuracy_score(y_test, y_pred_lr)
    lr_f1 = f1_score(y_test, y_pred_lr)
    lr_roc = roc_auc_score(y_test, y_prob_lr)
    lr_prec = precision_score(y_test, y_pred_lr)
    lr_rec = recall_score(y_test, y_pred_lr)

    metrics_lr = {
        "accuracy": lr_acc, "f1_score": lr_f1, "roc_auc": lr_roc,
        "precision": lr_prec, "recall": lr_rec, "train_time_sec": lr_train_time
    }
    mlflow.log_metrics(metrics_lr)

    sig = mlflow.models.signature.infer_signature(X_train, y_pred_lr)
    mlflow.sklearn.log_model(lr_model, "logistic_regression_model", signature=sig)

    print(f"  ✅ Trained in {lr_train_time:.1f}s")
    print(f"     Accuracy: {lr_acc:.4f} | F1: {lr_f1:.4f} | ROC-AUC: {lr_roc:.4f}")
    print(f"     Precision: {lr_prec:.4f} | Recall: {lr_rec:.4f}")

    results.append({
        "model": "Logistic Regression", "accuracy": lr_acc,
        "f1_score": lr_f1, "roc_auc": lr_roc, "precision": lr_prec,
        "recall": lr_rec, "train_time": lr_train_time, "sklearn_model": lr_model,
        "log_func": mlflow.sklearn.log_model, "artifact_path": "logistic_regression_model",
        "needs_scaling": False
    })

# ═══════════════════════════════════════════════════════════
#  MODEL 2: Random Forest (Explainable Tree Ensemble)
# ═══════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  🌲 MODEL 2: Random Forest (Explainable Ensemble)")
print("─" * 60)

with mlflow.start_run(run_name="Model2_RandomForest"):
    mlflow.log_params({f"rf_{k}": v for k, v in RF_PARAMS.items()})
    mlflow.set_tag("model_type", "RandomForest")
    mlflow.set_tag("purpose", "explainable_ensemble")

    t0 = time.time()
    rf_model = RandomForestClassifier(**RF_PARAMS)
    rf_model.fit(X_train, y_train)
    rf_train_time = time.time() - t0

    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

    rf_acc = accuracy_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf)
    rf_roc = roc_auc_score(y_test, y_prob_rf)
    rf_prec = precision_score(y_test, y_pred_rf)
    rf_rec = recall_score(y_test, y_pred_rf)

    metrics_rf = {
        "accuracy": rf_acc, "f1_score": rf_f1, "roc_auc": rf_roc,
        "precision": rf_prec, "recall": rf_rec, "train_time_sec": rf_train_time
    }
    mlflow.log_metrics(metrics_rf)

    # Log feature importance
    feat_importance = dict(zip(PROXY_FEATURES, rf_model.feature_importances_))
    print(f"\n  📊 Feature Importance (Random Forest):")
    for feat, imp in sorted(feat_importance.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"     {feat:<25} {imp:.4f}  {bar}")
    mlflow.log_params({f"importance_{k}": round(v, 4) for k, v in feat_importance.items()})

    sig = mlflow.models.signature.infer_signature(X_train, y_pred_rf)
    mlflow.sklearn.log_model(rf_model, "random_forest_model", signature=sig)

    print(f"\n  ✅ Trained in {rf_train_time:.1f}s")
    print(f"     Accuracy: {rf_acc:.4f} | F1: {rf_f1:.4f} | ROC-AUC: {rf_roc:.4f}")
    print(f"     Precision: {rf_prec:.4f} | Recall: {rf_rec:.4f}")

    results.append({
        "model": "Random Forest", "accuracy": rf_acc,
        "f1_score": rf_f1, "roc_auc": rf_roc, "precision": rf_prec,
        "recall": rf_rec, "train_time": rf_train_time, "sklearn_model": rf_model,
        "log_func": mlflow.sklearn.log_model, "artifact_path": "random_forest_model",
        "needs_scaling": False
    })

# ═══════════════════════════════════════════════════════════
#  MODEL 3: XGBoost (Best-in-Class Gradient Boosting)
# ═══════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  ⚡ MODEL 3: XGBoost (Gradient Boosting)")
print("─" * 60)

# Calculate scale_pos_weight for imbalanced dataset
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count

xgb_params_runtime = XGB_PARAMS.copy()
xgb_params_runtime["scale_pos_weight"] = round(scale_pos_weight, 2)

with mlflow.start_run(run_name="Model3_XGBoost"):
    mlflow.log_params({f"xgb_{k}": v for k, v in xgb_params_runtime.items()})
    mlflow.set_tag("model_type", "XGBoost")
    mlflow.set_tag("purpose", "best_in_class_boosting")

    t0 = time.time()
    xgb_model = XGBClassifier(**xgb_params_runtime)
    xgb_model.fit(X_train, y_train)
    xgb_train_time = time.time() - t0

    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

    xgb_acc = accuracy_score(y_test, y_pred_xgb)
    xgb_f1 = f1_score(y_test, y_pred_xgb)
    xgb_roc = roc_auc_score(y_test, y_prob_xgb)
    xgb_prec = precision_score(y_test, y_pred_xgb)
    xgb_rec = recall_score(y_test, y_pred_xgb)

    metrics_xgb = {
        "accuracy": xgb_acc, "f1_score": xgb_f1, "roc_auc": xgb_roc,
        "precision": xgb_prec, "recall": xgb_rec, "train_time_sec": xgb_train_time
    }
    mlflow.log_metrics(metrics_xgb)

    # Log feature importance
    feat_importance_xgb = dict(zip(PROXY_FEATURES, xgb_model.feature_importances_))
    print(f"\n  📊 Feature Importance (XGBoost):")
    for feat, imp in sorted(feat_importance_xgb.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"     {feat:<25} {imp:.4f}  {bar}")

    sig = mlflow.models.signature.infer_signature(X_train, y_pred_xgb)
    mlflow.xgboost.log_model(xgb_model, "xgboost_model", signature=sig)

    print(f"\n  ✅ Trained in {xgb_train_time:.1f}s")
    print(f"     Accuracy: {xgb_acc:.4f} | F1: {xgb_f1:.4f} | ROC-AUC: {xgb_roc:.4f}")
    print(f"     Precision: {xgb_prec:.4f} | Recall: {xgb_rec:.4f}")

    results.append({
        "model": "XGBoost", "accuracy": xgb_acc,
        "f1_score": xgb_f1, "roc_auc": xgb_roc, "precision": xgb_prec,
        "recall": xgb_rec, "train_time": xgb_train_time, "sklearn_model": xgb_model,
        "log_func": mlflow.xgboost.log_model, "artifact_path": "xgboost_model",
        "needs_scaling": False
    })


# ═══════════════════════════════════════════════════════════
#  STEP 4: Head-to-Head Comparison & Champion Selection
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  🏆 STEP 4: Model Comparison & Champion Selection")
print("=" * 70)

# Build comparison table
df_comparison = pd.DataFrame([{
    "Model": r["model"],
    "Accuracy": r["accuracy"],
    "F1-Score": r["f1_score"],
    "ROC-AUC": r["roc_auc"],
    "Precision": r["precision"],
    "Recall": r["recall"],
    "Train Time (s)": round(r["train_time"], 1)
} for r in results])

print("\n" + df_comparison.to_string(index=False))

# Select Champion based on ROC-AUC (primary), then F1 (tiebreaker)
best_idx = max(range(len(results)), key=lambda i: (results[i]["roc_auc"], results[i]["f1_score"]))
champion = results[best_idx]

print(f"\n  🥇 CHAMPION: {champion['model']}")
print(f"     ROC-AUC: {champion['roc_auc']:.4f} (primary criterion)")
print(f"     F1:      {champion['f1_score']:.4f} (secondary criterion)")

# ─────────────────────────────────────────────────────────────
#  STEP 5: Register Champion Model in MLflow
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print(f"  💾 STEP 5: Registering {champion['model']} as @Champion")
print("─" * 60)

with mlflow.start_run(run_name=f"Champion_{champion['model'].replace(' ', '_')}"):
    # Log all champion metrics
    mlflow.log_params({"champion_model_type": champion["model"]})
    mlflow.log_metrics({
        "champion_accuracy": champion["accuracy"],
        "champion_f1_score": champion["f1_score"],
        "champion_roc_auc": champion["roc_auc"],
        "champion_precision": champion["precision"],
        "champion_recall": champion["recall"],
    })
    mlflow.set_tag("deployment_status", "Champion")
    mlflow.set_tag("selection_reason", "Highest ROC-AUC across 3-model comparison")

    sig = mlflow.models.signature.infer_signature(
        X_train,
        champion["sklearn_model"].predict(X_train)
    )

    champion["log_func"](
        champion["sklearn_model"],
        champion["artifact_path"],
        registered_model_name=full_model_name,
        signature=sig
    )

    # Set alias on Databricks (UC does NOT support stages)
    if IS_DATABRICKS:
        from mlflow import MlflowClient
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{full_model_name}'")
        latest = max([int(v.version) for v in versions])
        client.set_registered_model_alias(full_model_name, "Champion", str(latest))
        print(f"  ✅ @Champion alias set on version {latest}")

print(f"  ✅ Registered: {full_model_name} (@Champion)")

# ─────────────────────────────────────────────────────────────
#  STEP 6: Analysis & Reasoning Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  📋 PHASE 3 COMPLETE — Model Selection Report")
print("=" * 70)

print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  🏆 CHAMPION: {champion['model']:<44} │
  ├─────────────────────────────────────────────────────────────┤
  │                                                             │
  │  📊 Final Metrics:                                          │
  │     Accuracy:   {champion['accuracy']:.4f}                                    │
  │     F1-Score:   {champion['f1_score']:.4f}                                    │
  │     ROC-AUC:    {champion['roc_auc']:.4f}                                    │
  │     Precision:  {champion['precision']:.4f}                                    │
  │     Recall:     {champion['recall']:.4f}                                    │
  │                                                             │
  │  🧠 Selection Reasoning:                                    │
  │     • ROC-AUC chosen as primary metric because credit       │
  │       risk models must rank-order borrowers correctly,      │
  │       not just classify at a fixed threshold.               │
  │     • F1-Score as tiebreaker: balances precision/recall     │
  │       for the heavily imbalanced dataset (~8% default).     │
  │     • Accuracy alone is misleading — a naive "always good"  │
  │       model gets ~92% accuracy but catches 0 defaults.      │
  │                                                             │
  │  📦 MLflow:                                                 │
  │     Model Name:  {full_model_name:<40} │
  │     Status:      @Champion (production-ready)               │
  └─────────────────────────────────────────────────────────────┘
""")

print("  📊 Full Comparison Table:")
print("  " + "─" * 55)
for _, row in df_comparison.iterrows():
    marker = " 🥇" if row["Model"] == champion["model"] else "   "
    print(f"  {marker} {row['Model']:<22} AUC={row['ROC-AUC']:.4f}  F1={row['F1-Score']:.4f}  Acc={row['Accuracy']:.4f}  [{row['Train Time (s)']}s]")
print("  " + "─" * 55)

print("\n🏁 Phase 3 Complete. Champion model is registered and ready for drift detection.\n")
