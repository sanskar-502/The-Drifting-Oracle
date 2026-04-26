"""
============================================================
  DRIFTING ORACLE — Central Configuration
============================================================
  Smart config that auto-detects Local vs Databricks environment.
  All paths, thresholds, and constants in one place.
============================================================
"""

import os
import sys

# ──────────────────────────────────────────────
#  ENVIRONMENT DETECTION
# ──────────────────────────────────────────────
IS_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ

# ──────────────────────────────────────────────
#  UNITY CATALOG NAMES (Databricks only)
# ──────────────────────────────────────────────
CATALOG_NAME   = "drifting_oracle_db"
SCHEMA_NAME    = "credit_risk"
FULL_SCHEMA    = f"{CATALOG_NAME}.{SCHEMA_NAME}"

# ──────────────────────────────────────────────
#  LOCAL FILE PATHS (VS Code development)
# ──────────────────────────────────────────────
# Resolve base directory (parent of config/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Medallion layers on local filesystem
BRONZE_DIR = os.path.join(OUTPUT_DIR, "bronze")
SILVER_DIR = os.path.join(OUTPUT_DIR, "silver")
GOLD_DIR   = os.path.join(OUTPUT_DIR, "gold")

# Raw data file paths
HOME_CREDIT_CSV = os.path.join(DATA_DIR, "application_train.csv")
GERMAN_CREDIT_CSV = os.path.join(DATA_DIR, "german_credit_data.csv")
POLICY_TEXT_FILE = os.path.join(DATA_DIR, "rbi_sebi_policy.txt")

# ──────────────────────────────────────────────
#  TABLE NAMES — Used in both environments
# ──────────────────────────────────────────────
# Bronze (raw, untouched)
BRONZE_HOME_CREDIT   = "bronze_home_credit"
BRONZE_GERMAN_CREDIT = "bronze_german_credit"
BRONZE_POLICY_TEXT   = "bronze_policy_text"

# Silver (cleaned, aligned features)
SILVER_HOME_CREDIT   = "silver_home_credit"
SILVER_GERMAN_CREDIT = "silver_german_credit"

# Gold (analytics-ready)
GOLD_DRIFT_METRICS       = "gold_drift_metrics"
GOLD_MODEL_COMPARISON    = "gold_model_comparison"
GOLD_EXPLANATION_AUDIT   = "gold_explanation_audit"
GOLD_FAIRNESS_METRICS    = "gold_fairness_metrics"
GOLD_SHAP_EXPLANATIONS   = "gold_shap_explanations"
GOLD_RISK_PREDICTIONS    = "gold_risk_predictions"

# ──────────────────────────────────────────────
#  PATH RESOLUTION HELPER
# ──────────────────────────────────────────────
def get_table_path(layer: str, table_name: str) -> str:
    """
    Returns the correct path/table reference based on environment.
    - Local:      output/bronze/bronze_home_credit  (Delta/Parquet path)
    - Databricks: drifting_oracle_db.credit_risk.bronze_home_credit
    """
    if IS_DATABRICKS:
        return f"{FULL_SCHEMA}.{table_name}"
    else:
        layer_dir = {"bronze": BRONZE_DIR, "silver": SILVER_DIR, "gold": GOLD_DIR}[layer]
        return os.path.join(layer_dir, table_name)


def get_bronze_path(table_name: str) -> str:
    return get_table_path("bronze", table_name)


def get_silver_path(table_name: str) -> str:
    return get_table_path("silver", table_name)


def get_gold_path(table_name: str) -> str:
    return get_table_path("gold", table_name)


# ──────────────────────────────────────────────
#  FEATURE ENGINEERING CONSTANTS
# ──────────────────────────────────────────────

# The 5 proxy features aligned across both datasets
PROXY_FEATURES = [
    "age",
    "credit_amount",
    "duration_months",
    "employment_years",
    "income_proxy",
]

# Demographic columns preserved for fairness audit (25% weight!)
FAIRNESS_COLUMNS = ["gender"]

# All feature columns for modeling
ALL_FEATURE_COLUMNS = PROXY_FEATURES + FAIRNESS_COLUMNS

# Target column
TARGET_COLUMN = "target"

# Home Credit anomaly: DAYS_EMPLOYED = 365243 means unemployed/NA
DAYS_EMPLOYED_ANOMALY = 365243

# ──────────────────────────────────────────────
#  DRIFT DETECTION THRESHOLDS
# ──────────────────────────────────────────────
PSI_THRESHOLD = 0.20       # Trigger retraining if ANY feature exceeds this
PSI_BINS = 10              # Number of decile bins
PSI_EPSILON = 0.0001       # Small constant to avoid log(0)

# ──────────────────────────────────────────────
#  MODEL TRAINING CONFIGURATION
# ──────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "/drifting_oracle/credit_risk"
MODEL_NAME = "credit_risk_model"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Random Forest hyperparameters
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "class_weight": "balanced",
}

# Logistic Regression hyperparameters (second model per eval criteria)
LR_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
    "solver": "lbfgs",
}

# XGBoost hyperparameters (alternative tree model)
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss",
    "use_label_encoder": False,
}

# ──────────────────────────────────────────────
#  RAG / GENAI CONFIGURATION
# ──────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"      # sentence-transformers model
VECTOR_DB_DIR = os.path.join(OUTPUT_DIR, "vector_db")
CHUNK_SIZE = 500          # chars per policy chunk
CHUNK_OVERLAP = 50        # overlap between chunks
TOP_K_RETRIEVAL = 3       # top-k similar chunks for grounding

# ──────────────────────────────────────────────
#  PRINT CONFIG SUMMARY (for debugging)
# ──────────────────────────────────────────────
def print_config():
    env = "🔶 DATABRICKS" if IS_DATABRICKS else "💻 LOCAL (VS Code)"
    print("=" * 60)
    print(f"  🔮 DRIFTING ORACLE — Configuration")
    print(f"  Environment: {env}")
    print("=" * 60)
    if IS_DATABRICKS:
        print(f"  Catalog:     {CATALOG_NAME}")
        print(f"  Schema:      {SCHEMA_NAME}")
    else:
        print(f"  Base Dir:    {BASE_DIR}")
        print(f"  Data Dir:    {DATA_DIR}")
        print(f"  Output Dir:  {OUTPUT_DIR}")
    print(f"  PSI Thresh:  {PSI_THRESHOLD}")
    print(f"  ML Experiment: {MLFLOW_EXPERIMENT_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
