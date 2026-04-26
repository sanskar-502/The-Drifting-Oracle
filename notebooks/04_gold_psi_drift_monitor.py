"""
============================================================
  DRIFTING ORACLE — NOTEBOOK 04: PSI DRIFT MONITORING
============================================================
  Phase 4: Data Drift Detection (Gold Layer)
  
  Calculates the Population Stability Index (PSI) comparing
  the Baseline Training Distribution (Home Credit) against
  the Incoming Production Batch (German Credit) across the
  5 aligned proxy features.
  
  Triggers:
    • PSI < 0.10: Stable
    • 0.10 ≤ PSI < 0.20: Monitor
    • PSI ≥ 0.20: Retraining Trigger
    
  Output: gold_drift_metrics (Delta Table)
============================================================
"""

import os
import sys
import numpy as np
import pandas as pd

# ── Project path setup ──────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import (
    SILVER_HOME_CREDIT,
    SILVER_GERMAN_CREDIT,
    GOLD_DRIFT_METRICS,
    PROXY_FEATURES,
    PSI_BINS,
    PSI_EPSILON,
    PSI_THRESHOLD,
    get_silver_path,
    get_gold_path,
    print_config,
)
from utils.spark_utils import get_spark_session, read_table, save_table

print("\n" + "=" * 70)
print("  🔮 DRIFTING ORACLE — Phase 4: Population Stability Index (PSI)")
print("=" * 70)

spark = get_spark_session("DriftingOracle_Phase4_PSI")

# ─────────────────────────────────────────────────────────────
#  STEP 1: Load Both Silver Distributions
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  📥 STEP 1: Loading Silver Layer Datasets")
print("─" * 60)

home_path = get_silver_path(SILVER_HOME_CREDIT)
german_path = get_silver_path(SILVER_GERMAN_CREDIT)

df_base_spark = read_table(spark, home_path)
df_incoming_spark = read_table(spark, german_path)

print("  ⏳ Converting Spark DataFrames to Pandas for Mathematical Binning...")
# Only grab the features we are auditing for drift
df_base = df_base_spark.select(*PROXY_FEATURES).toPandas().dropna()
df_incoming = df_incoming_spark.select(*PROXY_FEATURES).toPandas().dropna()

print(f"  ✅ Base Distribution (Home Credit):     {len(df_base):,} reliable records")
print(f"  ✅ Incoming Distribution (German Batch): {len(df_incoming):,} reliable records")

# ─────────────────────────────────────────────────────────────
#  STEP 2: Define PSI Calculation Logic
# ─────────────────────────────────────────────────────────────
def calculate_psi(expected_series, actual_series, bins=10):
    """
    Calculates Population Stability Index using quantile binning from the expected distribution.
    """
    # 1. Create decile bins based purely on the base (expected) distribution
    binned_base, bin_edges = pd.qcut(expected_series, q=bins, retbins=True, duplicates='drop')
    
    # Expand edges slightly to catch extremes in actual data
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    
    # 2. Extract bin counts and convert to percentages
    expected_percents = pd.cut(expected_series, bins=bin_edges).value_counts(normalize=True).sort_index().values
    actual_percents = pd.cut(actual_series, bins=bin_edges).value_counts(normalize=True).sort_index().values
    
    # 3. Add Epsilon to avoid divide-by-zero during log calculation
    expected_percents = np.where(expected_percents == 0, PSI_EPSILON, expected_percents)
    actual_percents = np.where(actual_percents == 0, PSI_EPSILON, actual_percents)
    
    # 4. PSI Formula: sum( (Actual% - Expected%) * ln(Actual% / Expected%) )
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    psi_total = np.sum(psi_values)
    
    return round(psi_total, 4)

def determine_drift_status(psi_score):
    if psi_score < 0.10:
        return "Stable"
    elif psi_score < 0.20:
        return "Monitor"
    else:
        return "Retraining Trigger"

# ─────────────────────────────────────────────────────────────
#  STEP 3: Execute Distribution Audits
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  ⚖️ STEP 3: Executing Feature Binning & PSI Formula")
print("─" * 60)

drift_results = []
global_retraining_trigger = False

for feature in PROXY_FEATURES:
    psi_score = calculate_psi(df_base[feature], df_incoming[feature], bins=PSI_BINS)
    status = determine_drift_status(psi_score)
    
    if status == "Retraining Trigger":
        global_retraining_trigger = True
        
    drift_results.append({
        "feature_name": feature,
        "psi_score": float(psi_score),
        "drift_status": status,
        "audit_timestamp": pd.Timestamp.now().isoformat()
    })
    
# Convert to a DataFrame for elegant reporting and saving
df_drift_report = pd.DataFrame(drift_results)

# ─────────────────────────────────────────────────────────────
#  STEP 4: Display Output & Trigger Flags
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  🚨 POPULATION STABILITY INDEX (PSI) DRIFT REPORT")
print("=" * 70)

for _, row in df_drift_report.iterrows():
    if row['drift_status'] == "Stable":
        icon = "🟢"
    elif row['drift_status'] == "Monitor":
        icon = "🟡"
    else:
        icon = "🔴"
        
    print(f"  {icon} Feature: {row['feature_name']:<18} | PSI: {row['psi_score']:<6.4f} | Status: [{row['drift_status']}]")

print("\n  " + "─" * 60)
if global_retraining_trigger:
    print("  🚨 DATA DRIFT DETECTED: Master Retraining Trigger = [TRUE]")
    print(f"     One or more proxy features breached the {PSI_THRESHOLD} PSI threshold.")
    print("     This mandates an immediate Model Retraining Action!")
else:
    print("  ✅ DATA DISTRIBUTION IS IN TACT: Master Retraining Trigger = [FALSE]")
    print(f"     All proxy features remain beneath the {PSI_THRESHOLD} critical PSI limit.")

# ─────────────────────────────────────────────────────────────
#  STEP 5: Save to Gold Layer
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  💾 STEP 5: Saving Drift Detection Table to Gold Layer")
print("─" * 60)

import json
import tempfile

with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".json") as f:
    # Convert DF to list of dicts and dump
    for record in df_drift_report.to_dict(orient="records"):
        f.write(json.dumps(record) + "\n")
    temp_json_path = f.name

df_drift_spark = spark.read.json(temp_json_path)
gold_drift_path = get_gold_path(GOLD_DRIFT_METRICS)
save_table(df_drift_spark, gold_drift_path)

os.remove(temp_json_path)
print(f"  ✅ Written to Gold Delta table: {GOLD_DRIFT_METRICS}")
print("\n🏁 Phase 4 Complete. Drift metrics recorded.\n")
