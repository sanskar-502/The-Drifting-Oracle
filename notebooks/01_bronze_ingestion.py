"""
============================================================
  DRIFTING ORACLE — NOTEBOOK 01: BRONZE LAYER INGESTION
============================================================
  Phase 1, Step 1: Raw Data → Bronze Delta Tables
  
  This notebook loads the three raw datasets and saves them
  as Bronze Delta Tables with ZERO modifications.
  
  Bronze = Raw, untouched, source-of-truth data.
  
  Tables Created:
    • bronze_home_credit   — 307K rows, 122 cols (training distribution)
    • bronze_german_credit  — 1K rows, ~10 cols (drift/post-inflation batch)
    • bronze_policy_text    — RBI/SEBI regulatory guidelines (RAG ground truth)
============================================================
"""

import os
import sys
import time
from datetime import datetime

# ── Project path setup ──────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import (
    IS_DATABRICKS,
    HOME_CREDIT_CSV,
    GERMAN_CREDIT_CSV,
    POLICY_TEXT_FILE,
    BRONZE_HOME_CREDIT,
    BRONZE_GERMAN_CREDIT,
    BRONZE_POLICY_TEXT,
    CATALOG_NAME,
    SCHEMA_NAME,
    get_bronze_path,
    print_config,
)
from utils.spark_utils import get_spark_session, save_table

# ─────────────────────────────────────────────────────────────
#  STEP 0: Initialize Spark & Print Environment
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  🔮 DRIFTING ORACLE — Phase 1: Bronze Layer Ingestion")
print("=" * 70)

spark = get_spark_session("DriftingOracle_Phase1_Bronze")
print_config()

start_time = time.time()

# ─────────────────────────────────────────────────────────────
#  STEP 1.1: Unity Catalog Initialization (Databricks Only)
# ─────────────────────────────────────────────────────────────
if IS_DATABRICKS:
    print("\n" + "─" * 60)
    print("  🏗️ STEP 1.1: Unity Catalog Initialization")
    print("─" * 60)
    # Create the dedicated Catalog
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG_NAME}")
    # Create the Schema beneath it
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}")
    # Switch session context to this governed catalog and schema
    spark.sql(f"USE CATALOG {CATALOG_NAME}")
    spark.sql(f"USE SCHEMA {SCHEMA_NAME}")
    
    # Optionally set permissions (example for broad team access logic)
    print(f"  🔐 Assigning permissions for {CATALOG_NAME}.{SCHEMA_NAME}")
    try:
        spark.sql(f"GRANT USE CATALOG ON CATALOG {CATALOG_NAME} TO `account users`")
        spark.sql(f"GRANT USE SCHEMA, CREATE TABLE ON SCHEMA {CATALOG_NAME}.{SCHEMA_NAME} TO `account users`")
    except Exception as e:
        print(f"  ⚠️ Could not grant permissions (skip if running as basic admin or non-UC enabled cluster): {e}")

    print(f"  ✅ Unity Catalog successfully provisioned!")

# ─────────────────────────────────────────────────────────────
#  STEP 1.2: Ingest Home Credit (Training Distribution)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  📥 STEP 1.2: Loading Home Credit Default Risk Dataset")
print("─" * 60)



# Read CSV — inferSchema for automatic type detection
df_home_credit = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(HOME_CREDIT_CSV)
)

# Cache for multiple operations
df_home_credit.cache()

# Print schema summary
row_count = df_home_credit.count()
col_count = len(df_home_credit.columns)
print(f"  ✅ Loaded: {row_count:,} rows × {col_count} columns")
print(f"\n  📋 Schema Preview (first 15 columns):")
for col_name, col_type in df_home_credit.dtypes[:15]:
    print(f"     {col_name:<35} {col_type}")
print(f"     ... and {col_count - 15} more columns")

# Target distribution
print(f"\n  🎯 Target Distribution:")
df_home_credit.groupBy("TARGET").count().show()

# Save as Bronze Delta Table — NO MODIFICATIONS
bronze_home_path = get_bronze_path(BRONZE_HOME_CREDIT)
save_table(df_home_credit, bronze_home_path)
print(f"  💾 Bronze table saved: {BRONZE_HOME_CREDIT}")

# ─────────────────────────────────────────────────────────────
#  STEP 1.3: Ingest German Credit (Drift / Post-Inflation Batch)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  📥 STEP 1.3: Loading German Credit Dataset (Drift Batch)")
print("─" * 60)



df_german_credit = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(GERMAN_CREDIT_CSV)
)

df_german_credit.cache()

row_count_gc = df_german_credit.count()
col_count_gc = len(df_german_credit.columns)
print(f"  ✅ Loaded: {row_count_gc:,} rows × {col_count_gc} columns")
print(f"\n  📋 Full Schema:")
for col_name, col_type in df_german_credit.dtypes:
    print(f"     {col_name:<35} {col_type}")

# Purpose distribution
print(f"\n  🎯 Purpose Distribution:")
df_german_credit.groupBy("Purpose").count().show()

# Save as Bronze Delta Table — NO MODIFICATIONS
bronze_german_path = get_bronze_path(BRONZE_GERMAN_CREDIT)
save_table(df_german_credit, bronze_german_path)
print(f"  💾 Bronze table saved: {BRONZE_GERMAN_CREDIT}")

# ─────────────────────────────────────────────────────────────
#  STEP 1.4: Ingest RBI/SEBI Policy Text (RAG Ground Truth)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  📥 STEP 1.4: Loading RBI/SEBI Policy Guidelines")
print("─" * 60)

if not IS_DATABRICKS and not os.path.exists(POLICY_TEXT_FILE):
    raise FileNotFoundError(
        f"❌ Policy text file not found at: {POLICY_TEXT_FILE}\n"
        f"   This file should have been created by the project setup."
    )

# Read the policy text file line by line
# Each non-empty line becomes a row (paragraph)
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

with open(POLICY_TEXT_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Split into paragraphs (sections delineated by blank lines)
paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]

# Create structured DataFrame with paragraph ID and text
policy_rows = [
    Row(
        paragraph_id=i + 1,
        section=p.split(":")[0].strip() if ":" in p[:80] else "GENERAL",
        text=p,
        char_count=len(p),
        word_count=len(p.split()),
        ingestion_timestamp=datetime.now().isoformat(),
    )
    for i, p in enumerate(paragraphs)
]

import json
import tempfile

with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".json") as f:
    for row in policy_rows:
        # Convert Row to dict
        f.write(json.dumps(row.asDict()) + "\n")
    temp_json_path = f.name

df_policy = spark.read.json(temp_json_path)
df_policy.cache()

para_count = df_policy.count()
total_words = df_policy.agg({"word_count": "sum"}).collect()[0][0]
print(f"  ✅ Loaded: {para_count} paragraphs, {total_words:,} total words")
os.remove(temp_json_path)
print(f"\n  📋 Paragraph Preview:")
df_policy.select("paragraph_id", "section", "word_count").show(10, truncate=40)

# Save as Bronze Delta Table — NO MODIFICATIONS
bronze_policy_path = get_bronze_path(BRONZE_POLICY_TEXT)
save_table(df_policy, bronze_policy_path)
print(f"  💾 Bronze table saved: {BRONZE_POLICY_TEXT}")

# ─────────────────────────────────────────────────────────────
#  STEP 4: Bronze Layer Validation & Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  🏆 BRONZE LAYER — INGESTION COMPLETE")
print("=" * 70)

elapsed = time.time() - start_time

print(f"""
  ┌────────────────────────────────────────────────────────┐
  │  Bronze Layer Summary                                  │
  ├────────────────────────────────────────────────────────┤
  │  📊 bronze_home_credit:    {row_count:>8,} rows × {col_count:>3} cols  │
  │  📊 bronze_german_credit:  {row_count_gc:>8,} rows × {col_count_gc:>3} cols  │
  │  📝 bronze_policy_text:    {para_count:>8,} paragraphs           │
  │                                                        │
  │  ⏱️  Total time: {elapsed:.1f}s                                │
  │  💾 Storage: {'Unity Catalog' if IS_DATABRICKS else 'Local Delta/Parquet'}                         │
  └────────────────────────────────────────────────────────┘
""")

# ─────────────────────────────────────────────────────────────
#  STEP 5: Quick Data Quality Checks (for judges)
# ─────────────────────────────────────────────────────────────
print("─" * 60)
print("  🔍 Quick Data Quality Report — Home Credit")
print("─" * 60)

# Key columns for our pipeline
key_cols = [
    "TARGET", "DAYS_BIRTH", "AMT_CREDIT", "AMT_ANNUITY",
    "DAYS_EMPLOYED", "AMT_INCOME_TOTAL", "CODE_GENDER",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS"
]

from pyspark.sql.functions import col, count, when, isnan, isnull

# Null analysis for key columns
print("\n  Null counts for key columns:")
null_exprs = [
    count(when(isnull(col(c)) | isnan(col(c)), c)).alias(c)
    for c in key_cols
    if c in df_home_credit.columns
]
df_home_credit.select(null_exprs).show(vertical=True)

print("─" * 60)
print("  🔍 Quick Data Quality Report — German Credit")
print("─" * 60)
print("\n  Null counts:")
gc_null_exprs = [
    count(when(isnull(col(c)), c)).alias(c)
    for c in df_german_credit.columns
]
df_german_credit.select(gc_null_exprs).show(vertical=True)

# Descriptive stats
print("\n  📈 German Credit — Descriptive Statistics:")
df_german_credit.describe().show()

print("\n🏁 Phase 1 — Notebook 01 Complete. Bronze layer is live.\n")

# Cleanup
df_home_credit.unpersist()
df_german_credit.unpersist()
df_policy.unpersist()

# Don't stop spark in case next notebook is chained
# spark.stop()
