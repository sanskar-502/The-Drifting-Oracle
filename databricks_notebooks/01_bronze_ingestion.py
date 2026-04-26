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

# ── Databricks Pipeline Constants ───────────────────────────
CATALOG_NAME = "workspace"
SCHEMA_NAME = "default"
VOLUME_PATH = f"/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}/raw_hackathon_data/"
HOME_CREDIT_CSV = VOLUME_PATH + "application_train.csv"
GERMAN_CREDIT_CSV = VOLUME_PATH + "german_credit_data.csv"
POLICY_TEXT_FILE = VOLUME_PATH + "rbi_sebi_policy.txt"

BRONZE_HOME_CREDIT = f"{CATALOG_NAME}.{SCHEMA_NAME}.bronze_home_credit"
BRONZE_GERMAN_CREDIT = f"{CATALOG_NAME}.{SCHEMA_NAME}.bronze_german_credit"
BRONZE_POLICY_TEXT = f"{CATALOG_NAME}.{SCHEMA_NAME}.bronze_policy_text"

# ─────────────────────────────────────────────────────────────
#  STEP 0: Initialize Spark & Print Environment
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  🔮 DRIFTING ORACLE — Phase 1: Bronze Layer Ingestion")
print("=" * 70)

# In Databricks, 'spark' is natively available in the global scope.
print(f"  ✅ Using Native Databricks SparkSession")

start_time = time.time()

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
# df_home_credit.cache() # Removed: Not supported on Serverless Web Compute

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
df_home_credit.write.format("delta").mode("overwrite").saveAsTable(BRONZE_HOME_CREDIT)
print(f"  💾 Bronze table saved to Unity Catalog: {BRONZE_HOME_CREDIT}")

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

# df_german_credit.cache() # Removed: Not supported on Serverless Web Compute

row_count_gc = df_german_credit.count()
col_count_gc = len(df_german_credit.columns)
print(f"  ✅ Loaded: {row_count_gc:,} rows × {col_count_gc} columns")
print(f"\n  📋 Full Schema:")
for col_name, col_type in df_german_credit.dtypes:
    print(f"     {col_name:<35} {col_type}")

# Purpose distribution
print(f"\n  🎯 Purpose Distribution:")
df_german_credit.groupBy("Purpose").count().show()

# Clean column names for Unity Catalog compatibility (No Spaces)
cleaned_columns = [c.replace(" ", "_") for c in df_german_credit.columns]
df_german_credit = df_german_credit.toDF(*cleaned_columns)

# Save as Bronze Delta Table — NO MODIFICATIONS
df_german_credit.write.format("delta").mode("overwrite").saveAsTable(BRONZE_GERMAN_CREDIT)
print(f"  💾 Bronze table saved to Unity Catalog: {BRONZE_GERMAN_CREDIT}")

# ─────────────────────────────────────────────────────────────
#  STEP 1.4: Ingest RBI/SEBI Policy Text (RAG Ground Truth)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  📥 STEP 1.4: Loading RBI/SEBI Policy Guidelines")
print("─" * 60)


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

df_policy = spark.createDataFrame(policy_rows)
# df_policy.cache() # Removed: Not supported on Serverless Web Compute

para_count = df_policy.count()
total_words = df_policy.agg({"word_count": "sum"}).collect()[0][0]
print(f"  ✅ Loaded: {para_count} paragraphs, {total_words:,} total words")
print(f"\n  📋 Paragraph Preview:")
df_policy.select("paragraph_id", "section", "word_count").show(10, truncate=40)

# Save as Bronze Delta Table — NO MODIFICATIONS
df_policy.write.format("delta").mode("overwrite").saveAsTable(BRONZE_POLICY_TEXT)
print(f"  💾 Bronze table saved to Unity Catalog: {BRONZE_POLICY_TEXT}")

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
  │  💾 Storage: Unity Catalog                              │
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

from pyspark.sql.functions import col, count, when, isnull

# Null analysis for key columns (isnull only — isnan fails on string columns)
print("\n  Null counts for key columns:")
null_exprs = [
    count(when(isnull(col(c)), c)).alias(c)
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
