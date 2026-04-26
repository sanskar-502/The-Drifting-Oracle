# 🔮 The Drifting Oracle: Databricks Serverless Implementation

### An Enterprise-Grade Credit Risk ML System Refactored for Databricks Unity Catalog, Serverless Compute, and MLflow Model Registry

---

> **🏆 Hackathon Pitch — Databricks Edition:**  
> *"We took a self-healing credit risk AI and natively integrated it into the Databricks Lakehouse. It leverages Unity Catalog for governance, Serverless Compute for auto-scaling execution, MLflow for model registry lifecycle management, and deeply integrates `mlflow.evaluate()` with custom metrics for GenAI hallucination grounding—all culminating in a fully autonomous 9-stage pipeline that writes to a centralized Gold Audit Table."*

---

## 📑 Table of Contents

1. [Databricks Implementation Overview](#-databricks-implementation-overview)
2. [Serverless Architecture & Refactoring](#-serverless-architecture--refactoring)
3. [Data Flow Pipeline (Medallion)](#-data-flow-pipeline-medallion)
4. [Phase-by-Phase Notebook Breakdown](#-phase-by-phase-notebook-breakdown)
5. [Unity Catalog Schema](#-unity-catalog-schema)
6. [MLflow Model Registry](#-mlflow-model-registry)
7. [GenAI Integration (Pinecone & Gemini)](#-genai-integration-pinecone--gemini)
8. [Setup & Execution on Databricks](#-setup--execution-on-databricks)

---

## 🎯 Databricks Implementation Overview

This directory contains the **fully refactored** notebook suite optimized specifically for execution on **Databricks Serverless Compute**. 

While the original logic was built for local PySpark/Windows environments, this suite utilizes native Databricks globals (`spark`, `dbutils`) and seamlessly interacts with **Unity Catalog** and the **managed Databricks MLflow Registry**.

### Key Technology Upgrades:
- **Compute:** Databricks Serverless (Photon-accelerated)
- **Governance:** Unity Catalog (`workspace` catalog, `default` schema)
- **Data Ingestion:** Unity Catalog Volumes (`/Volumes/workspace/default/raw_hackathon_data/`)
- **Tracking:** Native Databricks MLflow tracking and Model Registry (`models:/workspace.default.credit_risk_model@Champion`)
- **RAG Engine:** Cloud Vector Database (Pinecone) + Google Gemini LLM API

---

## 🏗️ Serverless Architecture & Refactoring

To ensure compatibility with **Databricks Serverless** and **DBFS root security restrictions**, several major architectural changes were made to these notebooks:

1. **Eliminated Local Tempfile JSON Bridging:**
   Serverless compute blocks arbitrary writes to the DBFS root. All instances of passing pandas dataframes to Spark via local temp files were ripped out and replaced with optimized, direct `spark.createDataFrame()` initializations.
   
2. **Removed RDD Caching (`.cache()`, `.unpersist()`):**
   Calls mapped to traditional RDD caching memory trigger `[NOT_SUPPORTED_WITH_SERVERLESS]` errors on modern serverless clusters. All caching logic was abstracted, letting Databricks Photon automatically handle memory optimization.

3. **Dynamic Library Installation:**
   Notebooks requiring external libraries (`xgboost`, `shap`, `pinecone`, `langchain-google-genai`) utilize `%pip install -q` at the top of the notebook to dynamically pin package states to the serverless cluster execution context.

---

## 🔄 Data Flow Pipeline (Medallion)

The pipeline writes back and forth from **Unity Catalog**, enforcing a strict Medallion Architecture data lineage.

```text
/Volumes/workspace/default/raw_hackathon_data/
  ├── application_train.csv
  ├── german_credit_data.csv
  └── rbi_sebi_policy.txt

        │
        ▼

🟤 BRONZE LAYER (Notebook 01)
  ├── workspace.default.bronze_home_credit             
  ├── workspace.default.bronze_german_credit           
  └── workspace.default.bronze_policy_text             

        │
        ▼

⚪ SILVER LAYER (Notebook 02)
  ├── workspace.default.silver_home_credit             
  └── workspace.default.silver_german_credit           

        │
        ▼

🤖 ML ENGINE (Notebooks 03–06)
  ├── 03: Trains & Logs XGBoost to workspace.default.credit_risk_model@Champion
  ├── 04: Calculates PSI drift across features
  ├── 05: Evaluates Challenger, promotes to Version 2@Champion if better
  └── 06: Loads @Champion, generates SHAP values

        │
        ▼

🟡 GOLD LAYER (Notebooks 04–09)
  ├── workspace.default.gold_drift_metrics             
  ├── workspace.default.gold_model_comparison          
  ├── workspace.default.gold_shap_explanations         
  ├── workspace.default.gold_hallucination_cost        
  └── workspace.default.gold_audit_table      ← UNIFIED RECORD HUB
```

---

## 📊 Phase-by-Phase Notebook Breakdown

### Phase 1: Ingestion (`01_bronze_ingestion.py`)
Reads the raw data directly from Databricks Unity Catalog Volumes. Safely handles corrupted text fields and loads completely raw, unmodified tables into the Bronze schema. 

### Phase 2: Preprocessing (`02_silver_preprocessing.py`)
Applies data quality transformations, handles NULL values with median imputation, engineers synthetic feature proxies (`income_proxy`, `duration_months`), and writes clean feature-ready tables to the Silver schema. Spaces in column names are standardized to underscores to comply with Delta table restrictions.

### Phase 3: Baseline Training (`03_baseline_model_training.py`)
Trains Logistic Regression, Random Forest, and XGBoost on Silver data. Automatically evaluates based on ROC-AUC, registers the best-performing model to Unity Catalog MLflow Registry, and assigns the custom alias `@Champion`.

### Phase 4: Drift Detection (`04_gold_psi_drift_monitor.py`)
Calculates the Population Stability Index (PSI) iteratively across features to compare current applicant distributions against the baseline distributions. Any PSI > 0.20 triggers a master `Retraining Trigger`, logged to the Gold Layer.

### Phase 5: Autonomous Retraining (`05_gold_retraining_loop.py`)
Downloads the current active `@Champion` from MLflow. Simulates ground-truth maturity on the drifted batch, trains a Challenger model, evaluates them head-to-head. Due to the high drift, the Challenger defeats the baseline and is automatically published as a newly incremented Version, seizing the `@Champion` alias.

### Phase 6: Interpretable AI (`06_gold_shap_explainability.py`)
Pulls the live `@Champion` model. Instantiates a SHAP `TreeExplainer` on the cluster, iterating through applicant records to calculate global and local feature importance limits, explicitly attributing percentage risk points to specific fields. 

### Phase 7: RAG Grounding Verification (`07_gold_rag_grounding.py`)
Sets up a Pinecone Vector Database index on the fly. Embeds the 344 Bronze policy texts sequentially using Gemini APIs. Retrieves SHAP explanations and prompts a Google Gemini LLM instance to mathematically verify if the AI explained its decisions using features explicitly regulated by the banking guidelines. Output is marked `[GROUNDED]` or `[UNSUPPORTED]`.

### Phase 8: Financial Hallucination Cost via `mlflow.evaluate()` (`08_gold_hallucination_cost.py`)
Replaces manual Python iteration by registering custom `factual_grounding` and `hallucination_risk` metrics into Databricks MLflow using `mlflow.metrics.make_metric()`. The AI-generated explanations are scored natively out-of-the-box using the powerful `mlflow.evaluate()` methodology. It parses the results into theoretical Banking compliance fines based on severity banding (e.g., assigning a ₹50L regulatory penalty). 

### Phase 9: The Final Master Schema (`09_gold_audit_table.py`)
Aggregates Outputs 1 through 8 into the paramount `workspace.default.gold_audit_table`. This serves as the single source-of-truth for Dashboards, offering granular insight spanning mathematical score generation all the way through LLM hallucination oversight.

---

## 🗄️ Unity Catalog Schema

Ensure that you are running within a Databricks Workspace that has Unity Catalog enabled. The default catalog referenced throughout code is `workspace` and schema is `default`. 
- **Catalog:** `workspace`
- **Schema:** `default`
- **Volume:** `/Volumes/workspace/default/raw_hackathon_data/`

All `spark.read.table()` and `.write.saveAsTable()` function calls assume this environment taxonomy.

---

## 📦 MLflow Model Registry

Instead of saving `.pkl` files loosely, all models are pushed into Unity Catalog MLflow:
- **Registry Name:** `workspace.default.credit_risk_model`
- **Artifact Logging:** Retains full SKLearn/XGBoost context objects alongside models.
- **Aliases:** Automated workflows utilize `@Champion` tags instead of hardcoded numbers to ensure robust, auto-healing system architectures.

---

## 🤖 GenAI Integration (Pinecone & Gemini)

For Notebook 07 to function, the script connects to two external services: **Pinecone** Serverless indexes and **Google Gemini** for embedding/RAG.
> **Note:** The API keys are injected at the top of the notebook variables `api_key` and `pinecone_api_key`. In a true Databricks Production workspace, these should be securely stored in **Databricks Secret Scopes** (`dbutils.secrets.get()`). For the hackathon context, they are supplied in-script payload.
> The embedding uploads are automatically chunked into batches of 90 with intelligent `try/catch` rate-limiting logic to seamlessly respect free-tier Gemini API limitations.

---

## 🚀 Setup & Execution on Databricks

Since this folder's scripts are natively designed for execution inside Databricks Workspaces:

1. **Import the Notebooks:** Sync this `/databricks_notebooks/` folder directly to your Databricks Repo via Git.
2. **Mount the Data:** Ensure `application_train.csv`, `german_credit_data.csv`, and `rbi_sebi_policy.txt` are loaded into your UC Volume.
3. **Provision the Cluster:** Attach a Databricks **Serverless Compute** cluster.
4. **Execution Sequence:** Simply open each notebook (01 to 09) and press `Run All` in sequential order. 
5. **Dashboarding:** Upon finishing Notebook 09, navigate to **Databricks SQL** and connect a dashboard directly to the resultant `workspace.default.gold_audit_table`. 

---

*Engineered natively for Databricks Lakehouse & Unity Catalog.*
