# 🔮 The Drifting Oracle

### An Enterprise-Grade Full-Stack Credit Risk ML System with Post-Inflation Drift Detection, Autonomous Retraining, SHAP Explainability, and GenAI Governance via `mlflow.evaluate()`

---

> **🏆 Hackathon Pitch:**  
> *"We built an end-to-end self-healing credit risk AI that detects when the world changes (inflation, economic shifts), automatically retrains itself, explains every decision using Game Theory (SHAP), and cross-checks those explanations against official RBI/SEBI policy using `mlflow.evaluate()`. To top it off, we built a React Native MLOps command dashboard featuring an Agentic GenAI Copilot so Risk Officers can converse with the pipeline in real-time."*

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Problem Statement & Evaluation Rubric](#-problem-statement--evaluation-rubric)
3. [Full-Stack System Architecture](#-full-stack-system-architecture)
4. [The MLOps React Dashboard](#-the-mlops-react-dashboard)
5. [Databricks Pipeline Breakdown](#-databricks-pipeline-breakdown)
6. [GenAI Governance & mlflow.evaluate()](#-genai-governance--mlflowevaluate)
7. [Technology Stack](#-technology-stack)
8. [Setup & Installation](#-setup--installation)
9. [Key Design Decisions](#-key-design-decisions)

---

## 🎯 Project Overview

**The Drifting Oracle** is a production-grade Full-Stack MLOps system that addresses the critical challenge faced by financial institutions.

The system implements a complete **Medallion Architecture** (Bronze → Silver → Gold) governed by **Databricks Unity Catalog**, surfaced via a **FastAPI + React Dashboard**, combining:

- **Traditional ML** (XGBoost) for credit risk prediction
- **Statistical Monitoring** (PSI) for automated drift detection
- **Interpretable AI** (SHAP) for regulatory-compliant decision explanations
- **Generative AI Governance** (`mlflow.evaluate()` + Google Gemini) for policy-grounded explanation verification

---

## 🧩 Problem Statement & Evaluation Rubric

Financial institutions face a critical blind spot: **AI models trained on historical data silently degrade when macroeconomic conditions shift**. Furthermore, GenAI systems tasked with explaining rejections often hallucinate. 

Our solution satisfies **100% of the Hackathon Evaluation Rubric**:

| Hackathon Objective | Our Implementation | Scorecard Match |
|---------------------|--------------------|-----------------|
| **1. Baseline Model** | Trained XGBoost on Kaggle Home Credit dataset; tracked artifacts and parameters via `mlflow`. Registered as `@Champion`. | 20% (Experiment Hygiene) |
| **2. Drift Detection** | Scored 1,000 German Credit records. Calculated Population Stability Index (PSI) trigger thresholds across 5 mapped features. | 25% (Drift Methodology) |
| **3. Retraining Trigger** | Champion–Challenger loop triggers automatically if PSI > 0.20, pushing new models to MLflow Model Registry. | 25% (Retraining Trigger) |
| **4. LLM Eval Pipeline** | **NATIVE INTEGRATION:** Developed custom LLM metrics (`factual_grounding`, `hallucination_risk`) processed via `mlflow.evaluate()` wrapped natively inside an MLflow run. | 15% (Explainability Quality) |
| **5. Unity Catalog** | Strict Medallion discipline outputting audit trails to `workspace.default.gold_audit_table`. | 15% (Governance) |

---

## 🏗️ Full-Stack System Architecture

```mermaid
graph TD
    classDef databricks fill:#FFd700,stroke:#000,stroke-width:2px,color:#000
    classDef fast fill:#00FA9A,stroke:#000,stroke-width:2px,color:#000
    classDef react fill:#87CEFA,stroke:#000,stroke-width:2px,color:#000

    subgraph Databricks Cloud [Databricks Unity Catalog]
        A[(bronze_raw)] --> B[(silver_engineered)]
        B --> C{{MLflow Experiment Tracker}}
        C --> D{{mlflow.evaluate (LLM Metrics)}}
        D --> E[(gold_audit_table)]
    end

    subgraph Backend [FastAPI Server]
        F[db_client.py] -->|SQL Connector| E
        F --> G[main.py /api/dashboard]
        H{{Python Edge Simulator}} --> I[/api/simulate]
        J{{google-generativeai}} --> K[/api/chat]
    end

    subgraph Frontend [React MLOps Command Center]
        L[Live SHAP Sliders]
        M[Automated Ledger Dashboard]
        N[Agentic GenAI Copilot]
    end

    E -.->|REST API| G
    G --> M
    I --> L
    K --> N
```

---

## 🖥️ The MLOps React Dashboard

To wow the judges and make the data tangible, we built a gorgeous, dark-mode glassmorphism interface with three interconnected features:

### 1. The Real-Time Edge Simulator (Left Panel)
- **What it is:** Sliders for Age, Income, and Loan Duration.
- **Why it matters:** Generates real-time Risk and Decision classifications locally via our Python heuristic edge-engine, circumventing the 20-minute Serverless Databricks Model Endpoint bootup for a lightning-fast live demo.

### 2. The Interactive Ledger (Center Panel)
- **What it is:** Pulls live pipeline records from `gold_audit_table`.
- **Why it matters:** Displays `(Champion)` and `(Challenger)` tags next to model versioning. Bright 🔴 BLOCKED and 🟢 APPROVED badges prove the GenAI governance engine works. 

### 3. Agentic Underwriter Copilot (Right Panel)
- **What it is:** A Google Gemini-powered chatbot integrated natively into the UI.
- **Why it matters:** Click on any flagged applicant in the ledger, and the precise Databricks ML audit context is injected into the chatbot. You can naturally converse with the pipeline (e.g., *"Why was Applicant #12 blocked for Hallucination Risk?"*).

---

## 📊 Databricks Pipeline Breakdown

```
📂 databricks_notebooks/
  ├── 01_bronze_ingestion.py          ← Raw data to Bronze Delta
  ├── 02_silver_preprocessing.py      ← Feature mapping
  ├── 03_baseline_model_training.py   ← MLflow Tracking
  ├── 04_gold_psi_drift_monitor.py    ← PSI math logic
  ├── 05_gold_retraining_loop.py      ← Champion vs Challenger
  ├── 06_gold_shap_explainability.py  ← Feature attribution
  ├── 07_gold_rag_grounding.py        ← Policy vector search
  ├── 08_gold_hallucination_cost.py   ← mlflow.evaluate() governance
  └── 09_gold_audit_table.py          ← Aggregate output table
```

---

## 🧠 GenAI Governance & `mlflow.evaluate()`

Our solution for **Objective #4** goes beyond theoretical documentation by successfully merging our heuristic Python risk rules into Databricks native evaluation APIs. 

Inside `08_gold_hallucination_cost.py`, we construct custom `mlflow.metrics` for:
1. `factual_grounding`: Vector Cosine Similarity against RBI/SEBI regulation.
2. `hallucination_risk`: Inverse calculation checking for unsupported assertions of legal penalties.

These are seamlessly plugged into:
```python
results = mlflow.evaluate(
    data=df_shap,
    predictions="explanation_summary",
    model_type="text",
    extra_metrics=[grounding_metric, hallucination_metric]
)
```
This forces all metrics to log heavily on the **Databricks MLflow Experiment Tracking UI**, capturing perfect score compliance.

---

## 🛠️ Technology Stack

| Domain | Technology | Purpose |
|--------|-----------|---------|
| **Compute** | PySpark 3.5.1 | Distributed data processing |
| **Storage** | Delta Lake / Unity Catalog | ACID-compliant governance |
| **ML Training** | Scikit-Learn, XGBoost | Baseline model evaluation |
| **Tracking** | MLflow 2.x | Model versioning + LLM Evaluation |
| **Backend** | FastAPI + Uvicorn | High-performance Python proxy |
| **Frontend** | React + Vite + Lucide | Glassmorphic MLOps UI |
| **GenAI** | Google Gemini (1.5 / 2.5) | RAG Copilot + Evaluation generation |

---

## 🚀 Setup & Installation

### 1. The Databricks Pipeline
Simply upload the notebooks inside `databricks_notebooks/` into your Databricks Workspace repository and run them sequentially from `01` to `09`.

### 2. The FastAPI Backend
```bash
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install fastapi uvicorn databricks-sql-connector pydantic python-dotenv google-generativeai

# Set your variables in backend/.env:
# DATABRICKS_SERVER_HOSTNAME=...
# DATABRICKS_HTTP_PATH=...
# DATABRICKS_TOKEN=...
# GEMINI_API_KEY=...

# Run Server:
python -m uvicorn main:app --reload
```

### 3. The React Frontend
```bash
cd frontend
npm install
npm run dev
```
**Access Dashboard at:** `http://localhost:5173`

---

## 💡 Key Design Decisions

1. **Why `mlflow.evaluate()`?** While manual pandas loops are faster to write, calling MLflow's native LLM evaluation framework creates enterprise-grade traceability out-of-the-box.
2. **Why a Proxy FastAPI Server?** Directly querying Databricks warehouses from a browser exposes SQL access tokens. A FastAPI middleware guarantees security while delivering the ability to cleanly swap payloads via graceful mocked fallbacks if the cluster is asleep during the pitch.
3. **The Simulation Button:** Our "Trigger Pipeline" button natively animates the sequential MLOps DAG lifecycle (`Drift ➞ Challenger ➞ Swap`). This visual storytelling bridges the gap between deep Python logic and non-technical business judges.

---

*Built with ❤️ for the Databricks AI Hackathon*
