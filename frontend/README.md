# 🔮 The Drifting Oracle — React MLOps Dashboard

### A glassmorphic, real-time command center for monitoring Credit Risk AI, GenAI Governance, and Model Drift

---

## 🎯 Purpose

This is the **visual layer** of The Drifting Oracle. It is a React + Vite single-page application designed for a **Bank Risk Officer** to monitor, query, and interact with the entire Databricks ML pipeline from a single screen.

The dashboard features three interconnected panels:

| Panel | Feature | Data Source |
|-------|---------|-------------|
| **Left** | Live SHAP Edge Simulator | FastAPI `/api/simulate` (heuristic engine) |
| **Center** | Batch Applicant Ledger | FastAPI `/api/dashboard/audit_records` → Databricks SQL |
| **Right** | Agentic Underwriter Copilot | FastAPI `/api/chat` → Google Gemini |

---

## 📁 File Structure

```
frontend/
├── src/
│   ├── App.jsx           ← Main application component (all 3 panels)
│   ├── index.css          ← Glassmorphism design system + layouts
│   └── main.jsx           ← Vite entry point
├── index.html             ← Root HTML shell
├── package.json           ← Dependencies & scripts
├── vite.config.js         ← Vite dev server configuration
└── README.md              ← You are here
```

---

## ✨ UI Features

### 1. Top KPI Ribbon (4 Cards)
Displays four real-time aggregate metrics pulled from the Databricks Gold layer:
- **Feature PSI Drift** — Average Population Stability Index with warning icon
- **Hallucination Rate** — Percentage of AI explanations flagged as risky
- **Blocks Prevented** — Count of GenAI outputs intercepted before reaching customers
- **Fines Saved** — Estimated regulatory penalty savings in ₹ Crores

### 2. Live SHAP Edge Simulator (Left Panel)
Three interactive sliders (Age, Income, Loan Duration) that call the FastAPI backend in real-time. As the user adjusts values, the predicted risk percentage and classification update instantly — demonstrating how the ML model's decision boundary shifts with input changes.

### 3. Batch Applicant Ledger (Center Panel)
A scrollable, clickable table displaying live records from `gold_audit_table`:
- **Prediction column** — Shows ML risk score with classification label
- **GenAI Governance column** — Color-coded badges: 🟢 APPROVED, 🟡 REVIEW, 🔴 BLOCKED
- **Model column** — Tags showing `v1 (Champion)` in green or `v2 (Challenger)` in orange

Clicking any row selects that applicant as context for the Copilot chatbot.

### 4. Agentic Underwriter Copilot (Right Panel)
A Google Gemini-powered chatbot that receives the selected applicant's full Databricks context (risk score, SHAP factors, governance label). The Risk Officer can ask natural-language questions like:
- *"Why was this applicant blocked?"*
- *"What would the regulatory consequence have been?"*
- *"Is there enough evidence to override the block?"*

### 5. Pipeline Orchestration Button
The "Trigger Databricks Workflow" button in the header animates through the ML pipeline stages:
1. `Evaluating German Credit Batch...`
2. `⚠️ High PSI Drift Detected...`
3. `Retraining AutoML Challenger...`
4. `Swapping Champion to v2...`

This visually tells the complete MLOps lifecycle story during the hackathon pitch.

---

## 🎨 Design System

The UI uses a custom **glassmorphism** design system built with vanilla CSS:

- **Font:** Inter (Google Fonts)
- **Theme:** Dark mode with deep navy background (`#0a0f18`)
- **Effects:** `backdrop-filter: blur(12px)`, subtle radial gradients, smooth hover transitions
- **Color Palette:**
  - Safe: `#10b981` (emerald green)
  - Monitor: `#f59e0b` (amber)
  - Danger: `#ef4444` (red)
  - Accent: `#3b82f6` → `#8b5cf6` (blue-to-purple gradient)

---

## 🚀 Quick Start

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

**Dashboard available at:** `http://localhost:5173`

> ⚠️ The FastAPI backend must be running on `http://localhost:8000` for data to load.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `react` | UI framework |
| `vite` | Lightning-fast dev server & bundler |
| `lucide-react` | Beautiful, consistent icon library |

---

*Part of The Drifting Oracle — Databricks AI Hackathon*
