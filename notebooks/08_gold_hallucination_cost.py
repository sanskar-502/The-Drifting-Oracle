"""
============================================================
  DRIFTING ORACLE — NOTEBOOK 08: HALLUCINATION COST ANALYSIS
============================================================
  Phase 8: Business-Aware GenAI Governance Scoring
  
  For every AI-generated credit explanation, this script
  quantifies the financial and regulatory risk of deploying
  that explanation to a customer. It assigns:
  
    1. Grounding Score      — How well the explanation aligns with policy
    2. Hallucination Risk   — Probability the explanation is fabricated
    3. Impact Band          — Business severity (Low → Very High)
    4. Review Flag          — Auto-Approve / Manual Review / Block
    
  Output: gold_hallucination_cost (Delta Table)
============================================================
"""

import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd

# ── Project path setup ──────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import (
    BRONZE_POLICY_TEXT,
    get_bronze_path,
    get_gold_path,
    print_config,
)
from utils.spark_utils import get_spark_session, read_table, save_table

print("\n" + "=" * 70)
print("  🔮 DRIFTING ORACLE — Phase 8: Hallucination Cost Analysis")
print("=" * 70)

spark = get_spark_session("DriftingOracle_Phase8_HallucinationCost")
GOLD_HALLUCINATION_COST = "gold_hallucination_cost"

# ─────────────────────────────────────────────────────────────
#  STEP 1: Load Policy Vector DB + SHAP Explanations
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  📥 STEP 1: Loading Policy Corpus & AI Explanations")
print("─" * 60)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load policy chunks
policy_path = get_bronze_path(BRONZE_POLICY_TEXT)
df_policy = read_table(spark, policy_path).toPandas()

policy_chunks = []
for _, row in df_policy.iterrows():
    if pd.isna(row['text']):
        continue
    policy_chunks.append(row['text'])

print(f"  ✅ Policy Corpus: {len(policy_chunks)} regulatory modules loaded")

# Build TF-IDF Vector Space from policy
vectorizer = TfidfVectorizer(stop_words='english')
policy_vectors = vectorizer.fit_transform(policy_chunks)

# Load SHAP explanations from Gold layer
gold_shap_path = get_gold_path("gold_shap_explanations")
try:
    df_shap = read_table(spark, gold_shap_path).toPandas()
    print(f"  ✅ SHAP Explanations: {len(df_shap)} applicant audits loaded")
except Exception as e:
    print(f"  ⚠️ Could not load SHAP Gold table: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
#  STEP 2: Define Hallucination Risk Scoring Engine
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  🧮 STEP 2: Initializing Risk Scoring Engine")
print("─" * 60)

# High-risk regulatory keywords — if the explanation mentions these
# but they DON'T appear in the retrieved policy context, the risk
# escalates dramatically (wrong legal/compliance claims)
COMPLIANCE_KEYWORDS = [
    "rbi", "sebi", "regulation", "guideline", "compliance",
    "mandate", "statutory", "legal", "penalty", "violation",
    "reserve bank", "securities board", "act", "circular"
]

def calculate_grounding_score(explanation, vectorizer, policy_vectors):
    """Cosine similarity between explanation and nearest policy chunk."""
    query_vec = vectorizer.transform([explanation])
    similarities = cosine_similarity(query_vec, policy_vectors).flatten()
    # Top-2 average for stability
    top_2 = np.sort(similarities)[-2:]
    return float(np.mean(top_2))

def calculate_hallucination_risk(grounding_score):
    """Inverse of grounding — higher means more likely hallucinated."""
    return round(1.0 - grounding_score, 4)

def detect_compliance_claims(explanation):
    """Check if the explanation makes legal/regulatory claims."""
    explanation_lower = explanation.lower()
    found = [kw for kw in COMPLIANCE_KEYWORDS if kw in explanation_lower]
    return found

def classify_impact_band(grounding_score, compliance_claims_found):
    """
    Business impact classification:
      - Very High: Makes compliance claims that are unsupported
      - High:      Unsupported explanation on policy text
      - Medium:    Partially grounded, minor unsupported wording  
      - Low:       Well-grounded, safe to deploy
    """
    if compliance_claims_found and grounding_score < 0.30:
        return "VERY HIGH"
    elif grounding_score < 0.30:
        return "HIGH"
    elif grounding_score < 0.50:
        return "MEDIUM"
    else:
        return "LOW"

def assign_review_flag(impact_band):
    """Operational decision gate."""
    if impact_band == "VERY HIGH":
        return "🔴 BLOCK — Legal Review Required"
    elif impact_band == "HIGH":
        return "🔴 BLOCK — Senior Underwriter Review"
    elif impact_band == "MEDIUM":
        return "🟡 MANUAL REVIEW — Compliance Team"
    else:
        return "🟢 AUTO-APPROVE — Safe to Deploy"

def estimate_financial_exposure(impact_band, credit_amount=None):
    """
    Estimates the potential regulatory fine or reputational damage
    if a hallucinated explanation reaches a customer.
    Based on RBI penalty frameworks for unfair lending practices.
    """
    exposure_map = {
        "VERY HIGH": "₹50L–₹5Cr+ (Regulatory Penalty + License Risk)",
        "HIGH":      "₹10L–₹50L (Compliance Violation Fine)",
        "MEDIUM":    "₹1L–₹10L (Customer Complaint Escalation)",
        "LOW":       "Minimal (Within acceptable risk tolerance)"
    }
    return exposure_map.get(impact_band, "Unknown")

# ─────────────────────────────────────────────────────────────
#  STEP 3: Score Every AI Explanation
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  ⚖️ STEP 3: Scoring All AI Explanations")
print("─" * 60)

cost_results = []

for idx, row in df_shap.iterrows():
    explanation = row.get('explanation_summary', '')
    applicant_id = row.get('applicant_id', idx)
    risk_score_ml = row.get('total_risk_score', 0.0)
    
    # 1. Grounding Score
    grounding = calculate_grounding_score(explanation, vectorizer, policy_vectors)
    
    # 2. Hallucination Risk
    hallucination_risk = calculate_hallucination_risk(grounding)
    
    # 3. Compliance Claims Detection
    compliance_claims = detect_compliance_claims(explanation)
    
    # 4. Impact Band
    impact_band = classify_impact_band(grounding, compliance_claims)
    
    # 5. Review Flag
    review_flag = assign_review_flag(impact_band)
    
    # 6. Financial Exposure
    financial_exposure = estimate_financial_exposure(impact_band)
    
    cost_results.append({
        "applicant_id": int(applicant_id),
        "ml_risk_score": float(risk_score_ml),
        "explanation": str(explanation),
        "grounding_score": round(grounding, 4),
        "hallucination_risk": hallucination_risk,
        "compliance_claims_detected": ", ".join(compliance_claims) if compliance_claims else "None",
        "impact_band": impact_band,
        "review_flag": review_flag,
        "estimated_financial_exposure": financial_exposure,
        "audit_timestamp": pd.Timestamp.now().isoformat()
    })

df_cost_report = pd.DataFrame(cost_results)

# ─────────────────────────────────────────────────────────────
#  STEP 4: Display Governance Dashboard
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  🏛️ HALLUCINATION COST & GOVERNANCE REPORT")
print("=" * 70)

for _, r in df_cost_report.iterrows():
    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  👤 Applicant ID: {r['applicant_id']:<42} │
  ├─────────────────────────────────────────────────────────────┤
  │  🤖 ML Risk Score:        {r['ml_risk_score']:<34.1%} │
  │  📝 Explanation:          {r['explanation'][:40]:<34} │
  │                                                             │
  │  📊 Grounding Score:      {r['grounding_score']:<34.4f} │
  │  🎲 Hallucination Risk:   {r['hallucination_risk']:<34.4f} │
  │  ⚠️  Compliance Claims:   {r['compliance_claims_detected']:<34} │
  │                                                             │
  │  🏷️  Impact Band:         {r['impact_band']:<34} │
  │  🚦 Review Flag:          {r['review_flag']:<34} │
  │  💰 Financial Exposure:   {r['estimated_financial_exposure'][:34]:<34} │
  └─────────────────────────────────────────────────────────────┘""")

# Summary statistics
print("\n" + "─" * 60)
print("  📊 AGGREGATE GOVERNANCE SUMMARY")
print("─" * 60)

band_counts = df_cost_report['impact_band'].value_counts()
total = len(df_cost_report)
for band in ["LOW", "MEDIUM", "HIGH", "VERY HIGH"]:
    count = band_counts.get(band, 0)
    pct = (count / total * 100) if total > 0 else 0
    icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "VERY HIGH": "🚨"}.get(band, "⚪")
    print(f"  {icon} {band:<12}: {count}/{total} explanations ({pct:.0f}%)")

avg_grounding = df_cost_report['grounding_score'].mean()
avg_hallucination = df_cost_report['hallucination_risk'].mean()
print(f"\n  📈 Average Grounding Score:       {avg_grounding:.4f}")
print(f"  📉 Average Hallucination Risk:    {avg_hallucination:.4f}")

if avg_hallucination > 0.7:
    print("\n  🚨 SYSTEM ALERT: High average hallucination risk detected!")
    print("     Recommendation: Improve policy coverage or restrict GenAI outputs.")
elif avg_hallucination > 0.4:
    print("\n  🟡 SYSTEM NOTICE: Moderate hallucination risk across the board.")
    print("     Recommendation: Add more policy documents to the vector store.")
else:
    print("\n  🟢 SYSTEM HEALTHY: GenAI outputs are well-grounded in regulatory policy.")

# ─────────────────────────────────────────────────────────────
#  STEP 5: Save to Gold Layer
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  💾 STEP 5: Archiving to Gold Layer")
print("─" * 60)

with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".json") as f:
    for record in df_cost_report.to_dict(orient="records"):
        f.write(json.dumps(record) + "\n")
    temp_json_path = f.name

gold_path = get_gold_path(GOLD_HALLUCINATION_COST)
spark_cost = spark.read.json(temp_json_path)
save_table(spark_cost, gold_path)
os.remove(temp_json_path)

print(f"  ✅ Saved to Gold Delta table: {GOLD_HALLUCINATION_COST}")
print("\n🏁 Phase 8 Complete. GenAI governance risk quantified.\n")
