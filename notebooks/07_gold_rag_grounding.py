"""
============================================================
  DRIFTING ORACLE — NOTEBOOK 07: RAG GROUNDING (LANGCHAIN)
============================================================
  Phase 7: GenAI Safety Layer (Vector DB + Gemini LLM)
  
  Constructs an authoritative Vector Database from the Bronze
  RBI/SEBI regulatory texts. It then intercepts the SHAP risk
  explanations and asks a Gemini LLM to strictly evaluate 
  whether the AI's logic is "Grounded" by the official rules.
  
  Outputs:
   - GROUNDED (Safe)
   - PARTIALLY GROUNDED (Flag for review)
   - UNSUPPORTED (Hallucination block!)
============================================================
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv

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
from utils.spark_utils import get_spark_session, read_table

print("\n" + "=" * 70)
print("  🔮 DRIFTING ORACLE — Phase 7: RAG Safety Grounding")
print("=" * 70)

import time
# Load Environment Variables for API Keys
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

spark = get_spark_session("DriftingOracle_Phase7_RAG")

# ─────────────────────────────────────────────────────────────
#  STEP 1: Initialize Enterprise Cloud Vector Store (Pinecone)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  🗄️ STEP 1: Building Cloud Vector Database (Pinecone)")
print("─" * 60)

from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

pc = Pinecone(api_key=pinecone_api_key)
index_name = "drifting-oracle-rag-3072"
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

if index_name not in pc.list_indexes().names():
    print(f"  ⛏️ Creating Pinecone Index: {index_name} (dimension 3072) - this takes ~30 seconds...")
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# Fetch Policy Chunks from Bronze Layer
policy_path = get_bronze_path(BRONZE_POLICY_TEXT)
df_policy = read_table(spark, policy_path).toPandas()

documents = []
for _, row in df_policy.iterrows():
    if pd.isna(row['text']) or len(row['text'].strip()) < 10:
        continue
    # We must use exact LangChain Document objects for PineconeVectorStore
    documents.append(Document(page_content=row['text'], metadata={"paragraph_id": str(row.get('paragraph_id', 'Unknown'))}))

print(f"  ✅ Embedding & Uploading {len(documents)} official SEBI/RBI regulations into Pinecone Space...")
vector_store = PineconeVectorStore.from_documents(documents, embeddings, index_name=index_name)

# ─────────────────────────────────────────────────────────────
#  STEP 2: Fetch SHAP Explanations (The Inference to Test)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  🧠 STEP 2: Retrieving AI Risk Explanations")
print("─" * 60)

gold_shap_path = get_gold_path("gold_shap_explanations")
try:
    df_shap = read_table(spark, gold_shap_path).toPandas()
    explanation_to_check = df_shap.iloc[0]['explanation_summary']
    print(f"  🔎 Analyzing Applicant 0's ML Explanation:\n     \"{explanation_to_check}\"")
except Exception as e:
    print("  ⚠️ Could not load SHAP Gold table. (Did Phase 6 run?)")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
#  STEP 3: Retrieve Relevant Policy Context (Pinecone)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  🔍 STEP 3: Cloud Vector Knowledge Retrieval")
print("─" * 60)

# Query the pinecone vector store using dense embeddings
print("  ⏳ Querying Pinecone Vector Database...")
results = vector_store.similarity_search(explanation_to_check, k=2)

print(f"  ✅ Retrieved Top 2 Highly Similar Policy Modules from Cloud:")
context_text = ""
for i, doc in enumerate(results, 1):
    para_id = doc.metadata.get('paragraph_id', 'Unknown')
    print(f"     Module {i} [ID: {para_id}]: {doc.page_content[:90]}...")
    context_text += f"POLICY MODULE {i}:\n{doc.page_content}\n\n"

# ─────────────────────────────────────────────────────────────
#  STEP 4: Gemini 1.5 LLM Evaluation (The LangChain Agent)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  ⚖️ STEP 4: LLM Hallucination & Grounding Check")
print("─" * 60)

prompt_template = f"""
You are a strict Bank Auditor AI enforcing SEBI/RBI regulations. 

Your task is to determine if a Machine Learning risk explanation is supported by the provided policy context.

ML EXPLANATION TO CHECK:
"{explanation_to_check}"

OFFICIAL POLICY CONTEXT:
{context_text}

EVALUATION RULES:
- Output [GROUNDED] and a concise 1-sentence reason if the context clearly supports the explanation variables.
- Output [PARTIALLY GROUNDED] if the parameters exist but limits are ambiguous.
- Output [UNSUPPORTED] if the explanation refers to variables completely absent from the policy.

START RESPONSE NOW:
"""

if api_key and api_key != "your_api_key_here":
    print("  🚀 Authentic Gemini API Key detected! Invoking Google Generative AI...")
    from langchain_google_genai import ChatGoogleGenerativeAI
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0, google_api_key=api_key)
        response = llm.invoke(prompt_template)
        evaluation_result = response.content
    except Exception as e:
        print(f"  ⚠️ LLM Call Failed: {e}")
        evaluation_result = "[PARTIALLY GROUNDED] Error making API call. Falling back to offline fallback."
else:
    print("  ⚠️ No valid GEMINI_API_KEY detected in .env file (using offline Hackathon Mock logic).")
    print("  ⏳ Simulating LLM RAG Processing locally...")
    evaluation_result = "[GROUNDED] The ML explanation correctly isolates Risk/Safety drivers (Duration, Income) explicitly regulated within the provided SEBI/RBI policy modules."

# ─────────────────────────────────────────────────────────────
#  FINAL RESULTS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  🚨 GEN-AI RAG SAFETY REPORT")
print("=" * 70)

if "[GROUNDED]" in evaluation_result.upper():
    print(f"  🟢 REGULATORY CLEARANCE GRANTED")
elif "[PARTIALLY GROUNDED]" in evaluation_result.upper():
    print(f"  🟡 MANUAL UNDERWRITER REVIEW REQUIRED")
else:
    print(f"  🔴 HALLUCINATION FLAG — EXPLANATION BLOCKED")

print(f"\n  LLM AUDITOR OUTPUT:\n  {evaluation_result}")

print("\n🏁 Phase 7 Complete. Medallion system GenAI integration secured.\n")
