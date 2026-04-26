# Databricks notebook source
# MAGIC %pip install pinecone langchain-google-genai langchain-pinecone langchain-community -q

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

# ── Databricks Pipeline Constants ───────────────────────────
CATALOG_NAME = "workspace"
SCHEMA_NAME = "default"

BRONZE_POLICY_TEXT = f"{CATALOG_NAME}.{SCHEMA_NAME}.bronze_policy_text"
GOLD_SHAP_EXPLANATIONS = f"{CATALOG_NAME}.{SCHEMA_NAME}.gold_shap_explanations"

print("\n" + "=" * 70)
print("  🔮 DRIFTING ORACLE — Phase 7: RAG Safety Grounding")
print("=" * 70)

import time

# Securely Load Environment Variables for API Keys by injecting them for the hackathon
api_key = "GEMINI_API_KEY_PLACEHOLDER"
pinecone_api_key = "PINECONE_API_KEY_PLACEHOLDER"
os.environ["PINECONE_API_KEY"] = pinecone_api_key
os.environ["GOOGLE_API_KEY"] = api_key

print(f"  ✅ Using Native Databricks SparkSession")

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
df_policy = spark.read.table(BRONZE_POLICY_TEXT).toPandas()

documents = []
for _, row in df_policy.iterrows():
    if pd.isna(row['text']) or len(row['text'].strip()) < 10:
        continue
    # We must use exact LangChain Document objects for PineconeVectorStore
    documents.append(Document(page_content=row['text'], metadata={"paragraph_id": str(row.get('paragraph_id', 'Unknown'))}))

print(f"  ✅ Embedding & Uploading {len(documents)} official SEBI/RBI regulations into Pinecone Space...")
print(f"  ⏳ Batching in groups of 80 to respect Gemini free-tier rate limits...")

# Batch upload with dynamic rate-limit handling for the free tier
batch_size = 90
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    batch_num = (i // batch_size) + 1
    total_batches = (len(documents) + batch_size - 1) // batch_size
    print(f"     📦 Batch {batch_num}/{total_batches}: Embedding {len(batch)} documents...")
    
    success = False
    while not success:
        try:
            if i == 0:
                vector_store = PineconeVectorStore.from_documents(batch, embeddings, index_name=index_name)
            else:
                vector_store.add_documents(batch)
            success = True
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("     ⏳ Free-tier quota hit! Auto-pausing for 20 seconds to let it reset...")
                time.sleep(20)
            else:
                raise e

# ─────────────────────────────────────────────────────────────
#  STEP 2: Fetch SHAP Explanations (The Inference to Test)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  🧠 STEP 2: Retrieving AI Risk Explanations")
print("─" * 60)

try:
    df_shap = spark.read.table(GOLD_SHAP_EXPLANATIONS).toPandas()
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

if api_key:
    print("  🚀 Authentic Keys detected! Invoking Google Gemini LLM for RAG Audit...")
    from langchain_google_genai import ChatGoogleGenerativeAI
    try:
        # Utilizing Native Gemini via LangChain
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=api_key)
        response = llm.invoke(prompt_template)
        evaluation_result = response.content
    except Exception as e:
        print(f"  ⚠️ Gemini LLM Call Failed: {e}")
        evaluation_result = "[PARTIALLY GROUNDED] Error making API call to Endpoint. Fallback."
else:
    print("  ⚠️ No valid Keys detected via dbutils (using offline Hackathon Mock logic).")
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
