import os
import time
import asyncio
from typing import Dict, Any, List
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from db_client import DatabricksClient

app = FastAPI(title="Drifting Oracle API", description="Underwriter Dashboard Backend")

# Allow Frontend to communicate with Backend during local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = DatabricksClient()

# Initialize Gemini GenAI Client
from dotenv import load_dotenv
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    # Using the exact model requested by user
    chat_model = genai.GenerativeModel('gemini-2.5-flash-lite')
else:
    chat_model = None
    print("[WARNING] GEMINI_API_KEY not found. Chat endpoint will fall back to mock.")

# Pydantic Schemas for validation
class SimulateRequest(BaseModel):
    age: int
    income: float
    duration: int

class ChatRequest(BaseModel):
    message: str
    applicant_context: str = ""

@app.get("/")
def read_root():
    return {"status": "Drifting Oracle Backend is Running natively"}

@app.get("/api/dashboard/metrics")
def get_metrics():
    return db.fetch_dashboard_metrics()

@app.get("/api/dashboard/audit_records")
def get_audit_records(limit: int = 50):
    return {"data": db.fetch_audit_records(limit=limit)}

@app.post("/api/simulate")
async def run_live_prediction(data: SimulateRequest):
    """
    Heuristic SHAP Edge-Simulator for Hackathon.
    Normally this hits Databricks Model Serving Endpoints. 
    We simulate calculation latency here.
    """
    await asyncio.sleep(0.5) # Simulate API networking latency
    
    # Simple risk heuristic based on Income vs Duration
    risk_factor = (data.duration * 1000) / (data.income + 1)
    base_risk = 0.20
    
    if data.age < 25:
        base_risk += 0.15
        
    final_risk = min(0.99, max(0.01, base_risk + (risk_factor * 0.05)))
    
    classification = "MEDIUM RISK"
    label = "🟡 MANUAL REVIEW — Compliance Team"
    
    if final_risk > 0.65:
        classification = "HIGH RISK"
        label = "🔴 BLOCK — Senior Underwriter Review"
    elif final_risk < 0.25:
        classification = "LOW RISK"
        label = "🟢 AUTO-APPROVE"

    return {
        "status": "success",
        "predicted_risk": final_risk,
        "classification": classification,
        "shap_summary": f"Risk primarily driven UP by loan_duration (+{(data.duration/10):.2f}) and driven DOWN by income_proxy",
        "hallucination_flag": label
    }

@app.post("/api/chat")
async def ask_underwriter(req: ChatRequest):
    """
    Agentic AI endpoint utilizing Gemini to question the Databricks Gold Table context.
    """
    if chat_model:
        prompt = f"""You are the 'Drifting Oracle AI', a senior compliance risk auditor working for a bank.
        The user is asking you a question about a loan application. 
        Here is the Databricks Database Context for the applicant: {req.applicant_context}
        
        Question: {req.message}
        
        Respond naturally as an AI Copilot. Keep it brief (2 sentences max). Always cite the Database Context.
        """
        try:
            response = chat_model.generate_content(prompt)
            return {"reply": response.text}
        except Exception as e:
            return {"reply": f"[Gemini Error]: {str(e)}"}
    else:
        await asyncio.sleep(1)
        return {"reply": "I am operating in completely offline mock mode because the Gemini API key was missing. But if I had one, I would explain that Applicant #4 was blocked due to violation of RBI guidelines!"}

@app.post("/api/trigger_pipeline")
async def trigger_run():
    """
    Simulates sending an execution request to Databricks Workflows.
    """
    return {"status": "Job 7158 triggered successfully", "job_id": 7158}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
