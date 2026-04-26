import { useState, useEffect, useRef } from 'react';
import { Fingerprint, Play, MessageSquare, Bot, AlertTriangle, ShieldCheck, Activity, Send } from 'lucide-react';
import './index.css';

function App() {
  const [metrics, setMetrics] = useState(null);
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [pipelineState, setPipelineState] = useState("Trigger Databricks Workflow");
  const [selectedApp, setSelectedApp] = useState(null);

  // Copilot State
  const [messages, setMessages] = useState([{ role: "ai", text: "Hello! I am your Drifting Oracle Copilot. Select an applicant from the ledger and ask me anything." }]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Simulator State
  const [simValues, setSimValues] = useState({ age: 30, income: 50000, duration: 24 });
  const [simResult, setSimResult] = useState(null);

  useEffect(() => {
    fetchDashboard();
  }, []);

  const fetchDashboard = async () => {
    try {
      const metRes = await fetch("http://localhost:8000/api/dashboard/metrics");
      setMetrics(await metRes.json());
      const recRes = await fetch("http://localhost:8000/api/dashboard/audit_records?limit=15");
      const recData = await recRes.json();
      setRecords(recData.data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleSimulate = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(simValues)
      });
      setSimResult(await res.json());
    } catch (e) {
      console.error(e);
    }
  };

  const handleChat = async () => {
    if (!chatInput.trim()) return;
    
    const userMessage = { role: "user", text: chatInput };
    setMessages(prev => [...prev, userMessage]);
    setChatInput("");
    setChatLoading(true);

    let context = "No applicant selected.";
    if (selectedApp) {
      context = `Applicant #${selectedApp.applicant_id} has Risk: ${(selectedApp.model_prediction*100).toFixed(1)}%. Class: ${selectedApp.risk_classification}. SHAP: ${selectedApp.shap_summary}. Label: ${selectedApp.explanation_label}.`;
    }

    try {
      const res = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage.text, applicant_context: context })
      });
      const data = await res.json();
      setMessages(prev => [...prev, { role: "ai", text: data.reply }]);
    } catch (e) {
      setMessages(prev => [...prev, { role: "ai", text: "Error connecting to Gemini." }]);
    } finally {
      setChatLoading(false);
    }
  };

  const triggerPipeline = async () => {
    if (pipelineState !== "Trigger Databricks Workflow") return;
    
    setPipelineState("Evaluating German Credit Batch...");
    fetch("http://localhost:8000/api/trigger_pipeline", { method: "POST" });
    
    setTimeout(() => setPipelineState("⚠️ High PSI Drift Detected..."), 1200);
    setTimeout(() => setPipelineState("Retraining AutoML Challenger..."), 2800);
    setTimeout(() => setPipelineState("Swapping Champion to v2..."), 4500);
    
    setTimeout(() => {
      setPipelineState("Trigger Databricks Workflow");
      fetchDashboard();
    }, 6000);
  };

  // Auto-scroll chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Initial Simulator Run
  useEffect(() => {
    handleSimulate();
  }, [simValues.age, simValues.income, simValues.duration]);


  if (loading || !metrics) return <div style={{padding: '2rem'}}>Loading Workspace...</div>;

  return (
    <div className="container">
      <header className="header">
        <div>
          <h1 className="glow-text" style={{ fontSize: '2rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <Fingerprint size={32} />
            The Drifting Oracle
          </h1>
          <p className="text-muted" style={{ marginTop: '0.5rem' }}>Databricks Unified AI Operating System</p>
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <button 
            className="btn-primary" 
            onClick={triggerPipeline} 
            disabled={pipelineState !== "Trigger Databricks Workflow"}
            style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', minWidth: '300px', justifyContent: 'center' }}
          >
            <Play size={18} />
            {pipelineState}
          </button>
        </div>
      </header>

      {/* KPI Section */}
      <div className="grid-4">
        <div className="glass-panel kpi-card">
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span className="kpi-label">Feature PSI Drift</span>
          </div>
          <span className="kpi-value glow-text" style={{fontSize: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
            {metrics.avg_drift_psi} <AlertTriangle color="var(--monitor)" size={18} />
          </span>
        </div>
        <div className="glass-panel kpi-card">
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span className="kpi-label">Hallucination Rate</span>
          </div>
          <span className="kpi-value" style={{fontSize: '1.5rem'}}>{metrics.hallucination_rate}</span>
        </div>
        <div className="glass-panel kpi-card">
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span className="kpi-label">Blocks Prevented</span>
          </div>
          <span className="kpi-value" style={{fontSize: '1.5rem'}}>{metrics.blocks_prevented}</span>
        </div>
        <div className="glass-panel kpi-card">
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span className="kpi-label">Fines Saved</span>
          </div>
          <span className="kpi-value glow-text" style={{fontSize: '1.5rem'}}>{metrics.financial_exposure_saved}</span>
        </div>
      </div>

      <div className="main-layout">
        
        {/* COLUMN 1: EDGE SIMULATOR */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <h3 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Activity size={18} className="glow-text"/>
            Live SHAP Simulator
          </h3>
          <p className="text-muted" style={{ fontSize: '0.8rem', marginBottom: '1.5rem' }}>Adjust values to see risk drift on the edge.</p>
          
          <div className="slider-group">
            <label><span>Age</span> <span>{simValues.age} yrs</span></label>
            <input type="range" min="18" max="80" value={simValues.age} onChange={e => setSimValues({...simValues, age: parseInt(e.target.value)})} />
          </div>
          <div className="slider-group">
            <label><span>Income</span> <span>${simValues.income.toLocaleString()}</span></label>
            <input type="range" min="10000" max="150000" step="5000" value={simValues.income} onChange={e => setSimValues({...simValues, income: parseInt(e.target.value)})} />
          </div>
          <div className="slider-group">
            <label><span>Loan Duration</span> <span>{simValues.duration} mo</span></label>
            <input type="range" min="6" max="72" value={simValues.duration} onChange={e => setSimValues({...simValues, duration: parseInt(e.target.value)})} />
          </div>

          {simResult && (
            <div style={{ marginTop: '2rem', padding: '1rem', background: 'rgba(0,0,0,0.3)', borderRadius: '8px' }}>
              <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
                <span style={{ fontSize: '2rem', fontWeight: 600, color: simResult.predicted_risk > 0.65 ? 'var(--danger)' : 'var(--safe)' }}>
                  {(simResult.predicted_risk * 100).toFixed(1)}% Risk
                </span>
                <div style={{ fontSize: '0.8rem', marginTop: '0.5rem' }} className="badge monitor">{simResult.classification}</div>
              </div>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>"{simResult.shap_summary}"</p>
            </div>
          )}
        </div>

        {/* COLUMN 2: BATCH LEDGER */}
        <div className="glass-panel audit-table-wrapper">
          <table className="audit-table">
            <thead>
              <tr>
                <th>App ID</th>
                <th>Prediction</th>
                <th>GenAI Governance</th>
                <th>Model</th>
              </tr>
            </thead>
            <tbody>
              {records.map((rec, i) => {
                const isSelected = selectedApp?.applicant_id === rec.applicant_id;
                const isBlocked = rec.explanation_label.includes("BLOCK");
                const isManual = rec.explanation_label.includes("MANUAL");
                let badgeClass = 'safe';
                let badgeText = 'APPROVED';
                let Icon = ShieldCheck;
                if (isBlocked) { badgeClass = 'danger'; badgeText = 'BLOCKED'; Icon = AlertTriangle; }
                else if (isManual) { badgeClass = 'monitor'; badgeText = 'REVIEW'; Icon = Activity; }

                return (
                  <tr key={i} className={isSelected ? 'selected' : ''} onClick={() => setSelectedApp(rec)}>
                    <td>#{rec.applicant_id}</td>
                    <td>
                      <div style={{ fontWeight: 600 }}>{(rec.model_prediction * 100).toFixed(1)}%</div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{rec.risk_classification.split('—')[0]}</div>
                    </td>
                    <td>
                      <span className={`badge ${badgeClass}`}>
                        <Icon size={12} style={{marginRight: '4px'}}/>
                        {badgeText}
                      </span>
                    </td>
                    <td>
                      <span className={`tag-version ${rec.model_version === "2" ? 'tag-challenger' : 'tag-champion'}`}>
                        v{rec.model_version} {rec.model_version === "2" ? '(Challenger)' : '(Champion)'}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* COLUMN 3: COPILOT CHAT */}
        <div className="glass-panel chat-window">
          <div style={{ padding: '1rem', borderBottom: '1px solid var(--border-light)', background: 'rgba(255,255,255,0.02)' }}>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1rem' }}>
              <Bot size={18} className="glow-text"/>
              Underwriter Copilot
            </h3>
            {selectedApp && <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>Context: Applicant #{selectedApp.applicant_id}</p>}
          </div>
          
          <div className="chat-messages">
            {messages.map((m, i) => (
              <div key={i} className={`message ${m.role === 'user' ? 'user-msg' : 'ai-msg'}`}>
                {m.text}
              </div>
            ))}
            {chatLoading && <div className="message ai-msg" style={{ fontStyle: 'italic', color: 'var(--text-muted)' }}>Analyzing policy...</div>}
            <div ref={messagesEndRef} />
          </div>

          <div style={{ padding: '1rem', borderTop: '1px solid var(--border-light)' }}>
             <div style={{ display: 'flex', gap: '0.5rem' }}>
                <input 
                  type="text" 
                  value={chatInput}
                  onChange={e => setChatInput(e.target.value)}
                  onKeyPress={e => e.key === 'Enter' && handleChat()}
                  placeholder="Ask about compliance..." 
                />
                <button className="btn-primary" onClick={handleChat} style={{ padding: '0 1rem' }}>
                  <Send size={16} />
                </button>
             </div>
          </div>
        </div>

      </div>
    </div>
  );
}

export default App;
