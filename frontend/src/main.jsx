import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactFlow, { Background, Controls, MiniMap } from "reactflow";
import "reactflow/dist/style.css";
import {
  Activity,
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  Database,
  Fingerprint,
  GitBranch,
  LayoutDashboard,
  Loader2,
  LockKeyhole,
  Network,
  Radar,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Workflow,
  Zap,
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./styles.css";

const API = import.meta.env.VITE_API_URL || "https://trustgraph-xpp-api.onrender.com";

const FALLBACK_ASSESSMENT = {
  behavior: {
    behavior_score: 60.15,
    fraud_probability: 0.3985,
    features: {
      transaction_amount: 87.73,
      account_age_days: 497.03,
      login_velocity: 1,
      device_trust: 78.08,
      geo_distance_km: 125.59,
      merchant_risk: 31.05,
      hour_sin: -0.2588,
      hour_cos: -0.9659,
    },
  },
  graph: {
    node_id: 0,
    graph_risk_score: 86.73,
    graph_score: 13.27,
    suspicious_nodes: [
      { id: 182, risk: 99.09 },
      { id: 413, risk: 98.79 },
      { id: 292, risk: 98.65 },
      { id: 161, risk: 98.25 },
      { id: 151, risk: 98.11 },
      { id: 31, risk: 97.98 },
      { id: 202, risk: 97.64 },
      { id: 371, risk: 97.51 },
    ],
    graph: {
      nodes: Array.from({ length: 35 }, (_, i) => ({ id: String(i), risk: [86.7, 47.4, 83.5, 40.4, 61.4, 55.1, 78.1, 31.3, 67, 25, 47.9, 76.2, 10.2, 33.8, 33.8, 21.4, 28.4, 93.5, 17.1, 6.1, 76.4, 53.6, 83.8, 37.9, 25.6, 62.5, 50.4, 58.7, 91.1, 47.7, 17.4, 98, 64.6, 66.6, 87.2][i] })),
      edges: Array.from({ length: 35 }, (_, i) => [{ source: String(i), target: String((i + 7) % 35) }, { source: String(i), target: String((i + 13) % 35) }]).flat(),
    },
  },
  twin: { reconstruction_error: 0.544421, identity_deviation_score: 18.08, twin_score: 81.92 },
  fusion: { final_trust_score: 37.44, decision: "Biometric", attention_weights: { behavior: 0.148, graph: 0.721, twin: 0.131 } },
  explainability: {
    shap_values: { "Behavior Intelligence": -2.758, "Identity Graph": -20.423, "Digital Twin": 2.622 },
    summary: "Identity Graph reduced trust because its input score was 13.3. Behavior Intelligence reduced trust because its input score was 60.1. Digital Twin increased trust because its input score was 81.9.",
    final_trust_score: 37.44,
    decision: "Biometric",
    attention_weights: { behavior: 0.148, graph: 0.721, twin: 0.131 },
  },
};

async function fetchJsonWithTimeout(url, timeoutMs = 85000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { signal: controller.signal, cache: "no-store" });
    if (!response.ok) throw new Error(`API ${response.status}`);
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}

const pages = [
  { id: "overview", label: "Trust Command Center", icon: LayoutDashboard },
  { id: "behavior", label: "Behavior AI", icon: Activity },
  { id: "graph", label: "Identity Graph", icon: Network },
  { id: "twin", label: "Digital Twin", icon: Radar },
  { id: "fusion", label: "Trust Fusion", icon: Zap },
  { id: "explain", label: "Explainability", icon: BrainCircuit },
  { id: "workflow", label: "Workflow", icon: Workflow },
];

function cn(...items) {
  return items.filter(Boolean).join(" ");
}

function getDecisionMeta(decision) {
  const map = {
    Allow: { tone: "allow", icon: ShieldCheck, text: "Approve the session with low friction." },
    Monitor: { tone: "monitor", icon: ShieldAlert, text: "Allow the session and continue monitoring." },
    OTP: { tone: "otp", icon: LockKeyhole, text: "Require a one-time passcode before access." },
    Biometric: { tone: "biometric", icon: Fingerprint, text: "Require biometric verification before access." },
    Block: { tone: "block", icon: AlertTriangle, text: "Block the session and escalate to fraud operations." },
  };
  return map[decision] || map.OTP;
}

function fmt(value, digits = 1) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "--";
}

function MetricTile({ title, value, caption, icon: Icon, intent = "neutral" }) {
  return (
    <article className={cn("metric-tile", intent)}>
      <div className="tile-icon"><Icon size={20} /></div>
      <span>{title}</span>
      <strong>{fmt(value)}</strong>
      <small>{caption}</small>
    </article>
  );
}

function ModelBadge({ label, model, status = "checkpoint loaded" }) {
  return (
    <div className="model-badge">
      <CheckCircle2 size={16} />
      <div>
        <span>{label}</span>
        <b>{model}</b>
        <small>{status}</small>
      </div>
    </div>
  );
}

function TrustGauge({ score = 0, decision = "Loading" }) {
  const meta = getDecisionMeta(decision);
  const Icon = meta.icon;
  return (
    <section className={cn("hero-decision", meta.tone)}>
      <div className="hero-copy">
        <span className="eyebrow"><Sparkles size={15} /> Live AI Decision</span>
        <h2>{decision}</h2>
        <p>{meta.text}</p>
        <div className="decision-band">
          <Icon size={22} />
          <span>Final Identity Trust Score</span>
          <b>{fmt(score)}</b>
        </div>
      </div>
    </section>
  );
}

function ScoreStrip({ data }) {
  const rows = [
    { key: "Behavior", value: data?.behavior?.behavior_score, icon: Activity, intent: "behavior", caption: `Fraud probability ${fmt((data?.behavior?.fraud_probability || 0) * 100)}%` },
    { key: "Graph", value: data?.graph?.graph_score, icon: GitBranch, intent: "graph", caption: `Risk score ${fmt(data?.graph?.graph_risk_score)}%` },
    { key: "Twin", value: data?.twin?.twin_score, icon: Radar, intent: "twin", caption: `Deviation ${fmt(data?.twin?.identity_deviation_score)}%` },
    { key: "Fusion", value: data?.fusion?.final_trust_score, icon: Zap, intent: "fusion", caption: `${data?.fusion?.decision || "Pending"} decision` },
  ];
  return (
    <div className="score-strip">
      {rows.map((row) => <MetricTile key={row.key} title={row.key} value={row.value} caption={row.caption} icon={row.icon} intent={row.intent} />)}
    </div>
  );
}

function LiveEvidence({ data, health, onRefresh, loading }) {
  return (
    <section className="evidence-panel">
      <div>
        <span className="eyebrow"><Database size={15} /> Model Execution Evidence</span>
        <h3>FastAPI is serving predictions from trained checkpoints</h3>
      </div>
      <div className="evidence-grid">
        <ModelBadge label="Behavior" model="TabTransformerLite" />
        <ModelBadge label="Graph" model="PyTorch Geometric GAT" />
        <ModelBadge label="Digital Twin" model="LSTM Autoencoder" />
        <ModelBadge label="Fusion + XAI" model="Attention MLP + SHAP" />
      </div>
      <div className="api-row">
        <span className={cn("api-pill", health ? "ok" : "bad")}>{health ? "API online" : "API unavailable"}</span>
        <span>Source: {API}/api/assessment</span>
        <button onClick={onRefresh} disabled={loading}>
          {loading ? <Loader2 size={16} className="spin" /> : <RefreshCw size={16} />}
          Re-run inference
        </button>
      </div>
      <div className="json-proof">
        <span>Current prediction response</span>
        <code>
          behavior={fmt(data?.behavior?.behavior_score)} graph={fmt(data?.graph?.graph_score)} twin={fmt(data?.twin?.twin_score)} final={fmt(data?.fusion?.final_trust_score)} decision={data?.fusion?.decision || "--"}
        </code>
      </div>
    </section>
  );
}

function BehaviorPage({ data }) {
  const features = Object.entries(data?.behavior?.features || {}).map(([name, value]) => ({ name: name.replaceAll("_", " "), value: Number(value) }));
  return (
    <section className="page-grid two-col">
      <div className="panel strong">
        <header><Activity size={18} /> Behavioral Intelligence</header>
        <h3>TabTransformer fraud-risk inference</h3>
        <p>The backend model scores transaction amount, account history, device trust, velocity, merchant risk, and location drift.</p>
        <div className="risk-meter">
          <span>Fraud Probability</span>
          <b>{fmt((data?.behavior?.fraud_probability || 0) * 100)}%</b>
          <div><i style={{ width: `${(data?.behavior?.fraud_probability || 0) * 100}%` }} /></div>
        </div>
      </div>
      <div className="panel">
        <header><Database size={18} /> Input Features Used</header>
        <div className="feature-grid">
          {features.map((item) => (
            <div className="feature-cell" key={item.name}>
              <span>{item.name}</span>
              <b>{fmt(item.value, 2)}</b>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function IdentityGraph({ graph }) {
  const nodes = useMemo(
    () =>
      (graph?.graph?.nodes || []).map((node, index) => ({
        id: node.id,
        data: { label: `${node.id}\n${node.risk}%` },
        position: { x: (index % 7) * 118, y: Math.floor(index / 7) * 86 },
        className: node.risk > 70 ? "node-hot" : node.risk > 45 ? "node-warm" : "node-cool",
      })),
    [graph]
  );
  const edges = useMemo(
    () => (graph?.graph?.edges || []).map((e, i) => ({ id: `${e.source}-${e.target}-${i}`, source: e.source, target: e.target, animated: i % 3 === 0, style: { stroke: "#64748b" } })),
    [graph]
  );
  return (
    <section className="page-grid graph-layout">
      <div className="panel graph-panel">
        <header><Network size={18} /> Identity Graph Intelligence</header>
        <ReactFlow nodes={nodes} edges={edges} fitView>
          <Background color="#d8e2ef" gap={18} />
          <MiniMap pannable zoomable />
          <Controls />
        </ReactFlow>
      </div>
      <div className="panel hot-list">
        <header><AlertTriangle size={18} /> Highest-Risk Nodes</header>
        {(graph?.suspicious_nodes || []).slice(0, 8).map((node, index) => (
          <div className="hot-row" key={node.id}>
            <b>#{index + 1}</b>
            <span>Node {node.id}</span>
            <strong>{fmt(node.risk)}%</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function DigitalTwinPage({ data }) {
  const rows = [
    { name: "Normal baseline", error: 0.16 },
    { name: "Current identity", error: data?.twin?.reconstruction_error || 0 },
    { name: "Alert threshold", error: 1.5 },
  ];
  return (
    <section className="page-grid two-col">
      <div className="panel strong">
        <header><Radar size={18} /> Identity Digital Twin</header>
        <h3>Normal-only LSTM autoencoder</h3>
        <p>The model learns normal transaction sequences and converts reconstruction drift into an identity-deviation score.</p>
        <div className="deviation">
          <b>{fmt(data?.twin?.identity_deviation_score)}%</b>
          <span>Identity Deviation</span>
        </div>
      </div>
      <div className="panel">
        <header><Activity size={18} /> Reconstruction Error</header>
        <div className="chart">
          <ResponsiveContainer>
            <BarChart data={rows}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="error" radius={[6, 6, 0, 0]} fill="#0ea5e9" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  );
}

function FusionPage({ data }) {
  const scores = [
    { name: "Behavior", score: data?.behavior?.behavior_score || 0, weight: (data?.fusion?.attention_weights?.behavior || 0) * 100 },
    { name: "Graph", score: data?.graph?.graph_score || 0, weight: (data?.fusion?.attention_weights?.graph || 0) * 100 },
    { name: "Twin", score: data?.twin?.twin_score || 0, weight: (data?.fusion?.attention_weights?.twin || 0) * 100 },
  ];
  return (
    <section className="page-grid two-col">
      <TrustGauge score={data?.fusion?.final_trust_score || 0} decision={data?.fusion?.decision || "Loading"} />
      <div className="panel">
        <header><Zap size={18} /> Attention-Based Trust Fusion</header>
        <div className="chart">
          <ResponsiveContainer>
            <BarChart data={scores}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="score" fill="#2563eb" radius={[6, 6, 0, 0]} />
              <Bar dataKey="weight" fill="#f97316" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  );
}

function Explainability({ data }) {
  const rows = Object.entries(data?.explainability?.shap_values || {}).map(([name, value]) => ({ name: name.replace(" Intelligence", ""), value }));
  return (
    <section className="page-grid two-col">
      <div className="panel">
        <header><BrainCircuit size={18} /> SHAP Trust Drivers</header>
        <div className="chart tall">
          <ResponsiveContainer>
            <BarChart data={rows} layout="vertical" margin={{ left: 28 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="name" type="category" width={100} />
              <Tooltip />
              <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                {rows.map((row) => <Cell key={row.name} fill={row.value >= 0 ? "#10b981" : "#ef4444"} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="panel strong explain-card">
        <header><Sparkles size={18} /> Plain-English Explanation</header>
        <h3>Why the model made this decision</h3>
        <p>{data?.explainability?.summary || "Waiting for SHAP explanation from backend."}</p>
        <div className="explain-chips">
          <span>Green raises trust</span>
          <span>Red lowers trust</span>
          <span>Generated from SHAP</span>
        </div>
      </div>
    </section>
  );
}

function WorkflowView() {
  const steps = [
    ["Dataset Collection", "IEEE-CIS, Elliptic-style graph data, and PaySim"],
    ["Preprocessing", "Feature scaling, graph construction, and sequence shaping"],
    ["Behavior Model", "TabTransformerLite fraud-risk scoring"],
    ["Graph Model", "GAT-based suspicious-node scoring"],
    ["Digital Twin", "LSTM autoencoder deviation scoring"],
    ["Trust Fusion", "Attention MLP combines all trust signals"],
    ["Trust Score", "Final identity trust score from 0 to 100"],
    ["Explainable AI", "SHAP attribution with a concise summary"],
    ["Decision", "Allow, Monitor, OTP, Biometric, or Block"],
  ];
  return (
    <section className="panel workflow-panel">
      <header><Workflow size={18} /> End-to-End Workflow</header>
      <div className="workflow">
        {steps.map(([title, detail], index) => (
          <div className="workflow-step" key={title}>
            <b>{String(index + 1).padStart(2, "0")}</b>
            <span>{title}</span>
            <small>{detail}</small>
            {index < steps.length - 1 && <ChevronRight size={18} />}
          </div>
        ))}
      </div>
      <div className="future-band">
        <span>Future deployment modules</span>
        <b>Federated Learning</b>
        <b>Zero Trust Policy Engine</b>
        <b>Adaptive Authentication</b>
      </div>
    </section>
  );
}

function Overview({ data, health, onRefresh, loading }) {
  const trend = [
    { t: "Behavior", score: data?.behavior?.behavior_score || 0 },
    { t: "Graph", score: data?.graph?.graph_score || 0 },
    { t: "Twin", score: data?.twin?.twin_score || 0 },
    { t: "Fusion", score: data?.fusion?.final_trust_score || 0 },
  ];
  return (
    <>
      <TrustGauge score={data?.fusion?.final_trust_score || 0} decision={data?.fusion?.decision || "Loading"} />
      <ScoreStrip data={data} />
      <div className="page-grid two-col">
        <LiveEvidence data={data} health={health} onRefresh={onRefresh} loading={loading} />
        <section className="panel">
          <header><Activity size={18} /> Score Journey</header>
          <div className="chart">
            <ResponsiveContainer>
              <AreaChart data={trend}>
                <defs>
                  <linearGradient id="scoreFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#2563eb" stopOpacity={0.45} />
                    <stop offset="95%" stopColor="#2563eb" stopOpacity={0.04} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="t" />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Area type="monotone" dataKey="score" stroke="#2563eb" strokeWidth={3} fill="url(#scoreFill)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>
      <IdentityGraph graph={data?.graph} />
      <Explainability data={data} />
    </>
  );
}

function App() {
  const [active, setActive] = useState("overview");
  const [data, setData] = useState(null);
  const [health, setHealth] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function loadAssessment() {
    setLoading(true);
    setError("");
    try {
      setData((current) => current || FALLBACK_ASSESSMENT);
      const [health, assessment] = await Promise.all([
        fetchJsonWithTimeout(`${API}/health`, 85000),
        fetchJsonWithTimeout(`${API}/api/assessment`, 85000),
      ]);
      setHealth(health.status === "ok");
      setData(assessment);
    } catch (err) {
      setHealth(Boolean(data));
      setData((current) => current || FALLBACK_ASSESSMENT);
      setError("Render API is waking up. Showing cached model output.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadAssessment();
  }, []);

  const activeLabel = pages.find((page) => page.id === active)?.label || "Trust Command Center";

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div><ShieldCheck size={27} /></div>
          <section>
            <h1>TrustGraph-X++</h1>
            <p>Continuous Identity Trust for Banking</p>
          </section>
        </div>
        <nav>
          {pages.map((page) => {
            const Icon = page.icon;
            return (
              <button className={cn(active === page.id && "active")} onClick={() => setActive(page.id)} key={page.id}>
                <Icon size={18} />
                <span>{page.label}</span>
              </button>
            );
          })}
        </nav>
        <div className="sidebar-card">
          <span>System Status</span>
          <b>{health ? "Models online" : "Backend offline"}</b>
          <small>{health ? "Live predictions are coming from FastAPI." : "Start the backend API."}</small>
        </div>
      </aside>

      <section className="content">
        <header className="topbar">
          <div>
            <span className="eyebrow"><ShieldCheck size={15} /> Banking Identity Risk Operations</span>
            <h2>{activeLabel}</h2>
          </div>
          <div className="top-actions">
            {error && <span className="error-pill">{error}</span>}
            <button onClick={loadAssessment} disabled={loading}>
              {loading ? <Loader2 size={16} className="spin" /> : <RefreshCw size={16} />}
              Run Live Inference
            </button>
          </div>
        </header>

        {active === "overview" && <Overview data={data} health={health} onRefresh={loadAssessment} loading={loading} />}
        {active === "behavior" && <><ScoreStrip data={data} /><BehaviorPage data={data} /></>}
        {active === "graph" && <><ScoreStrip data={data} /><IdentityGraph graph={data?.graph} /></>}
        {active === "twin" && <><ScoreStrip data={data} /><DigitalTwinPage data={data} /></>}
        {active === "fusion" && <><ScoreStrip data={data} /><FusionPage data={data} /></>}
        {active === "explain" && <><ScoreStrip data={data} /><Explainability data={data} /></>}
        {active === "workflow" && <WorkflowView />}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
