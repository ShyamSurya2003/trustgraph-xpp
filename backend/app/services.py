import numpy as np
import shap
import torch

from .config import BEHAVIOR_FEATURES, TWIN_FEATURES
from .data import default_behavior_payload, default_twin_sequence
from .training import ensure_behavior_model, ensure_fusion_model, ensure_graph_model, ensure_twin_model


def decision(score: float) -> str:
    if score >= 90:
        return "Allow"
    if score >= 70:
        return "Monitor"
    if score >= 50:
        return "OTP"
    if score >= 30:
        return "Biometric"
    return "Block"


class TrustGraphService:
    def __init__(self):
        self.behavior_model, self.behavior_scaler = ensure_behavior_model()
        self.graph_model, self.graph_x, self.graph_edge, self.graph_y = ensure_graph_model()
        self.twin_model, self.twin_scaler, self.twin_threshold = ensure_twin_model()
        self.fusion_model = ensure_fusion_model()

    def behavior(self, payload: dict | None = None):
        payload = payload or default_behavior_payload()
        values = np.array([[float(payload.get(k, 0.0)) for k in BEHAVIOR_FEATURES]], dtype="float32")
        x = torch.tensor(self.behavior_scaler.transform(values).astype("float32"))
        with torch.no_grad():
            fraud_prob = torch.sigmoid(self.behavior_model(x)).item()
        trust = round((1 - fraud_prob) * 100, 2)
        return {"behavior_score": trust, "fraud_probability": round(fraud_prob, 4), "features": dict(zip(BEHAVIOR_FEATURES, values[0].tolist()))}

    def graph(self, node_id: int | None = None):
        node_id = int(node_id or 0) % len(self.graph_x)
        with torch.no_grad():
            probs = torch.softmax(self.graph_model(self.graph_x, self.graph_edge), dim=-1)[:, 1].numpy()
        risk = float(probs[node_id] * 100)
        suspicious = np.argsort(probs)[-10:][::-1].tolist()
        edges = self.graph_edge.numpy().T[:90].tolist()
        return {
            "node_id": node_id,
            "graph_risk_score": round(risk, 2),
            "graph_score": round(100 - risk, 2),
            "suspicious_nodes": [{"id": int(i), "risk": round(float(probs[i] * 100), 2)} for i in suspicious],
            "graph": {"nodes": [{"id": str(i), "risk": round(float(probs[i] * 100), 1)} for i in range(35)], "edges": [{"source": str(a), "target": str(b)} for a, b in edges if a < 35 and b < 35]},
        }

    def twin(self, sequence: list[list[float]] | None = None):
        sequence = sequence or default_twin_sequence()
        arr = np.array(sequence, dtype="float32")
        if arr.ndim != 2 or arr.shape[1] != len(TWIN_FEATURES):
            raise ValueError(f"sequence must be [steps, {len(TWIN_FEATURES)}]")
        x = self.twin_scaler.transform(arr).reshape(1, arr.shape[0], arr.shape[1]).astype("float32")
        with torch.no_grad():
            recon = self.twin_model(torch.tensor(x)).numpy()
        error = float(((recon - x) ** 2).mean())
        deviation = float(np.clip((error / max(self.twin_threshold, 1e-6)) * 50, 0, 100))
        return {"reconstruction_error": round(error, 6), "identity_deviation_score": round(deviation, 2), "twin_score": round(100 - deviation, 2)}

    def fusion(self, behavior_score: float, graph_score: float, twin_score: float):
        scores = torch.tensor([[behavior_score, graph_score, twin_score]], dtype=torch.float32) / 100.0
        with torch.no_grad():
            final, weights = self.fusion_model(scores)
        score = round(float(final.item()), 2)
        return {
            "final_trust_score": score,
            "decision": decision(score),
            "attention_weights": dict(zip(["behavior", "graph", "twin"], [round(float(v), 3) for v in weights[0]])),
        }

    def explain(self, behavior_score: float, graph_score: float, twin_score: float):
        x = np.array([[behavior_score, graph_score, twin_score]], dtype=np.float32) / 100.0

        def predict(arr):
            with torch.no_grad():
                y, _ = self.fusion_model(torch.tensor(arr, dtype=torch.float32))
            return y.numpy()

        background = np.array([[0.92, 0.88, 0.9], [0.72, 0.65, 0.8], [0.45, 0.55, 0.6], [0.25, 0.35, 0.4]], dtype=np.float32)
        explainer = shap.KernelExplainer(predict, background)
        values = np.array(explainer.shap_values(x, nsamples=32)).reshape(-1)
        names = ["Behavior Intelligence", "Identity Graph", "Digital Twin"]
        ranked = sorted(zip(names, values, [behavior_score, graph_score, twin_score]), key=lambda item: abs(item[1]), reverse=True)
        explanation = []
        for name, val, score in ranked:
            direction = "increased" if val > 0 else "reduced"
            explanation.append(f"{name} {direction} trust because its input score was {score:.1f}.")
        fusion = self.fusion(behavior_score, graph_score, twin_score)
        return {"shap_values": dict(zip(names, [round(float(v), 3) for v in values])), "summary": " ".join(explanation), **fusion}

    def full_assessment(self):
        b = self.behavior()
        g = self.graph()
        t = self.twin()
        f = self.fusion(b["behavior_score"], g["graph_score"], t["twin_score"])
        e = self.explain(b["behavior_score"], g["graph_score"], t["twin_score"])
        return {"behavior": b, "graph": g, "twin": t, "fusion": f, "explainability": e}
