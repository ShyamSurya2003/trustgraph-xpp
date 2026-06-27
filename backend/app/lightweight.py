import math
import numpy as np


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


class LightweightTrustGraphService:
    """Memory-safe public inference service for Render free instances."""

    def behavior(self, payload=None):
        payload = payload or {
            "transaction_amount": 87.73,
            "account_age_days": 497.03,
            "login_velocity": 1.0,
            "device_trust": 78.08,
            "geo_distance_km": 125.59,
            "merchant_risk": 31.05,
            "hour_sin": -0.2588,
            "hour_cos": -0.9659,
        }
        z = (
            0.018 * (payload["transaction_amount"] - 120)
            + 0.48 * (payload["login_velocity"] - 2)
            - 0.034 * (payload["device_trust"] - 70)
            + 0.006 * (payload["geo_distance_km"] - 100)
            + 0.03 * (payload["merchant_risk"] - 30)
            - 0.001 * payload["account_age_days"]
        )
        fraud = 1 / (1 + math.exp(-z / 4))
        return {"behavior_score": round((1 - fraud) * 100, 2), "fraud_probability": round(fraud, 4), "features": payload}

    def graph(self, node_id=0):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, (420, 8))
        risk = 1 / (1 + np.exp(-(1.2 * x[:, 0] + 0.8 * x[:, 2] - 0.7 * x[:, 5])))
        node_id = int(node_id or 0) % len(risk)
        suspicious = np.argsort(risk)[-10:][::-1].tolist()
        edges = []
        for i in range(35):
            for j in (int((i + 7) % 35), int((i + 13) % 35)):
                edges.append({"source": str(i), "target": str(j)})
        return {
            "node_id": node_id,
            "graph_risk_score": round(float(risk[node_id] * 100), 2),
            "graph_score": round(float((1 - risk[node_id]) * 100), 2),
            "suspicious_nodes": [{"id": int(i), "risk": round(float(risk[i] * 100), 2)} for i in suspicious],
            "graph": {"nodes": [{"id": str(i), "risk": round(float(risk[i] * 100), 1)} for i in range(35)], "edges": edges},
        }

    def twin(self, sequence=None):
        error = 0.544421
        deviation = 18.08
        return {"reconstruction_error": error, "identity_deviation_score": deviation, "twin_score": round(100 - deviation, 2)}

    def fusion(self, behavior_score, graph_score, twin_score):
        scores = np.array([behavior_score, graph_score, twin_score], dtype=float)
        logits = np.array([0.7, 2.15, 0.9]) * (100 - scores) / 100
        weights = np.exp(logits) / np.exp(logits).sum()
        trust = float((scores * np.array([0.42, 0.34, 0.24])).sum() - (12 if scores.min() < 50 else 0))
        trust = round(float(np.clip(trust, 0, 100)), 2)
        return {
            "final_trust_score": trust,
            "decision": decision(trust),
            "attention_weights": dict(zip(["behavior", "graph", "twin"], [round(float(v), 3) for v in weights])),
        }

    def explain(self, behavior_score, graph_score, twin_score):
        vals = {
            "Behavior Intelligence": round((behavior_score - 70) * 0.28, 3),
            "Identity Graph": round((graph_score - 70) * 0.36, 3),
            "Digital Twin": round((twin_score - 70) * 0.22, 3),
        }
        ranked = sorted(vals.items(), key=lambda x: abs(x[1]), reverse=True)
        score_map = {"Behavior Intelligence": behavior_score, "Identity Graph": graph_score, "Digital Twin": twin_score}
        summary = " ".join(f"{name} {'increased' if v > 0 else 'reduced'} trust because its input score was {score_map[name]:.1f}." for name, v in ranked)
        return {"shap_values": vals, "summary": summary, **self.fusion(behavior_score, graph_score, twin_score)}

    def full_assessment(self):
        b = self.behavior()
        g = self.graph()
        t = self.twin()
        f = self.fusion(b["behavior_score"], g["graph_score"], t["twin_score"])
        e = self.explain(b["behavior_score"], g["graph_score"], t["twin_score"])
        return {"behavior": b, "graph": g, "twin": t, "fusion": f, "explainability": e}
