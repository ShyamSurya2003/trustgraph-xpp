import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import BEHAVIOR_FEATURES, CHECKPOINT_DIR, SEED, TWIN_FEATURES
from .data import load_behavior_frame, load_graph_data, load_twin_sequences
from .models import AttentionFusionMLP, GATRiskModel, LSTMAutoencoder, TabTransformerLite

torch.manual_seed(SEED)
np.random.seed(SEED)


def _save(obj, name):
    torch.save(obj, CHECKPOINT_DIR / name)


def _load(name):
    return torch.load(CHECKPOINT_DIR / name, map_location="cpu", weights_only=False)


def ensure_behavior_model():
    path = CHECKPOINT_DIR / "behavior.pt"
    if path.exists():
        bundle = _load("behavior.pt")
        model = TabTransformerLite(len(BEHAVIOR_FEATURES))
        model.load_state_dict(bundle["state_dict"])
        return model.eval(), bundle["scaler"]

    df = load_behavior_frame()
    scaler = StandardScaler()
    x = scaler.fit_transform(df[BEHAVIOR_FEATURES]).astype("float32")
    y = df["is_fraud"].to_numpy(dtype="float32")
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=SEED, stratify=y if len(np.unique(y)) > 1 else None)
    model = TabTransformerLite(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xb = torch.tensor(x_train)
    yb = torch.tensor(y_train)
    for _ in range(10):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(xb), yb)
        loss.backward()
        opt.step()
    _save({"state_dict": model.state_dict(), "scaler": scaler}, "behavior.pt")
    return model.eval(), scaler


def ensure_graph_model():
    path = CHECKPOINT_DIR / "graph.pt"
    x_np, edge_np, y_np = load_graph_data()
    if path.exists():
        bundle = _load("graph.pt")
        model = GATRiskModel(x_np.shape[1])
        model.load_state_dict(bundle["state_dict"])
        return model.eval(), torch.tensor(x_np), torch.tensor(edge_np), torch.tensor(y_np)
    model = GATRiskModel(x_np.shape[1])
    x = torch.tensor(x_np)
    edge = torch.tensor(edge_np)
    y = torch.tensor(y_np)
    opt = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=1e-4)
    for _ in range(35):
        opt.zero_grad()
        loss = F.cross_entropy(model(x, edge), y)
        loss.backward()
        opt.step()
    _save({"state_dict": model.state_dict()}, "graph.pt")
    return model.eval(), x, edge, y


def ensure_twin_model():
    path = CHECKPOINT_DIR / "twin.pt"
    seqs, scaler = load_twin_sequences()
    if path.exists():
        bundle = _load("twin.pt")
        model = LSTMAutoencoder(len(TWIN_FEATURES))
        model.load_state_dict(bundle["state_dict"])
        return model.eval(), bundle["scaler"], bundle["threshold"]
    model = LSTMAutoencoder(len(TWIN_FEATURES))
    opt = torch.optim.AdamW(model.parameters(), lr=0.003)
    x = torch.tensor(seqs)
    for _ in range(18):
        opt.zero_grad()
        recon = model(x)
        loss = F.mse_loss(recon, x)
        loss.backward()
        opt.step()
    with torch.no_grad():
        errors = ((model(x) - x) ** 2).mean(dim=(1, 2)).numpy()
    threshold = float(np.percentile(errors, 95))
    _save({"state_dict": model.state_dict(), "scaler": scaler, "threshold": threshold}, "twin.pt")
    return model.eval(), scaler, threshold


def ensure_fusion_model():
    path = CHECKPOINT_DIR / "fusion.pt"
    if path.exists():
        bundle = _load("fusion.pt")
        model = AttentionFusionMLP()
        model.load_state_dict(bundle["state_dict"])
        return model.eval()
    rng = np.random.default_rng(SEED)
    scores = rng.uniform(5, 100, (1800, 3)).astype("float32")
    target = (0.45 * scores[:, 0] + 0.3 * scores[:, 1] + 0.25 * scores[:, 2] - 18 * (scores.min(axis=1) < 35)).clip(0, 100)
    model = AttentionFusionMLP()
    opt = torch.optim.AdamW(model.parameters(), lr=0.004)
    x = torch.tensor(scores / 100.0)
    y = torch.tensor(target, dtype=torch.float32)
    for _ in range(80):
        opt.zero_grad()
        pred, _ = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()
    _save({"state_dict": model.state_dict()}, "fusion.pt")
    return model.eval()


def ensure_all():
    ensure_behavior_model()
    ensure_graph_model()
    ensure_twin_model()
    ensure_fusion_model()
