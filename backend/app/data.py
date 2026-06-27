import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import BEHAVIOR_FEATURES, RAW_DIR, SEED, TWIN_FEATURES


def _rng():
    return np.random.default_rng(SEED)


def load_behavior_frame(n: int = 2400) -> pd.DataFrame:
    """IEEE-CIS adapter with deterministic synthetic fallback."""
    tx = RAW_DIR / "ieee_cis_transaction.csv"
    identity = RAW_DIR / "ieee_cis_identity.csv"
    if tx.exists():
        df = pd.read_csv(tx)
        if identity.exists() and "TransactionID" in df:
            df = df.merge(pd.read_csv(identity), on="TransactionID", how="left")
        amount = df.get("TransactionAmt", pd.Series(np.zeros(len(df)))).fillna(0).to_numpy()
        hour = (df.get("TransactionDT", pd.Series(np.arange(len(df)))).fillna(0).to_numpy() // 3600) % 24
        out = pd.DataFrame(
            {
                "transaction_amount": amount,
                "account_age_days": np.nan_to_num(df.get("card1", pd.Series(1000, index=df.index)).to_numpy()) % 3650,
                "login_velocity": np.nan_to_num(df.get("C1", pd.Series(1, index=df.index)).to_numpy()),
                "device_trust": np.nan_to_num(df.get("D1", pd.Series(90, index=df.index)).to_numpy()) % 100,
                "geo_distance_km": np.nan_to_num(df.get("dist1", pd.Series(10, index=df.index)).to_numpy()),
                "merchant_risk": np.nan_to_num(df.get("addr1", pd.Series(100, index=df.index)).to_numpy()) % 100,
                "hour_sin": np.sin(2 * np.pi * hour / 24),
                "hour_cos": np.cos(2 * np.pi * hour / 24),
                "is_fraud": df.get("isFraud", pd.Series(0, index=df.index)).astype(int),
            }
        )
        return out.sample(min(n, len(out)), random_state=SEED).reset_index(drop=True)

    rng = _rng()
    amount = rng.lognormal(4.2, 0.9, n)
    account_age = rng.gamma(5, 180, n)
    login_velocity = rng.poisson(2, n)
    device_trust = rng.beta(8, 2, n) * 100
    geo_distance = rng.exponential(120, n)
    merchant_risk = rng.beta(2, 7, n) * 100
    hour = rng.integers(0, 24, n)
    logit = (
        0.018 * (amount - 120)
        + 0.5 * (login_velocity - 2)
        - 0.035 * (device_trust - 70)
        + 0.006 * (geo_distance - 100)
        + 0.03 * (merchant_risk - 30)
        - 0.001 * account_age
    )
    prob = 1 / (1 + np.exp(-logit / 4))
    fraud = (rng.random(n) < prob).astype(int)
    return pd.DataFrame(
        {
            "transaction_amount": amount,
            "account_age_days": account_age,
            "login_velocity": login_velocity,
            "device_trust": device_trust,
            "geo_distance_km": geo_distance,
            "merchant_risk": merchant_risk,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "is_fraud": fraud,
        }
    )


def load_graph_data(num_nodes: int = 420):
    """Elliptic-style graph adapter with synthetic transaction graph fallback."""
    rng = _rng()
    x = rng.normal(0, 1, (num_nodes, 8)).astype("float32")
    risk_latent = 1.2 * x[:, 0] + 0.8 * x[:, 2] - 0.7 * x[:, 5] + rng.normal(0, 0.4, num_nodes)
    y = (risk_latent > np.quantile(risk_latent, 0.78)).astype("int64")
    edges = []
    for i in range(num_nodes):
        for _ in range(3):
            j = int((i + rng.integers(1, 35) + (18 if y[i] else 0)) % num_nodes)
            edges.append((i, j))
            edges.append((j, i))
    return x, np.array(edges, dtype="int64").T, y


def load_twin_sequences(num_identities: int = 220, seq_len: int = 12):
    """PaySim adapter with normal-only synthetic fallback for LSTM autoencoder."""
    csv = RAW_DIR / "paysim.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        normal = df[df.get("isFraud", 0) == 0].copy()
        for col in TWIN_FEATURES:
            if col not in normal:
                normal[col] = 0.0
        arr = normal[TWIN_FEATURES].fillna(0).to_numpy(dtype="float32")
    else:
        rng = _rng()
        arr = np.column_stack(
            [
                rng.lognormal(3.6, 0.45, num_identities * seq_len),
                rng.lognormal(7.2, 0.35, num_identities * seq_len),
                rng.lognormal(7.1, 0.35, num_identities * seq_len),
                rng.lognormal(6.7, 0.45, num_identities * seq_len),
                rng.lognormal(6.75, 0.45, num_identities * seq_len),
            ]
        ).astype("float32")

    usable = (len(arr) // seq_len) * seq_len
    sequences = arr[:usable].reshape(-1, seq_len, len(TWIN_FEATURES))
    scaler = StandardScaler()
    flat = scaler.fit_transform(sequences.reshape(-1, len(TWIN_FEATURES)))
    return flat.reshape(sequences.shape).astype("float32"), scaler


def default_behavior_payload() -> dict:
    row = load_behavior_frame(1).iloc[0]
    return {k: float(row[k]) for k in BEHAVIOR_FEATURES}


def default_twin_sequence() -> list[list[float]]:
    sequences, scaler = load_twin_sequences(1, 12)
    raw = scaler.inverse_transform(sequences[0])
    return raw.round(2).tolist()
