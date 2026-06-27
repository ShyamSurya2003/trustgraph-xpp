from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

for path in (RAW_DIR, CHECKPOINT_DIR):
    path.mkdir(parents=True, exist_ok=True)

SEED = 42
BEHAVIOR_FEATURES = [
    "transaction_amount",
    "account_age_days",
    "login_velocity",
    "device_trust",
    "geo_distance_km",
    "merchant_risk",
    "hour_sin",
    "hour_cos",
]

TWIN_FEATURES = ["amount", "oldbalance_org", "newbalance_orig", "oldbalance_dest", "newbalance_dest"]
