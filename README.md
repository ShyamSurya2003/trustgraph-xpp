# TrustGraph-X++

AI-powered Continuous Identity Trust Framework for Banking.

This proof-of-concept prioritizes working model inference over UI decoration. The FastAPI backend trains compact PyTorch models automatically when checkpoints are missing, saves them in `backend/checkpoints`, and serves real predictions to the React dashboard.

## Implemented Modules

- Behavioral Intelligence: TabTransformer-style PyTorch model over IEEE-CIS-compatible features.
- Identity Graph Intelligence: Graph Attention Network using PyTorch Geometric when available, with a torch fallback if PyG is unavailable.
- Identity Digital Twin: LSTM autoencoder trained only on normal PaySim-style transactions.
- Trust Fusion: attention-based MLP that fuses behavior, graph, and twin scores.
- Explainable AI: SHAP values over the fusion model plus concise natural-language explanations.

Future deployment modules are shown in the dashboard architecture only:

- Federated Learning
- Zero Trust Policy Engine
- Adaptive Authentication

## Dataset Placement

The prototype works immediately with deterministic synthetic fallback data. To use real datasets, place files here:

- IEEE-CIS: `backend/data/raw/ieee_cis_transaction.csv` and optionally `backend/data/raw/ieee_cis_identity.csv`
- PaySim: `backend/data/raw/paysim.csv`
- Elliptic: the current PoC uses an Elliptic-style generated graph adapter; replace `load_graph_data` in `backend/app/data.py` with your parsed Elliptic CSV tensors for full dataset training.

Delete files in `backend/checkpoints` after changing datasets so models retrain.

## Backend Setup

```powershell
cd "C:\Users\Shyam Surya\OneDrive\Documents\New project\trustgraph-xpp\backend"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The first startup trains and saves checkpoints if they do not exist.

Useful endpoints:

- `GET /health`
- `GET /api/assessment`
- `POST /api/behavior`
- `GET /api/graph?node_id=0`
- `POST /api/twin`
- `POST /api/fusion`
- `POST /api/explain`

## Frontend Setup

```powershell
cd "C:\Users\Shyam Surya\OneDrive\Documents\New project\trustgraph-xpp\frontend"
npm install
npm run dev
```

Open the Vite URL, usually `http://127.0.0.1:5173`.

If the backend is on another address:

```powershell
$env:VITE_API_URL="http://127.0.0.1:8000"
npm run dev
```

## Architecture Flow

```text
Dataset Collection
↓
Preprocessing
↓
Behavior Model
↓
Graph Model
↓
Digital Twin
↓
Trust Fusion
↓
Trust Score
↓
Explainable AI
↓
Decision
```

## Notes

This is a proof-of-concept, not a production banking risk engine. The synthetic fallback data makes the project runnable without restricted competition datasets; production use requires licensed datasets, validation, monitoring, bias analysis, and secure deployment controls.

See `DEPLOYMENT.md` for Vercel, Render, and MongoDB Atlas setup.
