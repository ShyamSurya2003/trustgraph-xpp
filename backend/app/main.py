from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import BEHAVIOR_FEATURES, TWIN_FEATURES
from .db import init_db, log_assessment
from .services import TrustGraphService

service: TrustGraphService | None = None


class BehaviorRequest(BaseModel):
    transaction_amount: float = 149.0
    account_age_days: float = 640
    login_velocity: float = 2
    device_trust: float = 86
    geo_distance_km: float = 48
    merchant_risk: float = 22
    hour_sin: float = 0.5
    hour_cos: float = 0.866


class TwinRequest(BaseModel):
    sequence: list[list[float]] = Field(default_factory=list)


class FusionRequest(BaseModel):
    behavior_score: float
    graph_score: float
    twin_score: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    init_db()
    service = TrustGraphService()
    yield


app = FastAPI(title="TrustGraph-X++ API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def svc() -> TrustGraphService:
    if service is None:
        raise HTTPException(status_code=503, detail="Models are still loading")
    return service


@app.get("/health")
def health():
    return {"status": "ok", "models": ["TabTransformerLite", "GAT", "LSTM Autoencoder", "Attention Fusion MLP"]}


@app.post("/api/behavior")
def behavior(req: BehaviorRequest):
    return svc().behavior(req.model_dump())


@app.get("/api/graph")
def graph(node_id: int = 0):
    return svc().graph(node_id)


@app.post("/api/twin")
def twin(req: TwinRequest):
    try:
        return svc().twin(req.sequence or None)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/fusion")
def fusion(req: FusionRequest):
    return svc().fusion(req.behavior_score, req.graph_score, req.twin_score)


@app.post("/api/explain")
def explain(req: FusionRequest):
    return svc().explain(req.behavior_score, req.graph_score, req.twin_score)


@app.get("/api/assessment")
async def assessment():
    result = svc().full_assessment()
    await log_assessment(result)
    return result


@app.get("/api/schema")
def schemas():
    return {"behavior_features": BEHAVIOR_FEATURES, "twin_features": TWIN_FEATURES}
