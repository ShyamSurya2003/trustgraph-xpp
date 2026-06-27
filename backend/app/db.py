import os
from datetime import datetime, timezone

try:
    from motor.motor_asyncio import AsyncIOMotorClient
except Exception:  # pragma: no cover
    AsyncIOMotorClient = None


client = None
db = None


def init_db():
    global client, db
    uri = os.getenv("MONGODB_URI")
    if uri and AsyncIOMotorClient:
        client = AsyncIOMotorClient(uri)
        db = client[os.getenv("MONGODB_DB", "trustgraph_xpp")]


async def log_assessment(payload: dict):
    if db is None:
        return
    doc = {
        "created_at": datetime.now(timezone.utc),
        "decision": payload.get("fusion", {}).get("decision"),
        "final_trust_score": payload.get("fusion", {}).get("final_trust_score"),
        "payload": payload,
    }
    await db.assessments.insert_one(doc)
