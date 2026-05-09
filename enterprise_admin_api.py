from pathlib import Path

from fastapi import FastAPI

from backend.enterprise.ingestion import ingest_documents
from backend.enterprise.ops import (
    get_ab_test_config,
    get_hotwords,
    list_documents,
    update_ab_test_config,
)

app = FastAPI(title="Enterprise Doc Assistant Admin API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/documents")
def documents() -> dict:
    return {"items": list_documents()}


@app.post("/ingest")
def ingest() -> dict:
    result = ingest_documents(Path("enterprise_data/docs"))
    return result


@app.get("/hotwords")
def hotwords(top_n: int = 20) -> dict:
    return {"items": get_hotwords(top_n=top_n)}


@app.get("/ab-test")
def ab_test_get() -> dict:
    return get_ab_test_config()


@app.post("/ab-test")
def ab_test_set(config: dict) -> dict:
    update_ab_test_config(config)
    return {"ok": True}
