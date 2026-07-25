"""FastAPI app entry — will be expanded in later tasks."""
from fastapi import FastAPI

app = FastAPI(title="B2CQuantizer", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
