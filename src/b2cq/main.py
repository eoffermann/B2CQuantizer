"""FastAPI app entry."""
from fastapi import FastAPI
from b2cq.web.routes import router

app = FastAPI(title="B2CQuantizer", version="0.1.0")
app.include_router(router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
