"""
Module C1 + C3 - Unified Pipeline & API Layer
FastAPI entrypoint. All models (face recognition, product classifier,
sentiment model, chatbot) are loaded ONCE at startup via the service-layer
imports below (see app/services/*.py), then reused across every request.

Run locally:
    uvicorn app.main:app --reload --port 8000
Then open:
    http://localhost:8000/docs   (Swagger UI, auto-generated)
"""

import os

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from app.routers import chatbot, nlp, vision
from app.schemas import DashboardStatsResponse
from app.services import cv_service, nlp_service

API_KEY = os.getenv("API_KEY", "dev-demo-key-change-me")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str = Security(api_key_header)):
    """
    Module C4 - simulate production security with a simple API key header.
    Pass `X-API-Key: <your key>` on every request.
    """
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


app = FastAPI(
    title="Smart Retail & Customer Intelligence Platform API",
    description=(
        "Unified API for the AI-Powered Smart Retail & Customer Intelligence "
        "Platform: face recognition, product image classification, sentiment "
        "analysis, and an FAQ chatbot, all behind one FastAPI gateway."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# All module routers are mounted under one gateway and protected by the API key
app.include_router(vision.router, dependencies=[Depends(require_api_key)])
app.include_router(nlp.router, dependencies=[Depends(require_api_key)])
app.include_router(chatbot.router, dependencies=[Depends(require_api_key)])


@app.get("/", tags=["meta"])
async def root():
    return {
        "message": "Smart Retail & Customer Intelligence Platform API is running.",
        "docs": "/docs",
        "endpoints": [
            "POST /recognize-face",
            "POST /classify-product",
            "POST /analyze-sentiment",
            "POST /chatbot",
            "GET /dashboard/stats",
        ],
    }


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok"}


@app.get(
    "/dashboard/stats",
    response_model=DashboardStatsResponse,
    tags=["dashboard"],
    dependencies=[Depends(require_api_key)],
)
async def dashboard_stats():
    """Module C3 - aggregate visit/sentiment stats for a simple frontend chart."""
    return {
        "visits": cv_service.get_visit_stats(),
        "sentiment": nlp_service.get_sentiment_stats(),
    }
