"""
PlacementAI - FastAPI Application
AI-powered Placement Risk Modeling System for Indian education lenders.
"""

import os, sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import predict, portfolio, fraud

app = FastAPI(
    title="PlacementAI API",
    description="AI-powered Placement Risk Modeling for Education Lenders. "
                "Predicts student employability risk and generates RBI-compliant risk intelligence.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(predict.router, tags=["Predictions"])
app.include_router(portfolio.router, tags=["Portfolio"])
app.include_router(fraud.router, tags=["Fraud Detection"])


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "PlacementAI",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}
