import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.model_loader import model_state
from api.routes import health, predict, explain


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and preprocessor inside model_state singleton
    model_state.load()
    yield
    # Clean up on shutdown
    model_state.predictions_store.clear()


app = FastAPI(
    title="Patient Readmission Risk Predictor (PRRP) API",
    description="Provides real-time readmission risk scoring, SHAP waterfall parameters, and LIME explanations.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for Streamlit/dashboard access (configurable for production)
allowed_origins_raw = os.environ.get("ALLOWED_ORIGINS", "*")
allowed_origins = (
    [origin.strip() for origin in allowed_origins_raw.split(",") if origin.strip()]
    if allowed_origins_raw != "*"
    else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, tags=["Inference"])
app.include_router(explain.router, tags=["Explainability"])
