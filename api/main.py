import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import joblib
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.config import PREPROCESSOR_PATH, MODEL_DIR
from src.data.preprocessor import ReadmissionPreprocessor, preprocess_raw_dataset
from src.data.loader import generate_synthetic_data

# Import routes
from api.routes import health, predict, explain

def load_champion_model_and_preprocessor(app: FastAPI):
    """
    Loads preprocessor and champion model from MLflow registry or local runs.
    Falls back to a dummy model if none exists yet to prevent server crashes.
    """
    # 1. Load Preprocessor
    if os.path.exists(PREPROCESSOR_PATH):
        print(f"Loading fitted preprocessor from {PREPROCESSOR_PATH}...")
        app.state.preprocessor = joblib.load(PREPROCESSOR_PATH)
    else:
        print(f"Fitted preprocessor not found at {PREPROCESSOR_PATH}. Initializing and fitting dummy preprocessor...")
        # Create dummy data and fit preprocessor
        dummy_raw = generate_synthetic_data("data/raw/diabetic_data.csv", num_samples=100)
        dummy_clean = preprocess_raw_dataset(dummy_raw)
        X_dummy = dummy_clean.drop(columns=["readmitted"])
        
        preprocessor = ReadmissionPreprocessor()
        preprocessor.fit(X_dummy)
        
        os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        app.state.preprocessor = preprocessor
        
    # 2. Setup Background Dataset
    processed_train_path = os.path.join(os.path.dirname(PREPROCESSOR_PATH).replace("models", "data/processed"), "train_processed.csv")
    if os.path.exists(processed_train_path):
        print(f"Loading background dataset from {processed_train_path}...")
        train_df = pd.read_csv(processed_train_path)
        if "readmitted" in train_df.columns:
            train_df = train_df.drop(columns=["readmitted"])
        app.state.X_background = train_df
    else:
        print("Processed training split not found. Generating synthetic background dataset...")
        # Generate synthetic preprocessed features
        dummy_feats = app.state.preprocessor.feature_columns_
        dummy_data = np.random.normal(0, 1, size=(100, len(dummy_feats)))
        app.state.X_background = pd.DataFrame(dummy_data, columns=dummy_feats)

    # 3. Load Model from MLflow
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if tracking_uri.startswith("http"):
        import requests
        try:
            requests.get(tracking_uri, timeout=1.0)
            mlflow.set_tracking_uri(tracking_uri)
            print(f"Connected to MLflow tracking server at {tracking_uri}")
        except Exception:
            print(f"MLflow tracking server at {tracking_uri} is unreachable. Falling back to local 'mlruns' directory.")
            mlflow.set_tracking_uri("mlruns")
    else:
        mlflow.set_tracking_uri(tracking_uri)
        
    client = MlflowClient()
    
    app.state.model = None
    app.state.model_type = "unknown"
    app.state.model_run_id = "dummy"
    
    try:
        # Try loading from model registry
        prod_versions = client.get_latest_versions("readmission_champion", stages=["Production"])
        if prod_versions:
            latest_version = prod_versions[0]
            run_id = latest_version.run_id
            run_data = client.get_run(run_id)
            model_type = run_data.data.tags.get("model_type", "unknown")
            
            print(f"Loading champion model from registry. Run ID: {run_id}, Type: {model_type}")
            
            model_uri = f"models:/readmission_champion/Production"
            if model_type == "ann":
                app.state.model = mlflow.tensorflow.load_model(model_uri)
            else:
                app.state.model = mlflow.sklearn.load_model(model_uri)
                
            app.state.model_type = model_type
            app.state.model_run_id = run_id
            
    except Exception as e:
        print(f"Could not load champion model from MLflow registry: {e}")
        print("Attempting to search latest local runs as fallback...")
        
    # If registry loading failed, look for any completed MLflow runs in local file storage
    if app.state.model is None:
        try:
            experiments = client.search_experiments()
            best_run = None
            best_pr_auc = -1.0
            
            for exp in experiments:
                runs = client.search_runs(exp.experiment_id, order_by=["metrics.pr_auc DESC"])
                if runs:
                    top_run = runs[0]
                    pr_auc = top_run.data.metrics.get("pr_auc", 0.0)
                    if pr_auc > best_pr_auc:
                        best_pr_auc = pr_auc
                        best_run = top_run
            
            if best_run:
                run_id = best_run.info.run_id
                model_type = best_run.data.tags.get("model_type", "unknown")
                print(f"Loading best local run {run_id} (Type: {model_type}, PR-AUC: {best_pr_auc:.4f})")
                
                model_uri = f"runs:/{run_id}/model"
                if model_type == "ann":
                    app.state.model = mlflow.tensorflow.load_model(model_uri)
                else:
                    app.state.model = mlflow.sklearn.load_model(model_uri)
                    
                app.state.model_type = model_type
                app.state.model_run_id = run_id
        except Exception as local_err:
            print(f"Could not load model from local runs: {local_err}")

    # 4. Final Fallback: Train a dummy LogisticRegression so the server is functional
    if app.state.model is None:
        print("No models found in MLflow. Training and saving a dummy LogisticRegression model...")
        from sklearn.linear_model import LogisticRegression
        
        # Train on synthetic data
        dummy_target = np.random.choice([0, 1], size=100, p=[0.8, 0.2])
        dummy_model = LogisticRegression(random_state=42)
        dummy_model.fit(app.state.X_background.values, dummy_target)
        
        # Save as app state model
        app.state.model = dummy_model
        app.state.model_type = "logistic_regression"
        app.state.model_run_id = "dummy_fallback"
        print("Dummy fallback model trained and loaded.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize cache/store
    app.state.predictions_store = {}
    # Load model and preprocessor
    load_champion_model_and_preprocessor(app)
    yield
    # Clean up on shutdown
    app.state.predictions_store.clear()

app = FastAPI(
    title="Patient Readmission Risk Predictor (PRRP) API",
    description="Provides real-time readmission risk scoring, SHAP waterfall parameters, and LIME explanations.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for stream/dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, tags=["Inference"])
app.include_router(explain.router, tags=["Explainability"])
