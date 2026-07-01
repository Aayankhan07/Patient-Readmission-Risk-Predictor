import os
import joblib
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

from src.utils.config import PREPROCESSOR_PATH, MODEL_DIR
from src.data.preprocessor import ReadmissionPreprocessor, preprocess_raw_dataset
from src.data.loader import generate_synthetic_data

class ModelState:
    """Holds the loaded champion model, preprocessor, and background dataset for the API's lifetime."""

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.X_background = None
        self.model_type = "unknown"
        self.model_run_id = "dummy"
        self.model_version = "unloaded"
        self.loaded = False
        self.predictions_store = {}

    def load(self):
        """Loads preprocessor, background dataset, and champion model, training fallbacks if needed."""
        # 1. Load Preprocessor
        if os.path.exists(PREPROCESSOR_PATH):
            print(f"Loading fitted preprocessor from {PREPROCESSOR_PATH}...")
            self.preprocessor = joblib.load(PREPROCESSOR_PATH)
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
            self.preprocessor = preprocessor
            
        # 2. Setup Background Dataset
        processed_train_path = os.path.join(os.path.dirname(PREPROCESSOR_PATH).replace("models", "data/processed"), "train_processed.csv")
        if os.path.exists(processed_train_path):
            print(f"Loading background dataset from {processed_train_path}...")
            train_df = pd.read_csv(processed_train_path)
            if "readmitted" in train_df.columns:
                train_df = train_df.drop(columns=["readmitted"])
            self.X_background = train_df
        else:
            print("Processed training split not found. Generating synthetic background dataset...")
            # Generate synthetic preprocessed features
            dummy_feats = self.preprocessor.feature_columns_
            dummy_data = np.random.normal(0, 1, size=(100, len(dummy_feats)))
            self.X_background = pd.DataFrame(dummy_data, columns=dummy_feats)

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
                    self.model = mlflow.tensorflow.load_model(model_uri)
                else:
                    self.model = mlflow.sklearn.load_model(model_uri)
                    
                self.model_type = model_type
                self.model_run_id = run_id
                self.model_version = f"{model_type}_run_{run_id[:8]}"
                self.loaded = True
                
        except Exception as e:
            print(f"Could not load champion model from MLflow registry: {e}")
            print("Attempting to search latest local runs as fallback...")
            
        # If registry loading failed, look for any completed MLflow runs in local file storage
        if self.model is None:
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
                        self.model = mlflow.tensorflow.load_model(model_uri)
                    else:
                        self.model = mlflow.sklearn.load_model(model_uri)
                        
                    self.model_type = model_type
                    self.model_run_id = run_id
                    self.model_version = f"{model_type}_run_{run_id[:8]}"
                    self.loaded = True
            except Exception as local_err:
                print(f"Could not load model from local runs: {local_err}")

        # 4. Final Fallback: Train a dummy LogisticRegression so the server is functional
        if self.model is None:
            print("No models found in MLflow. Training and saving a dummy LogisticRegression model...")
            from sklearn.linear_model import LogisticRegression
            
            # Train on synthetic data matching background size
            n_samples = len(self.X_background)
            dummy_target = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
            if len(np.unique(dummy_target)) < 2 and n_samples >= 2:
                dummy_target[0] = 0
                dummy_target[1] = 1
                
            dummy_model = LogisticRegression(random_state=42)
            dummy_model.fit(self.X_background.values, dummy_target)
            
            self.model = dummy_model
            self.model_type = "logistic_regression"
            self.model_run_id = "dummy_fallback"
            self.model_version = "logistic_regression_run_dummy_fallback"
            self.loaded = True
            print("Dummy fallback model trained and loaded.")

    def predict_proba(self, df_preprocessed) -> float:
        """Runs inference on the preprocessed DataFrame and returns the class 1 probability."""
        if self.model_type == "ann":
            vals = df_preprocessed.values if isinstance(df_preprocessed, pd.DataFrame) else df_preprocessed
            pred_arr = self.model.predict(vals.astype(np.float32), verbose=0)
            return float(pred_arr[0][0])
        else:
            if hasattr(self.model, "predict_proba"):
                pred_proba = self.model.predict_proba(df_preprocessed)
                if len(pred_proba.shape) > 1 and pred_proba.shape[1] > 1:
                    return float(pred_proba[:, 1][0])
                else:
                    return float(pred_proba[0])
            else:
                return float(self.model.predict(df_preprocessed)[0])

model_state = ModelState()
