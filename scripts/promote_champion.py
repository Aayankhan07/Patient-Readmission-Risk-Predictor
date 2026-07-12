import os
import mlflow
from src.models.registry import promote_champion_model

if __name__ == "__main__":
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if tracking_uri.startswith("http"):
        import requests

        try:
            requests.get(tracking_uri, timeout=2.0)
            mlflow.set_tracking_uri(tracking_uri)
            print(f"Connected to MLflow tracking server at {tracking_uri}")
        except Exception:
            print(
                f"WARNING: MLflow tracking server at {tracking_uri} is unreachable. Falling back to local './mlruns' directory."
            )
            mlflow.set_tracking_uri("mlruns")
    else:
        mlflow.set_tracking_uri(tracking_uri)

    print("Starting champion model promotion...")
    promote_champion_model()
    print("Champion model promotion completed.")
