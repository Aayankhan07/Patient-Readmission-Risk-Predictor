import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

print("=== Registered Models ===")
try:
    models = client.search_registered_models()
    for m in models:
        print(f"Model: {m.name}")
        for v in m.latest_versions:
            print(
                f"  Version: {v.version}, Run ID: {v.run_id}, Source: {v.source}, Status: {v.status}"
            )
except Exception as e:
    print(f"Error: {e}")

print("\n=== Active Experiments ===")
try:
    exps = client.search_experiments()
    for exp in exps:
        print(f"Experiment: {exp.name} (ID: {exp.experiment_id})")
        runs = client.search_runs(exp.experiment_id)
        for r in runs:
            print(
                f"  Run ID: {r.info.run_id}, Name: {r.info.run_name}, Status: {r.info.status}, Artifact URI: {r.info.artifact_uri}"
            )
except Exception as e:
    print(f"Error: {e}")
