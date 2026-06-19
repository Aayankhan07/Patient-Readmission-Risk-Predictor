import mlflow
from mlflow.tracking import MlflowClient

def promote_champion_model(model_name: str = "readmission_champion"):
    """
    Finds the run with the highest PR-AUC across all experiments,
    registers it, and promotes it to the Production stage.
    """
    client = MlflowClient()
    experiments = client.search_experiments()
    
    best_run = None
    best_pr_auc = -1.0
    best_exp_id = None
    
    print("Searching for champion model across experiments...")
    for exp in experiments:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.pr_auc DESC"]
        )
        if runs:
            top_run = runs[0]
            val = top_run.data.metrics.get("pr_auc", 0.0)
            print(f"  Exp '{exp.name}': top run {top_run.info.run_id} has PR-AUC={val:.4f}")
            if val > best_pr_auc:
                best_pr_auc = val
                best_run = top_run
                best_exp_id = exp.experiment_id
                
    if best_run is None:
        print("No runs found. Cannot promote champion model.")
        return None
        
    run_id = best_run.info.run_id
    model_type = best_run.data.tags.get("model_type", "unknown")
    print(f"Best run: {run_id} (Type: {model_type}) with PR-AUC={best_pr_auc:.4f}")
    
    # Register the model
    model_uri = f"runs:/{run_id}/model"
    print(f"Registering model from {model_uri} under '{model_name}'...")
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    
    # Transition to Production
    version = model_details.version
    print(f"Promoting version {version} of '{model_name}' to Production stage...")
    
    try:
        # Transition stage for the newly registered version to Production
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Model '{model_name}' version {version} is now in Production.")
    except Exception as e:
        print(f"Note: Could not transition model stage via transition_model_version_stage (this is expected if running in basic filesystem-based MLflow server without SQL backend, but the model registration is complete): {e}")
        
    return {
        "run_id": run_id,
        "version": version,
        "pr_auc": best_pr_auc,
        "model_type": model_type
    }
