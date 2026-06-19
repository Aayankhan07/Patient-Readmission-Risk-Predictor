import os
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
import mlflow.tensorflow

from src.data.loader import load_raw_data
from src.data.preprocessor import preprocess_raw_dataset, ReadmissionPreprocessor, apply_smote
from src.data.splitter import split_data
from src.utils.config import PARAMS, PREPROCESSOR_PATH, MODEL_DIR
from src.models.classical import get_logistic_regression, get_svm, get_random_forest, get_xgboost
from src.models.ann import build_ann_model
from src.models.tuner import tune_model
from src.models.registry import promote_champion_model
from src.evaluation.metrics import (
    compute_metrics,
    save_confusion_matrix_plot,
    save_roc_curve_plot,
    save_pr_curve_plot
)

# Set up MLflow tracking URI (default to local mlruns if not in environment)
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))

def train_classical_models(X_train, y_train, X_val, y_val, X_test, y_test, do_tune, trials):
    """Trains and logs the classical models to MLflow."""
    mlflow.set_experiment("classical_ml")
    models_to_train = ["logistic_regression", "random_forest", "xgboost"] # omitting SVM by default for speed, but can be added
    
    for model_name in models_to_train:
        print(f"\n==================== Training {model_name} ====================")
        
        # 1. Hyperparameter Tuning (Optional)
        params = {}
        if do_tune:
            params = tune_model(model_name, X_train.values, y_train.values, n_trials=trials)
            print(f"Tuned params for {model_name}: {params}")
        else:
            print(f"Using default parameters from params.yaml")
            
        # 2. Get Model Instance
        if model_name == "logistic_regression":
            model = get_logistic_regression(params)
        elif model_name == "random_forest":
            model = get_random_forest(params)
        elif model_name == "xgboost":
            # Handle scale_pos_weight
            neg, pos = np.bincount(y_train)
            scale_pos_weight = neg / pos if pos > 0 else 1.0
            model = get_xgboost(params, scale_pos_weight=scale_pos_weight)
        elif model_name == "svm":
            model = get_svm(params)
            
        # 3. Fit Model
        model.fit(X_train, y_train)
        
        # 4. Evaluate Model on Val & Test Set
        y_prob_val = model.predict_proba(X_val)
        y_prob_test = model.predict_proba(X_test)
        
        val_metrics = compute_metrics(y_val, y_prob_val)
        test_metrics = compute_metrics(y_test, y_prob_test)
        
        print(f"Validation metrics: {val_metrics}")
        print(f"Test metrics: {test_metrics}")
        
        # 5. Log to MLflow
        with mlflow.start_run(run_name=model_name) as run:
            # Set tags
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("framework", "scikit-learn" if model_name != "xgboost" else "xgboost")
            
            # Log params
            mlflow.log_params(model.get_params())
            mlflow.log_param("tuned", do_tune)
            
            # Log metrics (we care most about test metrics for final comparison)
            for k, v in test_metrics.items():
                mlflow.log_metric(k, v)
                
            # Log validation metrics prefixed
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v)
                
            # Save and log plots
            os.makedirs("temp_plots", exist_ok=True)
            cm_path = "temp_plots/confusion_matrix.png"
            roc_path = "temp_plots/roc_curve.png"
            pr_path = "temp_plots/pr_curve.png"
            
            save_confusion_matrix_plot(y_test, y_prob_test, cm_path)
            save_roc_curve_plot(y_test, y_prob_test, roc_path)
            save_pr_curve_plot(y_test, y_prob_test, pr_path)
            
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(roc_path)
            mlflow.log_artifact(pr_path)
            
            # Clean up temp plots
            for p in [cm_path, roc_path, pr_path]:
                if os.path.exists(p):
                    os.remove(p)
                    
            # Log Model
            mlflow.sklearn.log_model(model, "model")
            print(f"Successfully logged {model_name} to MLflow.")

def train_ann_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """Trains and logs the Keras ANN model to MLflow."""
    import tensorflow as tf
    mlflow.set_experiment("deep_learning")
    
    print("\n==================== Training Keras ANN ====================")
    input_dim = X_train.shape[1]
    
    # 1. Build and Compile Model
    model = build_ann_model(input_dim)
    
    # 2. Setup Keras Callbacks
    cfg = PARAMS["ann"]["training"]
    best_weights_path = os.path.join(MODEL_DIR, "best_ann.keras")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg["patience_early_stopping"],
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg["lr_factor"],
            patience=cfg["patience_lr_plateau"]
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_weights_path,
            monitor="val_loss",
            save_best_only=True
        )
    ]
    
    # 3. Fit Keras Model
    with mlflow.start_run(run_name="keras_ann") as run:
        mlflow.set_tag("model_type", "ann")
        mlflow.set_tag("framework", "tensorflow/keras")
        
        # Log ANN architecture parameters
        mlflow.log_params(PARAMS["ann"]["architecture"])
        mlflow.log_params(PARAMS["ann"]["training"])
        
        class MLflowKerasLogger(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    for k, v in logs.items():
                        mlflow.log_metric(k, float(v), step=epoch)
                        
        callbacks.append(MLflowKerasLogger())
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            callbacks=callbacks,
            verbose=1
        )
        
        # 4. Evaluate ANN model
        # Load best weights
        if os.path.exists(best_weights_path):
            model.load_weights(best_weights_path)
            
        y_prob_val = model.predict(X_val).flatten()
        y_prob_test = model.predict(X_test).flatten()
        
        val_metrics = compute_metrics(y_val, y_prob_val)
        test_metrics = compute_metrics(y_test, y_prob_test)
        
        print(f"Validation metrics: {val_metrics}")
        print(f"Test metrics: {test_metrics}")
        
        # Log metrics (we care most about test metrics for final comparison)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"{k}_final", v)
            
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}_final", v)
            
        # Save and log plots
        os.makedirs("temp_plots", exist_ok=True)
        cm_path = "temp_plots/confusion_matrix.png"
        roc_path = "temp_plots/roc_curve.png"
        pr_path = "temp_plots/pr_curve.png"
        
        save_confusion_matrix_plot(y_test, y_prob_test, cm_path)
        save_roc_curve_plot(y_test, y_prob_test, roc_path)
        save_pr_curve_plot(y_test, y_prob_test, pr_path)
        
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(pr_path)
        
        # Clean up temp plots
        for p in [cm_path, roc_path, pr_path]:
            if os.path.exists(p):
                os.remove(p)
                
        # Log Model
        mlflow.tensorflow.log_model(model, "model")
        print("Successfully logged Keras ANN model to MLflow.")

def main():
    parser = argparse.ArgumentParser(description="Master training script for PRRP models.")
    parser.add_argument(
        "--model-type", 
        type=str, 
        choices=["classical", "ann", "all"], 
        default="all",
        help="Which model family to train."
    )
    parser.add_argument(
        "--tune", 
        action="store_true", 
        help="Whether to perform hyperparameter tuning using Optuna."
    )
    parser.add_argument(
        "--trials", 
        type=int, 
        default=5, 
        help="Number of tuning trials per model family if --tune is set."
    )
    args = parser.parse_args()
    
    processed_dir = os.path.join(MODEL_DIR.replace("models", "data/processed"))
    train_path = os.path.join(processed_dir, "train_processed.csv")
    val_path = os.path.join(processed_dir, "val_processed.csv")
    test_path = os.path.join(processed_dir, "test_processed.csv")
    
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print(f"Loading preprocessed datasets from {processed_dir}...")
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)
        
        X_train_res = df_train.drop(columns=["readmitted"])
        y_train_res = df_train["readmitted"]
        X_val_scaled = df_val.drop(columns=["readmitted"])
        y_val = df_val["readmitted"]
        X_test_scaled = df_test.drop(columns=["readmitted"])
        y_test = df_test["readmitted"]
        
        if os.path.exists(PREPROCESSOR_PATH):
            preprocessor = joblib.load(PREPROCESSOR_PATH)
        else:
            raise FileNotFoundError(f"Preprocessor pkl missing at {PREPROCESSOR_PATH}")
    else:
        print("Preprocessed datasets not found. Running full preprocessing pipeline...")
        # 1. Load Raw Dataset
        df = load_raw_data()
        
        # 2. Preprocess Raw Data (drop missing, mapping target)
        df_clean = preprocess_raw_dataset(df)
        
        # 3. Stratified Split 70/15/15
        df_train, df_val, df_test = split_data(df_clean)
        
        # Separate features and target
        X_train_raw = df_train.drop(columns=["readmitted"])
        y_train = df_train["readmitted"]
        X_val_raw = df_val.drop(columns=["readmitted"])
        y_val = df_val["readmitted"]
        X_test_raw = df_test.drop(columns=["readmitted"])
        y_test = df_test["readmitted"]
        
        # 4. Fit Preprocessor on Train set and transform all
        preprocessor = ReadmissionPreprocessor()
        preprocessor.fit(X_train_raw)
        
        X_train_scaled = preprocessor.transform(X_train_raw)
        X_val_scaled = preprocessor.transform(X_val_raw)
        X_test_scaled = preprocessor.transform(X_test_raw)
        
        # Save the preprocessor to models/preprocessor.pkl
        os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        print(f"Saved fitted preprocessor to {PREPROCESSOR_PATH}")
        
        # 5. Apply SMOTE to training set only
        X_train_res, y_train_res = apply_smote(X_train_scaled, y_train)
        
        # Ensure test arrays are numeric for models
        X_train_res = pd.DataFrame(X_train_res, columns=preprocessor.feature_columns_)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=preprocessor.feature_columns_)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=preprocessor.feature_columns_)
        
        # Save processed splits locally for DVC or reference
        os.makedirs(processed_dir, exist_ok=True)
        X_train_res.assign(readmitted=y_train_res).to_csv(train_path, index=False)
        X_val_scaled.assign(readmitted=y_val).to_csv(val_path, index=False)
        X_test_scaled.assign(readmitted=y_test).to_csv(test_path, index=False)

    
    # 6. Execute Model Training
    if args.model_type in ["classical", "all"]:
        train_classical_models(
            X_train_res, y_train_res, 
            X_val_scaled, y_val, 
            X_test_scaled, y_test, 
            do_tune=args.tune, 
            trials=args.trials
        )
        
    if args.model_type in ["ann", "all"]:
        train_ann_model(
            X_train_res, y_train_res, 
            X_val_scaled, y_val, 
            X_test_scaled, y_test
        )
        
    # 7. Promote Champion Model
    result = promote_champion_model()
    if result:
        print(f"\nTraining pipeline completed successfully! Champion model promoted: {result}")
    else:
        print("\nTraining pipeline completed but no champion model was promoted.")

if __name__ == "__main__":
    main()
