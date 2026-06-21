import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from src.models.classical import get_logistic_regression, get_svm, get_random_forest, get_xgboost

# Suppress Optuna logs to keep training output clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune_model(model_name: str, X: np.ndarray, y: np.ndarray, n_trials: int = 100, timeout: int = 1800):
    """
    Tunes hyperparameters for the specified model family using Optuna.
    Optimizes for PR-AUC (Average Precision) using 5-fold Stratified Cross-Validation.
    """
    print(f"Starting hyperparameter tuning for {model_name} (trials={n_trials})...")
    
    def objective(trial):
        if model_name == "logistic_regression":
            params = {
                "C": trial.suggest_float("C", 1e-3, 100.0, log=True)
            }
            model_fn = lambda: get_logistic_regression(params)
            
        elif model_name == "svm":
            params = {
                "C": trial.suggest_float("C", 1e-2, 10.0, log=True),
                "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"])
            }
            model_fn = lambda: get_svm(params)
            
        elif model_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
            }
            model_fn = lambda: get_random_forest(params)
            
        elif model_name == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0)
            }
            # Compute scale_pos_weight based on class distribution in y
            neg, pos = np.bincount(y)
            scale_pos_weight = neg / pos if pos > 0 else 1.0
            model_fn = lambda: get_xgboost(params, scale_pos_weight=scale_pos_weight)
            
        else:
            raise ValueError(f"Unknown model name for tuning: {model_name}")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pr_aucs = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = model_fn()
            model.fit(X_tr, y_tr)
            
            # Predict probabilities
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_val)[:, 1]
            else:
                y_prob = model.decision_function(X_val)
                
            pr_auc = average_precision_score(y_val, y_prob)
            pr_aucs.append(pr_auc)
            
        return np.mean(pr_aucs)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    print(f"Tuning complete. Best Average Precision (PR-AUC): {study.best_value:.4f}")
    return study.best_params
