from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.utils.config import PARAMS

def get_logistic_regression(params=None):
    """Returns a Logistic Regression model with given or default parameters."""
    cfg = PARAMS["models"]["logistic_regression"]
    merged_params = {
        "penalty": cfg["penalty"],
        "solver": cfg["solver"],
        "max_iter": cfg["max_iter"],
        "random_state": cfg["random_state"]
    }
    if params:
        merged_params.update(params)
    return LogisticRegression(**merged_params)

def get_svm(params=None):
    """Returns an SVM model with given or default parameters."""
    cfg = PARAMS["models"]["svm"]
    merged_params = {
        "probability": cfg["probability"],
        "gamma": cfg["gamma"],
        "random_state": cfg["random_state"]
    }
    if params:
        merged_params.update(params)
    return SVC(**merged_params)

def get_random_forest(params=None):
    """Returns a Random Forest classifier with given or default parameters."""
    cfg = PARAMS["models"]["random_forest"]
    merged_params = {
        "random_state": cfg["random_state"],
        "n_jobs": cfg["n_jobs"]
    }
    if params:
        merged_params.update(params)
    return RandomForestClassifier(**merged_params)

def get_xgboost(params=None, scale_pos_weight=None):
    """Returns an XGBoost classifier with given or default parameters."""
    cfg = PARAMS["models"]["xgboost"]
    merged_params = {
        "eval_metric": cfg["eval_metric"],
        "random_state": cfg["random_state"]
    }
    if scale_pos_weight is not None:
        merged_params["scale_pos_weight"] = scale_pos_weight
    else:
        merged_params["scale_pos_weight"] = cfg["scale_pos_weight"]
        
    if params:
        merged_params.update(params)
    return XGBClassifier(**merged_params)
