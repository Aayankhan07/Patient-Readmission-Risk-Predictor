import shap
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential

def get_shap_explainer(model, X_background: pd.DataFrame):
    """
    Returns an appropriate SHAP explainer based on the model type.
    """
    # Sample background dataset if too large to speed up kernel/linear computations
    if len(X_background) > 100:
        X_bg_sampled = shap.sample(X_background, 100)
    else:
        X_bg_sampled = X_background

    # Check model type
    if isinstance(model, (XGBClassifier, RandomForestClassifier)):
        return shap.TreeExplainer(model)
    elif isinstance(model, LogisticRegression):
        return shap.LinearExplainer(model, X_bg_sampled)
    elif isinstance(model, Sequential):
        # TensorFlow Keras ANN
        # Convert background to float32 numpy array for TF
        bg_array = X_bg_sampled.values.astype(np.float32)
        return shap.DeepExplainer(model, bg_array)
    else:
        # Generic fallback
        # Use predict_proba for classification models
        predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        return shap.KernelExplainer(predict_fn, X_bg_sampled)

def explain_patient_shap(model, X_background: pd.DataFrame, patient_preprocessed: pd.DataFrame):
    """
    Generates SHAP values for a single patient.
    Returns:
        dict: feature name -> SHAP value
        float: base/expected value
    """
    explainer = get_shap_explainer(model, X_background)
    
    # Generate SHAP values for the patient row
    if isinstance(model, Sequential):
        # Keras ANN case
        patient_arr = patient_preprocessed.values.astype(np.float32)
        shap_vals = explainer.shap_values(patient_arr)
        # DeepExplainer returns shape (1, num_features, 1) or list
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        shap_vals = shap_vals[0].flatten()
        base_val = float(explainer.expected_value[0])
    elif isinstance(explainer, shap.KernelExplainer):
        patient_arr = patient_preprocessed.values
        shap_vals = explainer.shap_values(patient_arr)
        # Handle binary classification where shap_values shape is (2, features) or list
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        elif len(shap_vals.shape) == 3:
            shap_vals = shap_vals[:, :, 1]
        shap_vals = shap_vals[0].flatten()
        # Handle list/numpy base val
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[1] if len(base_val) > 1 else base_val[0]
        base_val = float(base_val)
    else:
        # Tree or Linear explainer
        patient_arr = patient_preprocessed.values
        shap_vals = explainer.shap_values(patient_arr)
        # Handle output dimensions for binary classification (sometimes TreeExplainer output is list or 3D)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        elif len(shap_vals.shape) == 3:
            shap_vals = shap_vals[0, :, 1] if shap_vals.shape[2] > 1 else shap_vals[0, :, 0]
        elif len(shap_vals.shape) == 2 and shap_vals.shape[0] == 1:
            shap_vals = shap_vals[0]
        
        shap_vals = shap_vals.flatten()
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[1] if len(base_val) > 1 else base_val[0]
        base_val = float(base_val)
        
    # Map to features
    feature_names = patient_preprocessed.columns.tolist()
    shap_dict = {feat: float(val) for feat, val in zip(feature_names, shap_vals)}
    
    return shap_dict, base_val
