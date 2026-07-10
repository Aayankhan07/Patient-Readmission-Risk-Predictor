import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

try:
    from tensorflow.keras.models import Sequential

    HAS_TENSORFLOW = True
except ImportError:
    Sequential = None
    HAS_TENSORFLOW = False


def explain_patient_lime(
    model, X_train: pd.DataFrame, patient_preprocessed: pd.DataFrame
) -> str:
    """
    Generates a LIME tabular explanation for a single patient.
    Returns:
        str: HTML representation of the explanation.
    """
    # Create prediction function wrapper based on model type
    if HAS_TENSORFLOW and Sequential is not None and isinstance(model, Sequential):

        def predict_fn(x):
            # For tensorflow ANN model
            # Convert inputs to float32
            x_tensor = x.astype(np.float32)
            preds = model.predict(x_tensor, verbose=0)
            return np.hstack([1.0 - preds, preds])
    else:
        # Standard sklearn/xgboost predict_proba
        predict_fn = model.predict_proba

    # Instantiate LimeTabularExplainer
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=["Not Readmitted", "Readmitted"],
        mode="classification",
        random_state=42,
    )

    # Get explanation for the single patient instance
    exp = explainer.explain_instance(
        data_row=patient_preprocessed.iloc[0].values,
        predict_fn=predict_fn,
        num_features=8,
    )

    # Return HTML string
    return exp.as_html()
