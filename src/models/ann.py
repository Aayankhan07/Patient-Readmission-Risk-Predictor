import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from src.utils.config import PARAMS

def build_ann_model(input_dim: int, learning_rate: float = None) -> Sequential:
    """Builds and compiles the Keras ANN for Readmission Risk Prediction."""
    cfg = PARAMS["ann"]
    arch = cfg["architecture"]
    
    if learning_rate is None:
        learning_rate = cfg["training"]["learning_rate"]
        
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(arch["dense1"], activation="relu"),
        BatchNormalization(),
        Dropout(arch["dropout1"]),
        Dense(arch["dense2"], activation="relu"),
        BatchNormalization(),
        Dropout(arch["dropout2"]),
        Dense(arch["dense3"], activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="pr_auc", curve="PR"), tf.keras.metrics.AUC(name="roc_auc", curve="ROC")]
    )
    
    return model
