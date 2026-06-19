import os
import yaml

# Base directory setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PARAMS_PATH = os.path.join(BASE_DIR, "params.yaml")

def load_params():
    if not os.path.exists(PARAMS_PATH):
        raise FileNotFoundError(f"Configuration file not found at {PARAMS_PATH}")
    with open(PARAMS_PATH, "r") as f:
        return yaml.safe_load(f)

PARAMS = load_params()

# Path configurations
RAW_DATA_PATH = os.path.join(BASE_DIR, PARAMS["data"]["raw_path"])
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, PARAMS["data"]["processed_dir"])
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Ensure necessary directories exist
os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
