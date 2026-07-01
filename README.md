# 🏥 Patient Readmission Risk Predictor (PRRP)

[![CI](https://github.com/Aayankhan07/Patient-Readmission-Risk-Predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/Aayankhan07/Patient-Readmission-Risk-Predictor/actions/workflows/ci.yml)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit)](https://streamlit.io)
[![DVC](https://img.shields.io/badge/DVC-9CF?style=flat&logo=data-version-control)](https://dvc.org)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow)](https://mlflow.org)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python)](https://www.python.org)

🏥 **Patient Readmission Risk Predictor (PRRP)** is a production-grade, end-to-end Machine Learning platform designed to predict the likelihood of diabetic patients being readmitted to the hospital within 30 days of discharge. 

The platform implements a comparative model selection process between a classical ML family (Logistic Regression, Random Forest, XGBoost) and a deep learning artificial neural network (ANN). Runs, parameters, artifacts, and metrics are tracked via **MLflow**, data pipelines are orchestrated using **DVC**, and explanations (SHAP & LIME) are served via a **FastAPI** backend to a modern, clinical **Streamlit** dashboard.

---

## 1. System Architecture & Information Flow

The platform is fully containerized and orchestrated via Docker Compose:

```
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                              DOCKER COMPOSE NETWORK                                    │
 │                                                                                        │
 │  ┌───────────────────────┐          ┌─────────────────────┐      ┌──────────────────┐  │
 │  │                       │          │                     │      │                  │  │
 │  │  Streamlit Dashboard  │ ───────▶ │  FastAPI Web App    │ ───▶ │  MLflow Server   │  │
 │  │  (Clinician View)     │          │  (Port 8000)        │      │  (Port 5000)     │  │
 │  │  - Score Patient      │          │                     │      │                  │  │
 │  │  - Bulk CSV Upload    │          │  - ModelState       │      └──────────┬───────┘  │
 │  │  - Model Leaderboard  │          │    Manager          │                 │          │
 │  │  (Port 8501)          │          │                     │                 │          │
 │  └───────────────────────┘          └──────────┬──────────┘                 │          │
 │                                                │                            │          │
 │                                                ▼                            ▼          │
 │                                     ┌─────────────────────┐      ┌──────────────────┐  │
 │                                     │                     │      │                  │  │
 │                                     │    Preprocessor     │      │  Artifact Store  │  │
 │                                     │    (joblib pkl)     │      │  ./mlruns/       │  │
 │                                     │                     │      │                  │  │
 │                                     └─────────────────────┘      └──────────────────┘  │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

### 🛡️ Robust Model Loading & Fallback Strategy
When the FastAPI server starts, its lifecycle manager (`lifespan`) initializes a singleton `ModelState` that handles model resolving automatically, following a strict fallback sequence:
1. **MLflow Production Registry**: Attempts to download the registered champion model (`readmission_champion` in `Production` stage).
2. **Local Runs Directory Scanning**: If the MLflow server is offline or empty, it scans the local `./mlruns` directory, filters for all runs containing `pr_auc`, sorts them in descending order, and loads the best run as a local champion.
3. **Automated Fallback Training**: If no runs are found locally, the server dynamically generates synthetic training data, fits a `ReadmissionPreprocessor`, trains a baseline `LogisticRegression` model, and loads it. **This ensures the API remains fully functional and robust even in a completely fresh sandbox.**

---

## 2. Directory Structure

```
.
├── api/                       # FastAPI serving layer
│   ├── routes/                # Endpoint routers (health, predict, explain)
│   ├── schemas/               # Pydantic request and response schemas
│   ├── main.py                # FastAPI entrypoint and lifespan management
│   └── model_loader.py        # ModelState singleton & MLflow loading fallback logic
├── dashboard/                 # Streamlit clinician frontend
│   ├── components/            # UI components (sidebar, risk card, charts, leaderboard)
│   ├── styles.py              # Clinical visual theme and CSS overrides
│   └── app.py                 # Streamlit main multi-tab dashboard entrypoint
├── data/                      # Dataset repository
│   ├── raw/                   # Raw input clinical CSV
│   └── processed/             # Preprocessed stratified training splits (SMOTE applied)
├── docker/                    # Containerization assets
│   ├── Dockerfile.api         # FastAPI container
│   ├── Dockerfile.dashboard   # Streamlit container
│   └── Dockerfile.mlflow      # MLflow server container
├── models/                    # Serialized local artifacts (preprocessor.pkl)
├── scripts/                   # Model training and orchestration pipelines
│   ├── promote_champion.py    # Standalone script to register/promote the best run
│   └── train_all.py           # Master training & evaluation script
├── src/                       # Core ML package modules
│   ├── data/                  # Loader, preprocessor, and splitter modules
│   ├── evaluation/            # Custom metrics and plot generators
│   ├── explainability/        # SHAP and LIME explainer wrappers
│   ├── models/                # Classical estimators, ANN builder, and Optuna tuner
│   └── utils/                 # Parameter configuration loaders
├── tests/                     # Unit test suite
├── docker-compose.yml         # Container orchestrator
├── dvc.yaml                   # Data version control pipeline stages
├── params.yaml                # Global parameters (data splits, model parameters, networks)
└── requirements.txt           # Main python dependency manifest
```

---

## 3. Local Setup & Execution

### 3.1 Setup Virtual Environment
Ensure you have Python 3.10+ installed:
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 3.2 Execute DVC Pipeline
Data preprocessing, feature engineering, and model training steps are versioned and structured as reproducible stages in `dvc.yaml`. Run the pipeline using:
```bash
# Execute the entire machine learning pipeline
dvc repro

# Visualize the pipeline stages
dvc dag
```

The pipeline defines three stages:
1. **`prepare`**: Loads `data/raw/diabetic_data.csv`, cleans features, splits data stratifiably (70/15/15), fits the `ReadmissionPreprocessor`, exports `models/preprocessor.pkl`, and saves processed splits under `data/processed/`.
2. **`train_classical`**: Trains Logistic Regression, Random Forest, and XGBoost models, logs metrics, plots, and models to MLflow.
3. **`train_ann`**: Compiles and trains the Keras Artificial Neural Network with Early Stopping and logs the training curves to MLflow.

---

## 4. Run the Model Training Pipeline (Custom Flags)

The master training script allows for custom options bypass and hyperparameter search via **Optuna**:
```bash
# Fast training with default parameters in params.yaml
python -m scripts.train_all

# Hyperparameter optimization (Tuning 10 Optuna trials per model family)
python -m scripts.train_all --tune --trials 10
```

### Hyperparameter Configurations (`params.yaml`)
Parameters are controlled globally from [params.yaml](file:///c:/Users/Adeen/OneDrive/Desktop/Patient%20Readmission%20Risk%20Predictor/params.yaml). You can tweak model properties directly:
* **`data`**: Set validation and test split ratios.
* **`models`**: Configure penalty criteria, estimators, and class imbalance handling (`scale_pos_weight` for XGBoost).
* **`ann`**: Adjust network structure (neurons, dropout rates) and training constraints (epochs, batch size, learning rates, patience limits).

---

## 5. Running the Serving Architecture

### 5.1 Run with Docker Compose (Recommended)
Build and spin up MLflow, FastAPI, and Streamlit services within a virtual network using:
```bash
docker compose up --build
```
* **FastAPI Docs UI**: `http://localhost:8000/docs`
* **Streamlit Dashboard UI**: `http://localhost:8501`
* **MLflow Tracking UI**: `http://localhost:5000`

### 5.2 Start Services Standalone
If you prefer running services directly on your host:

1. **Start MLflow Tracking Server:**
   ```bash
   mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
   ```
2. **Start FastAPI Service:**
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
3. **Start Streamlit Clinician Dashboard:**
   ```bash
   streamlit run dashboard/app.py --server.port=8501
   ```

---

## 6. API Reference

### `GET /health`
Validates model status, preprocessor status, and system health.
* **Response:**
  ```json
  {
    "status": "ok",
    "model_loaded": true,
    "preprocessor_loaded": true,
    "version": "1.0.0"
  }
  ```

### `GET /models`
Retrieves a list of all logged runs and registered models in MLflow, sorted by `pr_auc` descending.
* **Response:**
  ```json
  {
    "models": [
      {
        "name": "Xgboost",
        "stage": "Production",
        "pr_auc": 0.742,
        "roc_auc": 0.812,
        "f1": 0.658,
        "accuracy": 0.792,
        "is_champion": true
      }
    ]
  }
  ```

### `POST /predict`
Scores a patient's risk profile. Unspecified secondary features are set to historical database medians.
* **Request Body Example:**
  ```json
  {
    "age": "[60-70)",
    "time_in_hospital": 5,
    "num_procedures": 2,
    "num_medications": 14,
    "number_diagnoses": 7,
    "A1Cresult": ">8",
    "insulin": "Steady",
    "diabetesMed": "Yes"
  }
  ```
* **Response Example:**
  ```json
  {
    "patient_id": "8f8303d9-9abf-4be1-987d-419b4ea4e857",
    "risk_score": 0.68,
    "risk_tier": "high",
    "model_version": "xgboost_run_8f8303d9",
    "inference_ms": 12.45
  }
  ```

### `GET /explain/{patient_id}`
Computes SHAP waterfall impacts and LIME interactive local visual structures for a cached prediction.
* **Response:**
  ```json
  {
    "patient_id": "8f8303d9-9abf-4be1-987d-419b4ea4e857",
    "shap_values": {
      "time_in_hospital": 0.12,
      "num_medications": -0.04
    },
    "top_risk_factors": ["time_in_hospital", "number_diagnoses", "insulin"],
    "base_value": 0.42,
    "lime_html": "<html>...</html>"
  }
  ```

---

## 7. Streamlit Clinician Dashboard Features

The dashboard provides a premium medical decision support workspace:
1. **Score Patient Tab**:
   * Renders real-time demographic and diagnostic inputs via sliders and dropdowns.
   * Renders a clean **Risk Assessment Card** detailing probability, tier bounds, and latency.
   * Displays an interactive **SHAP Feature Contribution Plot** showing individual feature weights driving the current risk score.
   * Embeds an interactive **LIME Explanation HTML frame** indicating local decision boundary approximations.
2. **Bulk CSV Upload Tab**:
   * Accepts multiple patient profiles in a CSV formatted file.
   * Validates dataset formatting and handles missing values.
   * Evaluates records asynchronously and generates a downloadable scored dataset with `Risk Score` and `Risk Tier`.
3. **Model Comparison Tab**:
   * Fetches the run leaderboard from MLflow and displays a comparative metrics table.
   * Outlines relative metrics (PR-AUC, ROC-AUC, F1, Accuracy) of all trained runs to assist in clinical audits.

---

## 8. Run Unit Tests & CI/CD

Verify the test suite for preprocessor scaling, prediction logic, metrics calculations, and API contracts by executing:
```bash
pytest tests/ -v
```

### GitHub Actions CI Workflow
The repository utilizes a CI workflow located at [.github/workflows/ci.yml](file:///c:/Users/Adeen/OneDrive/Desktop/Patient%20Readmission%20Risk%20Predictor/.github/workflows/ci.yml) that automatically runs on every push and pull request to the `main` branch. The job performs the following:
1. Spins up an `ubuntu-latest` runner.
2. Installs Python 3.10.
3. Installs package dependencies via `pip`.
4. Executes the full test suite via `pytest`.
