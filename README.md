# Patient Readmission Risk Predictor (PRRP)

🏥 **Patient Readmission Risk Predictor (PRRP)** is an end-to-end ML platform that predicts the likelihood of a patient being readmitted to the hospital within 30 days of discharge. It compares a classical ML suite (Logistic Regression, SVM, Random Forest, XGBoost) with a deep learning ANN, log runs to MLflow, and exposes predictions and explainability details (SHAP & LIME) via a FastAPI REST service and an interactive Streamlit clinician dashboard.

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Compose Network                    │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │  Streamlit   │───▶│  FastAPI     │───▶│  MLflow Server    │  │
│  │  Dashboard   │    │  :8000       │    │  :5000            │  │
│  │  :8501       │    │              │    │                   │  │
│  └──────────────┘    └──────┬───────┘    └───────────────────┘  │
│                             │                      │             │
│                    ┌────────▼────────┐   ┌─────────▼─────────┐  │
│                    │  Model Loader   │   │  Artifact Store   │  │
│                    │  (champion)     │   │  ./mlruns/        │  │
│                    └────────┬────────┘   └───────────────────┘  │
│                             │                                    │
│                    ┌────────▼────────┐                          │
│                    │  Preprocessor   │                          │
│                    │  (joblib pkl)   │                          │
│                    └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

- `data/` — contains raw diabetic patient CSV and processed stratified splits.
- `src/` — core ML modules:
  - `data/` — data loader, preprocessor class, and splitting.
  - `models/` — model definitions, Optuna tuning objectives, and registry promotion.
  - `explainability/` — SHAP and LIME wrappers.
  - `evaluation/` — metric evaluation and plotting functions.
  - `utils/` — application config loader.
- `api/` — FastAPI web server (routes, schemas).
- `dashboard/` — Streamlit dashboard (tabs, sidebar, charts, risk card).
- `scripts/` — master training and registry promotion triggers.
- `tests/` — unit testing suite.
- `docker/` — container configuration files.

---

## 3. Local Setup & Execution

### 3.1 Setup Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

### 3.2 Run the Model Training Pipeline
The master training script loads the raw data (generating a synthetic dataset if missing), preprocesses features, splits them stratifiably, oversamples with SMOTE, tunes hyperparameters using Optuna (optional), logs all runs to MLflow, and automatically promotes the best model to the Registry:
```bash
# Fast baseline training (no tuning)
python scripts/train_all.py

# Full model tuning with Optuna (10 trials per model family)
python scripts/train_all.py --tune --trials 10
```

### 3.3 Start the Services (Standalone)
**Start local MLflow Tracking Server:**
```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlruns
```

**Start FastAPI Serving Endpoint:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Start Streamlit Dashboard:**
```bash
streamlit run dashboard/app.py --server.port=8501
```

---

## 4. Run with Docker Compose (Recommended)
You can build and deploy the entire platform (MLflow, Serving API, and Streamlit Dashboard) with a single command:
```bash
docker compose up --build
```
- **Streamlit Dashboard UI:** `http://localhost:8501`
- **FastAPI Documentation:** `http://localhost:8000/docs`
- **MLflow Tracking UI:** `http://localhost:5000`

---

## 5. API Reference

- `GET /health` — Check server liveness and model status.
- `GET /models` — List the Production champion model version.
- `POST /predict` — Score a patient's readmission risk.
  - Body: Patient clinical features JSON.
  - Response: Risk score, tier (low/medium/high), and uuid `patient_id`.
- `GET /explain/{patient_id}` — Get SHAP feature values and LIME HTML report.

---

## 6. Run Unit Tests
To verify preprocessors, metrics, and API contract correctness, execute:
```bash
pytest tests/ -v
```
