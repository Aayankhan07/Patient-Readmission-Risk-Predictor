# Technical Requirements Document
## Patient Readmission Risk Predictor (PRRP)
**Version:** 1.0  
**Author:** Aayan Khan  
**Date:** June 2026

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

## 2. Technology Stack

### 2.1 Core ML
| Library | Version | Purpose |
|---------|---------|---------|
| scikit-learn | ≥1.4 | LR, SVM, RF, preprocessing, metrics |
| xgboost | ≥2.0 | XGBoost classifier |
| tensorflow / keras | ≥2.15 | ANN model |
| imbalanced-learn | ≥0.12 | SMOTE oversampling |
| optuna | ≥3.5 | Hyperparameter optimisation |
| shap | ≥0.44 | SHAP explainability |
| lime | ≥0.2 | LIME per-patient explanations |

### 2.2 MLOps
| Library | Version | Purpose |
|---------|---------|---------|
| mlflow | ≥2.12 | Experiment tracking + model registry |
| dvc | ≥3.49 | Data versioning |
| joblib | ≥1.3 | Preprocessor serialisation |

### 2.3 API & Serving
| Library | Version | Purpose |
|---------|---------|---------|
| fastapi | ≥0.111 | REST API framework |
| uvicorn | ≥0.29 | ASGI server |
| pydantic | ≥2.7 | Request/response validation |

### 2.4 Dashboard
| Library | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥1.34 | Clinician-facing UI |
| plotly | ≥5.21 | Interactive charts |
| pandas | ≥2.2 | Data handling in UI |

### 2.5 Infrastructure
| Tool | Purpose |
|------|---------|
| Docker 24+ | Container runtime |
| Docker Compose v2 | Multi-service orchestration |
| Python 3.10 | Runtime |
| GitHub Actions | CI (lint + test on push) |

---

## 3. Project Structure

```
prrp/
├── data/
│   ├── raw/                    # Original CSV (DVC tracked)
│   ├── processed/              # Cleaned, encoded, split datasets
│   └── .dvc/                   # DVC metadata
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   └── 02_baseline.ipynb       # Quick baseline comparisons
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # Raw data loading
│   │   ├── preprocessor.py     # Full preprocessing pipeline
│   │   └── splitter.py         # Train/val/test split with stratification
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classical.py        # LR, SVM, RF, XGBoost training
│   │   ├── ann.py              # Keras ANN definition + training
│   │   ├── tuner.py            # Optuna objective functions
│   │   └── registry.py         # MLflow model promotion logic
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── shap_explainer.py   # SHAP wrappers (Tree/Kernel/Deep)
│   │   └── lime_explainer.py   # LIME wrapper
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py          # Metric computation + plotting
│   └── utils/
│       ├── __init__.py
│       └── config.py           # Central config (paths, hyperparams)
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entry point
│   ├── routes/
│   │   ├── predict.py          # POST /predict
│   │   ├── explain.py          # GET /explain/{patient_id}
│   │   └── health.py           # GET /health, GET /models
│   └── schemas/
│       ├── patient.py          # PatientInput Pydantic model
│       └── response.py         # PredictionResponse, ExplainResponse
├── dashboard/
│   ├── app.py                  # Streamlit entry point
│   └── components/
│       ├── sidebar.py
│       ├── risk_card.py
│       ├── shap_chart.py
│       └── leaderboard.py
├── tests/
│   ├── test_preprocessor.py
│   ├── test_api.py
│   └── test_metrics.py
├── mlruns/                     # MLflow local artifact store (gitignored)
├── scripts/
│   ├── train_all.py            # Master training script
│   └── promote_champion.py     # Promote best model to registry
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.dashboard
│   └── Dockerfile.mlflow
├── docker-compose.yml
├── requirements.txt
├── .github/
│   └── workflows/
│       └── ci.yml
├── dvc.yaml                    # DVC pipeline stages
├── params.yaml                 # Centrally tracked hyperparams
└── README.md
```

---

## 4. Data Pipeline Specification

### 4.1 Preprocessing Steps (in order)
1. Drop columns with > 40% missing values
2. Drop duplicate rows
3. Map target: `readmitted` → binary (1 if `<30`, else 0)
4. Median imputation for numeric columns
5. Mode imputation for categorical columns
6. Ordinal encoding: `age` brackets → integer ordinal
7. One-hot encoding: all remaining categoricals (drop_first=True)
8. Train/Val/Test split: 70/15/15, stratified on target
9. SMOTE: applied to **training set only** (random_state=42)
10. Scaling: StandardScaler fit on training, transform all splits
11. Serialise preprocessor: `joblib.dump` to `models/preprocessor.pkl`

### 4.2 DVC Stages
```yaml
stages:
  prepare:
    cmd: python src/data/preprocessor.py
    deps: [data/raw/diabetic_data.csv]
    outs: [data/processed/]
  
  train_classical:
    cmd: python scripts/train_all.py --model-type classical
    deps: [data/processed/, src/models/classical.py]
    outs: [mlruns/]
  
  train_ann:
    cmd: python scripts/train_all.py --model-type ann
    deps: [data/processed/, src/models/ann.py]
```

---

## 5. Model Specifications

### 5.1 Classical Models

**Logistic Regression**
```python
LogisticRegression(
    C=optuna_tuned,          # search: [0.001, 100], log=True
    penalty='l2',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
```

**SVM**
```python
SVC(
    C=optuna_tuned,          # search: [0.01, 100]
    kernel=optuna_tuned,     # ['rbf', 'linear']
    gamma='scale',
    probability=True,        # required for predict_proba
    random_state=42
)
```

**Random Forest**
```python
RandomForestClassifier(
    n_estimators=optuna_tuned,   # search: [100, 500]
    max_depth=optuna_tuned,      # search: [3, 20]
    min_samples_split=optuna_tuned,
    random_state=42,
    n_jobs=-1
)
```

**XGBoost**
```python
XGBClassifier(
    n_estimators=optuna_tuned,   # search: [100, 500]
    max_depth=optuna_tuned,      # search: [3, 10]
    learning_rate=optuna_tuned,  # search: [0.01, 0.3]
    subsample=optuna_tuned,      # search: [0.6, 1.0]
    colsample_bytree=optuna_tuned,
    scale_pos_weight=computed,   # handles class imbalance
    eval_metric='aucpr',
    random_state=42
)
```

### 5.2 ANN Architecture
```
Input(n_features)
  → Dense(256, activation='relu')
  → BatchNormalization()
  → Dropout(0.3)
  → Dense(128, activation='relu')
  → BatchNormalization()
  → Dropout(0.2)
  → Dense(64, activation='relu')
  → Dense(1, activation='sigmoid')

Optimizer:   Adam(lr=1e-3)
Loss:        BinaryCrossentropy()
Callbacks:   EarlyStopping(patience=10, restore_best_weights=True)
             ReduceLROnPlateau(factor=0.5, patience=5)
             ModelCheckpoint('best_ann.keras')
Epochs:      100 (early stopping will trigger earlier)
Batch size:  256
```

### 5.3 Optuna Configuration
```python
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner()
)
study.optimize(objective, n_trials=100, timeout=1800)
```

---

## 6. MLflow Schema

### 6.1 Experiment Structure
```
Experiments:
  ├── classical_ml/
  │   ├── Run: logistic_regression_trial_001
  │   ├── Run: svm_trial_001
  │   ├── Run: random_forest_trial_001
  │   └── Run: xgboost_trial_001  ← champion (typically)
  └── deep_learning/
      └── Run: ann_v1
```

### 6.2 Logged Artifacts per Run
- **Params:** all hyperparameters
- **Metrics:** accuracy, f1, roc_auc, pr_auc, precision, recall
- **Artifacts:** 
  - `model/` — serialised model
  - `confusion_matrix.png`
  - `roc_curve.png`
  - `pr_curve.png`
  - `shap_summary.png` (feature importance)
  - `optuna_importance.png` (hyperparameter importance)
  - `preprocessor.pkl`

### 6.3 Model Registry
```
Registered Model: "readmission_champion"
  Stage: Production → champion model (best PR-AUC)
  Stage: Staging   → previous champion
  Stage: Archived  → all prior
```

---

## 7. API Specification

### 7.1 Endpoints

**POST /predict**
```json
// Request
{
  "age": "[60-70)",
  "time_in_hospital": 5,
  "num_procedures": 2,
  "num_medications": 14,
  "number_diagnoses": 7,
  "A1Cresult": ">8",
  "insulin": "Steady",
  "diabetesMed": "Yes"
  // ... all required features
}

// Response
{
  "patient_id": "uuid-generated",
  "risk_score": 0.73,
  "risk_tier": "high",     // low / medium / high
  "model_version": "xgboost_v3",
  "inference_ms": 12
}
```

**GET /explain/{patient_id}**
```json
{
  "patient_id": "uuid",
  "shap_values": {
    "time_in_hospital": 0.18,
    "num_medications": 0.14,
    "A1Cresult": 0.12
    // ...
  },
  "top_risk_factors": ["time_in_hospital", "num_medications", "A1Cresult"],
  "lime_explanation": "string (HTML)"
}
```

**GET /health**
```json
{ "status": "ok", "model_loaded": true, "version": "1.0.0" }
```

### 7.2 Risk Tier Thresholds
| Score | Tier |
|-------|------|
| < 0.35 | low |
| 0.35 – 0.65 | medium |
| > 0.65 | high |

---

## 8. Testing

### 8.1 Unit Tests
- `test_preprocessor.py`: imputation, encoding, SMOTE isolation, scaler fit/transform
- `test_api.py`: all endpoint contracts, Pydantic validation errors, health check
- `test_metrics.py`: metric computation correctness on synthetic data

### 8.2 CI Pipeline (GitHub Actions)
```yaml
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.10' }
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --tb=short
      - run: ruff check src/ api/
```

---

## 9. Docker Compose Services

```yaml
services:
  mlflow:
    build: ./docker/Dockerfile.mlflow
    ports: ["5000:5000"]
    volumes: ["./mlruns:/mlruns"]

  api:
    build: ./docker/Dockerfile.api
    ports: ["8000:8000"]
    depends_on: [mlflow]
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000

  dashboard:
    build: ./docker/Dockerfile.dashboard
    ports: ["8501:8501"]
    depends_on: [api]
    environment:
      - API_URL=http://api:8000
```
