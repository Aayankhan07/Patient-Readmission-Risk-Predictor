# Product Requirements Document
## Patient Readmission Risk Predictor (PRRP)
**Version:** 1.0  
**Author:** Aayan Khan  
**Date:** June 2026  
**Status:** Active

---

## 1. Overview

### 1.1 Product Summary
The Patient Readmission Risk Predictor (PRRP) is a machine learning platform that predicts the probability of a patient being readmitted to hospital within 30 days of discharge. It combines a classical ML model suite (Logistic Regression, SVM, Random Forest, XGBoost) with a deep learning ANN — comparing both paradigms on the same problem and exposing predictions through a clinician-facing dashboard and REST API.

### 1.2 Problem Statement
Hospital readmissions within 30 days cost health systems billions annually and are often preventable. Clinicians lack a fast, explainable tool to identify high-risk patients at discharge. This project builds that tool end-to-end — from raw data ingestion to a live scoring interface with per-patient SHAP explanations.

### 1.3 Target Users
| User | Goal |
|------|------|
| Clinician / Nurse | Review risk score and explanation for a patient at discharge |
| Data Scientist (portfolio viewer) | Inspect model comparison, experiment tracking, code quality |
| ML Engineer (interviewer) | Evaluate pipeline architecture, MLOps practices, reproducibility |

### 1.4 Success Metrics
- PR-AUC ≥ 0.72 on held-out test set (best model)
- API p99 inference latency ≤ 200ms
- All experiments reproducible via MLflow run IDs
- SHAP explanation generated for every prediction
- Full pipeline runnable with `docker compose up`

---

## 2. Scope

### 2.1 In Scope
- Data preprocessing pipeline (imputation, encoding, SMOTE, scaling)
- Classical ML suite: Logistic Regression, SVM, Random Forest, XGBoost
- Deep learning model: Keras ANN with Dropout + BatchNorm
- Hyperparameter tuning: Optuna for all models
- Experiment tracking: MLflow (local server)
- Explainability: SHAP (global + per-patient) and LIME (per-patient)
- Inference API: FastAPI with Pydantic validation
- Clinician dashboard: Streamlit
- Containerisation: Docker + Docker Compose
- Model registry: MLflow Model Registry (champion model promotion)

### 2.2 Out of Scope
- EHR system integration (HL7 / FHIR)
- Multi-hospital federated learning
- Real-time data streaming
- Mobile application
- Production cloud deployment (AWS/GCP/Azure)
- Patient authentication / HIPAA compliance

---

## 3. Functional Requirements

### FR-01 — Data Pipeline
- Load raw CSV (Kaggle Hospital Readmissions dataset or MIMIC-III subset)
- Handle missing values: median for numerics, mode for categoricals
- Encode categoricals: ordinal for ordered, one-hot for nominal
- Balance classes: SMOTE on training split only
- Scale features: StandardScaler (classical) / MinMaxScaler (ANN)
- Versioned with DVC

### FR-02 — Classical ML Suite
- Train: Logistic Regression, SVM, Random Forest, XGBoost
- Tune: Optuna with 100 trials per model, 5-fold stratified CV
- Evaluate: Accuracy, F1, ROC-AUC, PR-AUC, confusion matrix
- Log all runs to MLflow with params, metrics, and artifacts

### FR-03 — ANN Model
- Architecture: Input → Dense(256) → BatchNorm → Dropout(0.3) → Dense(128) → Dropout(0.2) → Dense(1, sigmoid)
- Training: Adam optimizer, binary cross-entropy, ReduceLROnPlateau, EarlyStopping
- Log training curves (loss/accuracy per epoch) to MLflow
- Save best weights via ModelCheckpoint

### FR-04 — Model Comparison
- Side-by-side leaderboard: all models ranked by PR-AUC
- Champion model auto-registered in MLflow Model Registry
- Confusion matrix and ROC curve plots for all models

### FR-05 — Explainability
- SHAP TreeExplainer for tree-based models
- SHAP KernelExplainer for SVM and LR
- SHAP DeepExplainer for ANN
- LIME for per-patient text-style explanation
- Both rendered in the Streamlit dashboard

### FR-06 — FastAPI Inference Endpoint
- `POST /predict` — accepts patient JSON, returns risk score + risk tier
- `GET /explain/{patient_id}` — returns SHAP values for a patient
- `GET /health` — liveness check
- `GET /models` — list registered models and champion
- Input validated with Pydantic; errors return structured JSON

### FR-07 — Streamlit Dashboard
- Patient risk scoring form (manual input)
- Bulk upload (CSV) → scored results table with download
- Model leaderboard tab
- SHAP global feature importance chart
- Per-patient SHAP waterfall plot
- LIME explanation panel

### FR-08 — MLflow Tracking Server
- Experiment per model family (classical / deep learning)
- Runs searchable by metric, date, model type
- Artifact store for models, plots, preprocessors

---

## 4. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-01 | API inference latency p99 ≤ 200ms |
| NFR-02 | Full pipeline setup time ≤ 5 min via Docker Compose |
| NFR-03 | All secrets via environment variables, no hardcoded values |
| NFR-04 | Unit test coverage ≥ 70% on preprocessing and API modules |
| NFR-05 | README runnable by a fresh clone with no manual steps beyond `docker compose up` |
| NFR-06 | Python 3.10+, reproducible via `requirements.txt` and pinned versions |

---

## 5. Dataset

**Primary:** Kaggle — "Hospital Readmissions" (diabetic patient dataset, ~100k rows, 50 features)  
**Fallback:** MIMIC-III 30-day readmission subset (requires PhysioNet credentialing)

**Key features:**
- Demographics: age, gender, race
- Admission info: time_in_hospital, num_procedures, num_medications
- Lab results: glucose, A1C, insulin
- Diagnosis codes: primary, secondary, tertiary (ICD-9)
- Target: `readmitted` → binary (readmitted within 30 days: yes/no)

---

## 6. Constraints
- No GPU required — ANN trains on CPU in under 10 min on standard laptop
- Dataset must be downloadable without institutional credentials (Kaggle preferred)
- All open-source tooling only (no paid APIs or cloud services)

---

## 7. Milestones

| Week | Deliverable |
|------|-------------|
| Week 1 | Data pipeline, EDA, classical ML suite trained and logged |
| Week 2 | ANN trained, SHAP/LIME integration, FastAPI endpoint |
| Week 3 | Streamlit dashboard, Docker Compose, README, GitHub polish |
