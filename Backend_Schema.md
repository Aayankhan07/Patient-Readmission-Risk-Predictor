# Backend Schema
## Patient Readmission Risk Predictor (PRRP)
**Version:** 1.0

---

## 1. Overview

This project is **not transactional** — there is no relational database for patient records (no PHI is persisted; every prediction is stateless and ephemeral, by design, to avoid HIPAA scope). Instead, "backend schema" here covers three things: the data schema used for training, the Pydantic request/response schemas for the API, and the MLflow tracking schema.

If persistent prediction logging is later required, a lightweight SQLite/Postgres schema is included in section 5 as an optional extension.

---

## 2. Training Data Schema

### 2.1 Raw Dataset Columns (Kaggle Hospital Readmissions)

| Column | Type | Description |
|--------|------|-------------|
| `encounter_id` | int | Unique encounter identifier (dropped before training) |
| `patient_nbr` | int | Patient identifier (dropped before training) |
| `race` | categorical | Patient race |
| `gender` | categorical | Male / Female |
| `age` | categorical (ordinal) | Bracketed: [0-10), [10-20), ... [90-100) |
| `time_in_hospital` | int | Days admitted |
| `num_lab_procedures` | int | Number of lab tests performed |
| `num_procedures` | int | Number of non-lab procedures |
| `num_medications` | int | Number of distinct medications |
| `number_outpatient` | int | Outpatient visits in prior year |
| `number_emergency` | int | Emergency visits in prior year |
| `number_inpatient` | int | Inpatient visits in prior year |
| `number_diagnoses` | int | Number of diagnoses entered |
| `max_glu_serum` | categorical | None / Norm / >200 / >300 |
| `A1Cresult` | categorical | None / Norm / >7 / >8 |
| `insulin` | categorical | No / Down / Steady / Up |
| `diabetesMed` | categorical | Yes / No |
| `change` | categorical | Whether medication changed: Ch / No |
| `readmitted` | categorical (target) | NO / >30 / <30 |

### 2.2 Processed Feature Schema (post-pipeline)

```python
PROCESSED_SCHEMA = {
    # Numeric (scaled)
    "time_in_hospital": "float64",
    "num_lab_procedures": "float64",
    "num_procedures": "float64",
    "num_medications": "float64",
    "number_outpatient": "float64",
    "number_emergency": "float64",
    "number_inpatient": "float64",
    "number_diagnoses": "float64",

    # Ordinal encoded
    "age_ordinal": "int8",          # 0-9 representing age brackets

    # One-hot encoded (examples — full set ~45 columns after encoding)
    "gender_Male": "uint8",
    "race_Caucasian": "uint8",
    "race_AfricanAmerican": "uint8",
    "max_glu_serum_Norm": "uint8",
    "max_glu_serum_>200": "uint8",
    "max_glu_serum_>300": "uint8",
    "A1Cresult_Norm": "uint8",
    "A1Cresult_>7": "uint8",
    "A1Cresult_>8": "uint8",
    "insulin_Down": "uint8",
    "insulin_Steady": "uint8",
    "insulin_Up": "uint8",
    "diabetesMed_Yes": "uint8",
    "change_Ch": "uint8",

    # Target
    "readmitted_30d": "uint8"       # 1 if <30 days, else 0
}
```

### 2.3 Train/Val/Test Split
```
Total: 100% of cleaned data
├── Train: 70% (SMOTE applied here only)
├── Validation: 15% (used for Optuna trials + early stopping)
└── Test: 15% (held out, used only for final reported metrics)

Stratification: on readmitted_30d to preserve class ratio across splits
Random state: 42 (fixed for reproducibility)
```

---

## 3. API Request/Response Schemas (Pydantic)

### 3.1 PatientInput (Request)
```python
from pydantic import BaseModel, Field
from typing import Literal

class PatientInput(BaseModel):
    age: Literal[
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
    ]
    gender: Literal["Male", "Female"]
    race: Literal[
        "Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"
    ]
    time_in_hospital: int = Field(ge=1, le=14)
    num_lab_procedures: int = Field(ge=0, le=132)
    num_procedures: int = Field(ge=0, le=6)
    num_medications: int = Field(ge=1, le=81)
    number_outpatient: int = Field(ge=0, le=42)
    number_emergency: int = Field(ge=0, le=76)
    number_inpatient: int = Field(ge=0, le=21)
    number_diagnoses: int = Field(ge=1, le=16)
    max_glu_serum: Literal["None", "Norm", ">200", ">300"]
    A1Cresult: Literal["None", "Norm", ">7", ">8"]
    insulin: Literal["No", "Down", "Steady", "Up"]
    diabetesMed: Literal["Yes", "No"]
    change: Literal["Ch", "No"]

    class Config:
        json_schema_extra = {
            "example": {
                "age": "[60-70)",
                "gender": "Female",
                "race": "Caucasian",
                "time_in_hospital": 5,
                "num_lab_procedures": 45,
                "num_procedures": 2,
                "num_medications": 14,
                "number_outpatient": 0,
                "number_emergency": 1,
                "number_inpatient": 0,
                "number_diagnoses": 7,
                "max_glu_serum": "None",
                "A1Cresult": ">8",
                "insulin": "Steady",
                "diabetesMed": "Yes",
                "change": "Ch"
            }
        }
```

### 3.2 PredictionResponse
```python
class PredictionResponse(BaseModel):
    patient_id: str            # generated UUID for this request
    risk_score: float          # 0.0 - 1.0
    risk_tier: Literal["low", "medium", "high"]
    model_version: str         # e.g. "xgboost_v3"
    inference_ms: float
```

### 3.3 ExplainResponse
```python
class ExplainResponse(BaseModel):
    patient_id: str
    shap_values: dict[str, float]      # feature_name -> shap contribution
    top_risk_factors: list[str]        # top 3 feature names, descending |shap|
    base_value: float                  # SHAP expected value
    lime_html: str                     # rendered LIME explanation as HTML

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
    version: str

class ModelInfo(BaseModel):
    name: str
    stage: Literal["Production", "Staging", "Archived"]
    pr_auc: float
    roc_auc: float
    f1: float
    accuracy: float
    is_champion: bool

class ModelsResponse(BaseModel):
    models: list[ModelInfo]
```

---

## 4. MLflow Tracking Schema

### 4.1 Run Naming Convention
```
{model_family}_{model_type}_{trial_or_version}
e.g.
  classical_logistic_regression_trial_047
  classical_xgboost_trial_012
  deep_learning_ann_v2
```

### 4.2 Logged Schema per Run
```python
mlflow.log_params({
    "model_type": str,
    "n_estimators": int,        # if applicable
    "max_depth": int,           # if applicable
    "learning_rate": float,     # if applicable
    "random_state": 42,
    "smote_applied": bool,
    "train_size": int,
    "val_size": int,
})

mlflow.log_metrics({
    "accuracy": float,
    "precision": float,
    "recall": float,
    "f1": float,
    "roc_auc": float,
    "pr_auc": float,
})

mlflow.log_artifacts({
    "model/": "serialized model (pickle/keras)",
    "confusion_matrix.png": "image",
    "roc_curve.png": "image",
    "pr_curve.png": "image",
    "shap_summary.png": "image",
})
```

### 4.3 Model Registry Schema
```
Registered Model Name: "readmission_champion"

Version 1: xgboost_trial_012   | Stage: Archived
Version 2: random_forest_t089  | Stage: Archived
Version 3: xgboost_trial_047   | Stage: Production  ← current champion
```

---

## 5. Optional Extension — Prediction Logging Schema

Not in MVP scope, but documented for future extension if audit trail is needed:

```sql
CREATE TABLE prediction_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMP DEFAULT now(),
    model_version   VARCHAR(64) NOT NULL,
    risk_score      FLOAT NOT NULL,
    risk_tier       VARCHAR(10) NOT NULL,
    inference_ms    FLOAT NOT NULL,
    input_hash      VARCHAR(64) NOT NULL,   -- SHA256 of input, no PHI stored
    top_risk_factor VARCHAR(64)
);

CREATE INDEX idx_prediction_log_created_at ON prediction_log(created_at);
CREATE INDEX idx_prediction_log_model_version ON prediction_log(model_version);
```

Note: `input_hash` is stored instead of raw patient data specifically to keep this extension HIPAA-scope-free — no PHI ever touches disk.
