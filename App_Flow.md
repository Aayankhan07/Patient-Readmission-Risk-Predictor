# App Flow Document
## Patient Readmission Risk Predictor (PRRP)
**Version:** 1.0

---

## 1. User Roles

| Role | Access |
|------|--------|
| Clinician | Dashboard: single + bulk scoring, explanations |
| Reviewer (interviewer/recruiter) | Dashboard: leaderboard, model comparison, GitHub repo |
| Developer | API direct access, MLflow UI, DVC pipeline |

---

## 2. Primary Flow — Single Patient Risk Scoring

```
┌──────────────┐
│ Clinician     │
│ opens         │
│ dashboard     │
└──────┬───────┘
       │
       ▼
┌─────────────────────────┐
│ Landing page             │
│ - Sidebar: nav tabs       │
│ - Default tab: "Score     │
│   Patient"                │
└──────┬───────────────────┘
       │
       ▼
┌─────────────────────────┐
│ Fill patient form         │
│ - Age bracket (dropdown)  │
│ - Time in hospital (slider)│
│ - # procedures (number)   │
│ - # medications (number)  │
│ - A1C result (dropdown)   │
│ - Insulin status (dropdown)│
│ - Diabetes med (toggle)   │
└──────┬───────────────────┘
       │ clicks "Calculate Risk"
       ▼
┌─────────────────────────┐
│ Dashboard → FastAPI       │
│ POST /predict              │
└──────┬───────────────────┘
       │
       ▼
┌─────────────────────────┐
│ API loads champion model  │
│ from MLflow registry      │
│ Preprocesses input         │
│ Returns risk_score +      │
│ risk_tier                 │
└──────┬───────────────────┘
       │
       ▼
┌─────────────────────────┐
│ Dashboard displays:        │
│ - Risk score gauge (0-100%)│
│ - Risk tier badge          │
│   (Low/Medium/High)        │
│ - "View Explanation" button│
└──────┬───────────────────┘
       │ clicks "View Explanation"
       ▼
┌─────────────────────────┐
│ Dashboard → FastAPI       │
│ GET /explain/{patient_id} │
└──────┬───────────────────┘
       │
       ▼
┌─────────────────────────┐
│ SHAP waterfall chart       │
│ + LIME text explanation    │
│ + Top 3 risk factors panel │
└─────────────────────────┘
```

---

## 3. Secondary Flow — Bulk Scoring

```
Clinician → Upload CSV (batch of patients)
    → Dashboard validates schema (column check)
    → If invalid: show error with missing/extra columns
    → If valid: POST each row to /predict (or batch endpoint)
    → Progress bar shown during scoring
    → Results table rendered:
        | Patient | Risk Score | Tier | Top Risk Factor |
    → Sortable/filterable by tier
    → "Download Results CSV" button
```

---

## 4. Tertiary Flow — Model Leaderboard (Reviewer View)

```
Reviewer → clicks "Model Comparison" tab
    → Dashboard → FastAPI → GET /models
    → Table renders:
        | Model | Accuracy | F1 | ROC-AUC | PR-AUC | Status |
        | XGBoost | 0.84 | 0.71 | 0.88 | 0.74 | ★ Champion |
        | Random Forest | 0.82 | 0.68 | 0.85 | 0.70 | |
        | ANN | 0.81 | 0.66 | 0.84 | 0.69 | |
        | SVM | 0.79 | 0.62 | 0.81 | 0.65 | |
        | Logistic Regression | 0.76 | 0.58 | 0.78 | 0.61 | |
    → Click any row → expands to show:
        - Confusion matrix
        - ROC curve
        - PR curve
        - Link to full MLflow run
```

---

## 5. Error & Edge-Case Flows

| Scenario | System Behaviour |
|----------|-------------------|
| Missing required field in form | Inline validation error, submit disabled |
| API unreachable | Dashboard shows "Service unavailable — check API connection" banner, no crash |
| Malformed CSV upload | Row-level error report listing which rows failed and why |
| Model registry empty (first run) | Dashboard shows "No champion model registered yet" with link to training docs |
| SHAP computation timeout (>5s) | Fallback to cached global feature importance, with note "Live explanation unavailable" |
| Risk score exactly at tier boundary | Rounds to nearest tier, boundary documented in tooltip |

---

## 6. Navigation Map

```
┌─────────────────────────────────────────┐
│              Sidebar (persistent)         │
│ ├─────────────────────────────────────────┤
│  ● Score Patient        (default)         │
│  ● Bulk Upload                            │
│  ● Model Comparison                       │
│  ● About / Methodology                    │
└─────────────────────────────────────────┘
```

Each tab is a stateless view — switching tabs does not lose form input within a session (Streamlit session_state persists values).

---

## 7. Developer Flow — Training & Promotion

```
Developer runs: dvc repro
    → prepare stage (preprocessing)
    → train_classical stage (4 models, Optuna tuning, MLflow logging)
    → train_ann stage (Keras training, MLflow logging)

Developer runs: python scripts/promote_champion.py
    → Queries MLflow for best PR-AUC across all experiments
    → Transitions that model version to "Production" stage
    → Archives previous Production model

Developer runs: docker compose up
    → mlflow service starts (serves registry + UI on :5000)
    → api service starts, loads Production model on boot
    → dashboard service starts, connects to api on :8000
```
