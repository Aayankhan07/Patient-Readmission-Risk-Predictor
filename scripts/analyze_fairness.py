import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

# Add project root to sys.path so we can import api modules
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.model_loader import ModelState  # noqa: E402


def calculate_group_metrics(df, group_mask, group_name, model_state):
    group_data = df[group_mask]
    n = len(group_data)

    if n == 0:
        return {
            "name": group_name,
            "sample_size": 0,
            "accuracy": 0.0,
            "roc_auc": 0.0,
            "selection_rate": 0.0,
        }

    X_group = group_data.drop(columns=["readmitted"])
    y_group = group_data["readmitted"]

    # 1. Run predictions directly using the underlying model object
    if model_state.model_type == "ann":
        preds_probs = model_state.model.predict(
            X_group.values.astype(np.float32), verbose=0
        ).flatten()
    else:
        if hasattr(model_state.model, "predict_proba"):
            preds_probs = model_state.model.predict_proba(X_group)
            if len(preds_probs.shape) > 1 and preds_probs.shape[1] > 1:
                preds_probs = preds_probs[:, 1]
            else:
                preds_probs = preds_probs.flatten()
        else:
            preds_probs = model_state.model.predict(X_group).flatten()

    preds_binary = (preds_probs >= 0.5).astype(int)

    # 2. Metrics
    accuracy = float(accuracy_score(y_group, preds_binary))

    # Compute ROC-AUC only if both classes are present in the subgroup slice
    if len(np.unique(y_group)) > 1:
        roc_auc = float(roc_auc_score(y_group, preds_probs))
    else:
        roc_auc = 0.5  # Neutral default for single-class subgroups

    selection_rate = float(np.mean(preds_binary))

    return {
        "name": group_name,
        "sample_size": n,
        "accuracy": round(accuracy, 4),
        "roc_auc": round(roc_auc, 4),
        "selection_rate": round(selection_rate, 4),
    }


def analyze_fairness():
    print("Initializing healthcare demographic fairness analysis...")

    # 1. Load active champion model state
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    model_state = ModelState()
    model_state.load()

    if not model_state.loaded:
        print(
            "WARNING: Model state did not load. Using default baseline configuration."
        )

    # 2. Load preprocessed test dataset
    test_path = "data/processed/test_processed.csv"
    if not os.path.exists(test_path):
        print(
            "ERROR: Processed test dataset 'test_processed.csv' is missing. Run preprocessing first."
        )
        return

    df = pd.read_csv(test_path)

    # 3. Identify and construct subgroup masks based on scaled numerical flags
    race_masks = {}

    # Caucasian
    if "race_Caucasian" in df.columns:
        race_masks["Caucasian"] = df["race_Caucasian"] > 0

    # Asian
    if "race_Asian" in df.columns:
        race_masks["Asian"] = df["race_Asian"] > 0

    # Other
    if "race_Other" in df.columns:
        race_masks["Other"] = df["race_Other"] > 0

    # African American / Baseline Minority (corresponds to all one-hot columns being False)
    all_known_races = [col for col in df.columns if col.startswith("race_")]
    if all_known_races:
        race_masks["African American / Baseline"] = df[all_known_races].max(axis=1) < 0

    # Gender subgroups
    gender_masks = {}
    if "gender_Male" in df.columns:
        gender_masks["Male"] = df["gender_Male"] > 0
        gender_masks["Female"] = df["gender_Male"] < 0
    else:
        gender_masks["Male"] = pd.Series([True] * len(df))
        gender_masks["Female"] = pd.Series([False] * len(df))

    # 4. Compute metrics across subgroups
    fairness_results = {
        "race": [],
        "gender": [],
        "demographic_parity_ratio_race": {},
        "demographic_parity_ratio_gender": {},
    }

    print("\n--- Race Subgroup Evaluations ---")
    for group_name, mask in race_masks.items():
        res = calculate_group_metrics(df, mask, group_name, model_state)
        print(
            f"Group: {group_name:<30} | n: {res['sample_size']:<5} | Acc: {res['accuracy']:.4f} | AUC: {res['roc_auc']:.4f} | SR: {res['selection_rate']:.4f}"
        )
        fairness_results["race"].append(res)

    print("\n--- Gender Subgroup Evaluations ---")
    for group_name, mask in gender_masks.items():
        res = calculate_group_metrics(df, mask, group_name, model_state)
        print(
            f"Group: {group_name:<30} | n: {res['sample_size']:<5} | Acc: {res['accuracy']:.4f} | AUC: {res['roc_auc']:.4f} | SR: {res['selection_rate']:.4f}"
        )
        fairness_results["gender"].append(res)

    # 5. Compute Demographic Parity Ratios
    ref_race_sr = 0.05  # Safe default to avoid division by zero
    for r in fairness_results["race"]:
        if r["name"] == "Caucasian":
            ref_race_sr = r["selection_rate"] if r["selection_rate"] > 0 else 0.05

    for r in fairness_results["race"]:
        ratio = r["selection_rate"] / ref_race_sr
        fairness_results["demographic_parity_ratio_race"][r["name"]] = round(ratio, 4)

    # Gender Demographic Parity: Female vs Male selection rates comparison
    female_sr = 0.05
    male_sr = 0.05
    for g in fairness_results["gender"]:
        if g["name"] == "Female":
            female_sr = g["selection_rate"] if g["selection_rate"] > 0 else 0.05
        elif g["name"] == "Male":
            male_sr = g["selection_rate"] if g["selection_rate"] > 0 else 0.05

    gender_ratio = female_sr / male_sr if male_sr > 0 else 1.0
    fairness_results["demographic_parity_ratio_gender"] = {
        "Female_vs_Male": round(gender_ratio, 4)
    }

    # 6. Save results locally
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fairness_metrics.json")
    with open(output_path, "w") as f:
        json.dump(fairness_results, f, indent=4)

    print(f"\nFairness and subgroup evaluations saved successfully to: {output_path}")


if __name__ == "__main__":
    analyze_fairness()
