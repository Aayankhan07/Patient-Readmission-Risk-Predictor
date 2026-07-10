import os
import pandas as pd
import numpy as np
from src.utils.config import RAW_DATA_PATH


def generate_synthetic_data(path: str, num_samples: int = 2000):
    """Generates synthetic data matching the Kaggle diabetic dataset schema."""
    print(f"Generating {num_samples} rows of synthetic diabetic dataset...")
    np.random.seed(42)

    data = {
        "encounter_id": np.arange(100000, 100000 + num_samples),
        "patient_nbr": np.random.randint(1000000, 9000000, size=num_samples),
        "race": np.random.choice(
            ["Caucasian", "AfricanAmerican", "?", "Hispanic", "Asian", "Other"],
            size=num_samples,
            p=[0.75, 0.18, 0.02, 0.02, 0.01, 0.02],
        ),
        "gender": np.random.choice(
            ["Female", "Male", "Unknown/Invalid"],
            size=num_samples,
            p=[0.53, 0.469, 0.001],
        ),
        "age": np.random.choice(
            [
                "[0-10)",
                "[10-20)",
                "[20-30)",
                "[30-40)",
                "[40-50)",
                "[50-60)",
                "[60-70)",
                "[70-80)",
                "[80-90)",
                "[90-100)",
            ],
            size=num_samples,
        ),
        "weight": np.random.choice(
            ["?", "[75-100)", "[50-75)", "[100-125)"],
            size=num_samples,
            p=[0.97, 0.01, 0.01, 0.01],
        ),
        "admission_type_id": np.random.randint(1, 8, size=num_samples),
        "discharge_disposition_id": np.random.randint(1, 25, size=num_samples),
        "admission_source_id": np.random.randint(1, 20, size=num_samples),
        "time_in_hospital": np.random.randint(1, 15, size=num_samples),
        "payer_code": np.random.choice(
            ["?", "MC", "MD", "HM", "UN"], size=num_samples, p=[0.4, 0.3, 0.1, 0.1, 0.1]
        ),
        "medical_specialty": np.random.choice(
            [
                "?",
                "InternalMedicine",
                "Family/GeneralPractice",
                "Cardiology",
                "Surgery-General",
            ],
            size=num_samples,
            p=[0.5, 0.2, 0.15, 0.1, 0.05],
        ),
        "num_lab_procedures": np.random.randint(1, 100, size=num_samples),
        "num_procedures": np.random.randint(0, 7, size=num_samples),
        "num_medications": np.random.randint(1, 60, size=num_samples),
        "number_outpatient": np.random.randint(0, 10, size=num_samples),
        "number_emergency": np.random.randint(0, 10, size=num_samples),
        "number_inpatient": np.random.randint(0, 10, size=num_samples),
        "diag_1": np.random.choice(
            ["250.xx", "428", "276", "414", "786"], size=num_samples
        ),
        "diag_2": np.random.choice(
            ["250.xx", "428", "276", "414", "786"], size=num_samples
        ),
        "diag_3": np.random.choice(
            ["250.xx", "428", "276", "414", "786"], size=num_samples
        ),
        "number_diagnoses": np.random.randint(3, 10, size=num_samples),
        "max_glu_serum": np.random.choice(
            ["None", "Normal", ">200", ">300"],
            size=num_samples,
            p=[0.94, 0.03, 0.015, 0.015],
        ),
        "A1Cresult": np.random.choice(
            ["None", "Normal", ">7", ">8"], size=num_samples, p=[0.82, 0.05, 0.05, 0.08]
        ),
        "metformin": np.random.choice(
            ["No", "Steady", "Up", "Down"], size=num_samples, p=[0.8, 0.18, 0.01, 0.01]
        ),
        "repaglinide": np.random.choice(
            ["No", "Steady", "Up", "Down"],
            size=num_samples,
            p=[0.98, 0.015, 0.002, 0.003],
        ),
        "nateglinide": np.random.choice(
            ["No", "Steady", "Up", "Down"],
            size=num_samples,
            p=[0.99, 0.008, 0.001, 0.001],
        ),
        "chlorpropamide": np.random.choice(
            ["No", "Steady", "Up", "Down"], size=num_samples, p=[0.999, 0.001, 0.0, 0.0]
        ),
        "glimepiride": np.random.choice(
            ["No", "Steady", "Up", "Down"],
            size=num_samples,
            p=[0.95, 0.04, 0.005, 0.005],
        ),
        "acetohexamide": np.random.choice(
            ["No", "Steady"], size=num_samples, p=[0.9999, 0.0001]
        ),
        "glipizide": np.random.choice(
            ["No", "Steady", "Up", "Down"], size=num_samples, p=[0.88, 0.10, 0.01, 0.01]
        ),
        "glyburide": np.random.choice(
            ["No", "Steady", "Up", "Down"], size=num_samples, p=[0.89, 0.09, 0.01, 0.01]
        ),
        "tolbutamide": np.random.choice(
            ["No", "Steady"], size=num_samples, p=[0.999, 0.001]
        ),
        "pioglitazone": np.random.choice(
            ["No", "Steady", "Up", "Down"],
            size=num_samples,
            p=[0.92, 0.07, 0.005, 0.005],
        ),
        "rosiglitazone": np.random.choice(
            ["No", "Steady", "Up", "Down"],
            size=num_samples,
            p=[0.93, 0.06, 0.005, 0.005],
        ),
        "acarbose": np.random.choice(
            ["No", "Steady", "Up", "Down"],
            size=num_samples,
            p=[0.99, 0.009, 0.0005, 0.0005],
        ),
        "miglitol": np.random.choice(
            ["No", "Steady", "Up", "Down"],
            size=num_samples,
            p=[0.999, 0.0008, 0.0001, 0.0001],
        ),
        "troglitazone": np.random.choice(
            ["No", "Steady"], size=num_samples, p=[0.9999, 0.0001]
        ),
        "tolazamide": np.random.choice(
            ["No", "Steady", "Up"], size=num_samples, p=[0.999, 0.0009, 0.0001]
        ),
        "examide": np.random.choice(["No"], size=num_samples),
        "citoglipton": np.random.choice(["No"], size=num_samples),
        "insulin": np.random.choice(
            ["No", "Steady", "Up", "Down"], size=num_samples, p=[0.48, 0.30, 0.11, 0.11]
        ),
        "glyburide-metformin": np.random.choice(
            ["No", "Steady", "Up", "Down"],
            size=num_samples,
            p=[0.99, 0.009, 0.0005, 0.0005],
        ),
        "glipizide-metformin": np.random.choice(
            ["No", "Steady"], size=num_samples, p=[0.9999, 0.0001]
        ),
        "glimepiride-pioglitazone": np.random.choice(
            ["No", "Steady"], size=num_samples, p=[0.9999, 0.0001]
        ),
        "metformin-rosiglitazone": np.random.choice(
            ["No", "Steady"], size=num_samples, p=[0.9999, 0.0001]
        ),
        "metformin-pioglitazone": np.random.choice(
            ["No", "Steady"], size=num_samples, p=[0.9999, 0.0001]
        ),
        "change": np.random.choice(["No", "Ch"], size=num_samples, p=[0.54, 0.46]),
        "diabetesMed": np.random.choice(
            ["No", "Yes"], size=num_samples, p=[0.23, 0.77]
        ),
        "readmitted": np.random.choice(
            ["NO", ">30", "<30"], size=num_samples, p=[0.54, 0.35, 0.11]
        ),
    }

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Synthetic dataset saved to {path}.")
    return df


def load_raw_data() -> pd.DataFrame:
    """Loads raw diabetic dataset, generating synthetic data if raw file is missing."""
    if not os.path.exists(RAW_DATA_PATH):
        print(
            f"Raw data file not found at {RAW_DATA_PATH}. Fallback to generating synthetic data."
        )
        return generate_synthetic_data(RAW_DATA_PATH)

    print(f"Loading raw data from {RAW_DATA_PATH}...")
    return pd.read_csv(RAW_DATA_PATH)
