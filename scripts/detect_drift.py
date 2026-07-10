import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset


def detect_drift():
    print("Initializing Evidently AI post-deployment drift analysis...")

    # 1. Resolve paths
    train_path = "data/processed/train_processed.csv"
    test_path = "data/processed/test_processed.csv"
    output_dir = "dashboard"
    output_path = os.path.join(output_dir, "drift_report.html")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(
            "ERROR: Processed train/test data sets are missing. Please run 'dvc repro' first."
        )
        return

    # 2. Load reference and current datasets
    reference_df = pd.read_csv(train_path)
    current_df = pd.read_csv(test_path)

    print(f"Loaded Reference data (Train): {reference_df.shape}")
    print(f"Loaded Current data (Test/Production): {current_df.shape}")

    # 3. Compile and Run Evidently Report
    # We use DataDriftPreset to detect distribution shift and TargetDriftPreset to detect prediction/target drift
    drift_report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])

    print("Running drift metrics calculations...")
    drift_report.run(reference_data=reference_df, current_data=current_df)

    # 4. Save report as standalone HTML
    os.makedirs(output_dir, exist_ok=True)
    drift_report.save_html(output_path)
    print(f"Evidently AI Drift Report saved successfully to: {output_path}")


if __name__ == "__main__":
    detect_drift()
