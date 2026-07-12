import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
)


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Computes key classification metrics."""
    # Handle y_prob being 2D (like predict_proba results)
    if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
        y_prob = y_prob[:, 1]

    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def save_confusion_matrix_plot(y_true, y_prob, save_path, threshold=0.5):
    """Generates and saves the confusion matrix plot."""
    if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
        y_prob = y_prob[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Not Readmitted", "Readmitted"]
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def save_roc_curve_plot(y_true, y_prob, save_path):
    """Generates and saves the ROC curve plot."""
    if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
        y_prob = y_prob[:, 1]

    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    plt.title("ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def save_pr_curve_plot(y_true, y_prob, save_path):
    """Generates and saves the Precision-Recall curve plot."""
    if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
        y_prob = y_prob[:, 1]

    fig, ax = plt.subplots(figsize=(6, 6))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
