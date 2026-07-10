import numpy as np
from src.evaluation.metrics import compute_metrics


def test_compute_metrics():
    # Arrange
    y_true = np.array([0, 1, 0, 1, 0, 0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.4, 0.7, 0.2, 0.1, 0.9])

    # Act
    metrics = compute_metrics(y_true, y_prob)

    # Assert
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0

    # Test with partial incorrects
    y_prob_imperfect = np.array(
        [0.1, 0.2, 0.2, 0.8, 0.3, 0.4, 0.7, 0.2, 0.1, 0.9]
    )  # true at index 1 has low probability
    metrics_imperfect = compute_metrics(y_true, y_prob_imperfect)
    assert metrics_imperfect["accuracy"] == 0.9
    assert metrics_imperfect["recall"] == 0.75  # 3 of 4 recalled
