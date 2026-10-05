# src/eval/metrics.py
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import average_precision_score


def regression_metrics(y_true, y_pred):
    """Spearman and Pearson correlation — standard for on-target efficiency evaluation."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    spearman_corr, _ = spearmanr(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    return {"spearman": spearman_corr, "pearson": pearson_corr}


def classification_metrics(y_true, y_scores):
    """AUPRC — the correct metric for imbalanced off-target classification,
    not plain accuracy or AUC-ROC (which can look misleadingly good on imbalanced data)."""
    y_true = np.asarray(y_true).flatten()
    y_scores = np.asarray(y_scores).flatten()
    auprc = average_precision_score(y_true, y_scores)
    return {"auprc": auprc}


if __name__ == "__main__":
    # Sanity check with dummy values
    y_true_reg = [0.1, 0.5, 0.9, 0.3, 0.7]
    y_pred_reg = [0.2, 0.4, 0.85, 0.35, 0.6]
    print("Regression metrics:", regression_metrics(y_true_reg, y_pred_reg))

    y_true_cls = [0, 1, 0, 0, 1]
    y_scores_cls = [0.1, 0.8, 0.2, 0.3, 0.6]
    print("Classification metrics:", classification_metrics(y_true_cls, y_scores_cls))