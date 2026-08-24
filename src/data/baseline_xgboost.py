"""Train and evaluate the tabular XGBoost baseline."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from src.data.loader import load_elliptic_data

from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier


alert_threshold = 0.20
threshold_candidates = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85)


def train_baseline(data):
    x_train = data.x[data.train_mask].numpy()
    y_train = data.y[data.train_mask].numpy()
    model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=1)

  
    with threadpool_limits(limits=1):
        model.fit(x_train, y_train)
    return model


def evaluate_baseline(model, data, mask_name="test_mask", threshold=alert_threshold):
    """Evaluate the baseline against one temporal split."""
    mask = getattr(data, mask_name)
    labels = data.y[mask].numpy()
    probabilities = model.predict_proba(data.x[mask].numpy())[:, 1]
    return classification_metrics(labels, probabilities, threshold)


def evaluate_thresholds(model, data, thresholds=threshold_candidates):
    """Compare candidate alert thresholds on the validation split."""
    mask = data.val_mask
    labels = data.y[mask].numpy()
    probabilities = model.predict_proba(data.x[mask].numpy())[:, 1]
    return {threshold: classification_metrics(labels, probabilities, threshold) for threshold in thresholds}


def classification_metrics(labels, probabilities, threshold):
    """Calculate PR-AUC and thresholded classification metrics."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in the interval [0, 1]")

    predictions = (probabilities >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]),
    }


def main():
    data = load_elliptic_data()
    model = train_baseline(data)
    validation_metrics = evaluate_baseline(model, data, mask_name="val_mask")
    test_metrics = evaluate_baseline(model, data)

    print("Training complete")
    print(f"Validation PR-AUC: {validation_metrics['pr_auc']:.4f}")
    print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"Threshold:   {alert_threshold:.2f}")
    print(f"Precision:   {test_metrics['precision']:.4f}")
    print(f"Recall:      {test_metrics['recall']:.4f}")
    print(f"F1:          {test_metrics['f1']:.4f}")
    print("Confusion matrix:")
    print(test_metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
