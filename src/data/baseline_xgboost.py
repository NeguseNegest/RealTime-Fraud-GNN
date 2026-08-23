
from sklearn.metrics import (average_precision_score,precision_score,recall_score,f1_score,confusion_matrix,
)
from xgboost import XGBClassifier

from src.data.loader import load_elliptic_data


Found_treshold = 0.20 # THE TRESHOLD FOUND IN EXPLORATORY NOTEBOOK, I WILL USE IT AS A BASELINE FOR THE ALERT SYSTEM.


def train_baseline(data):
    X_train = data.x[data.train_mask].numpy()
    y_train = data.y[data.train_mask].numpy()

    model = XGBClassifier(objective="binary:logistic",eval_metric="logloss",random_state=42,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_baseline(model, data, threshold=Found_treshold):
    X_test = data.x[data.test_mask].numpy()
    y_test = data.y[data.test_mask].numpy()

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {"pr_auc": average_precision_score(y_test, y_prob), "precision": precision_score(y_test, y_pred),
               "recall": recall_score(y_test, y_pred),
               "f1": f1_score(y_test, y_pred),
               "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    return metrics


if __name__ == "__main__":
    data = load_elliptic_data()

    model = train_baseline(data)

    metrics = evaluate_baseline(model, data)

    print("Training complete")
    print(f"Test PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Threshold:   {ALERT_THRESHOLD:.2f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1:          {metrics['f1']:.4f}")
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])