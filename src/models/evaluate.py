
import logging
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# 1. Logging configuration
# --------------------------------------------------

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 2. Evaluation function
# --------------------------------------------------

def evaluate_model():

    logger.info("Starting model evaluation")

    # --------------------------------------------------
    # 3. Define project paths
    # --------------------------------------------------

    project_root = Path(__file__).resolve().parents[2]

    model_path = project_root / "artifacts" / "best_model.joblib"

    test_data_path = (
        project_root / "data" / "processed" / "test_data.csv"
    )

    metrics_path = project_root / "artifacts" / "metrics.json"

    # --------------------------------------------------
    # 4. Load trained model
    # --------------------------------------------------

    logger.info("Loading trained model")

    model = joblib.load(model_path)

    logger.info("Model loaded successfully")

    # --------------------------------------------------
    # 5. Load test data
    # --------------------------------------------------

    logger.info("Loading test data")

    test_data = pd.read_csv(test_data_path)

    X_test = test_data["clean_comment"]
    y_test = test_data["category"]

    logger.info("Test data loaded successfully")
    logger.info("Test dataset shape: %s", test_data.shape)

    # --------------------------------------------------
    # 6. Make predictions
    # --------------------------------------------------

    logger.info("Making predictions")

    y_pred = model.predict(X_test)

    # --------------------------------------------------
    # 7. Calculate evaluation metrics
    # --------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------
    # 8. Classification report
    # --------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        labels=[-1, 0, 1],
        target_names=["Negative", "Neutral", "Positive"],
        zero_division=0
    )

    # --------------------------------------------------
    # 9. Confusion matrix
    # --------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=[-1, 0, 1]
    )

    # --------------------------------------------------
    # 10. Log evaluation results
    # --------------------------------------------------

    logger.info("Accuracy: %.4f", accuracy)
    logger.info("Precision: %.4f", precision)
    logger.info("Recall: %.4f", recall)
    logger.info("F1-score: %.4f", f1)

    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    print(matrix)

    # --------------------------------------------------
    # 11. Save metrics
    # --------------------------------------------------

    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=[-1, 0, 1],
            target_names=["Negative", "Neutral", "Positive"],
            output_dict=True,
            zero_division=0
        ),
        "confusion_matrix": matrix.tolist()
    }

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    logger.info("Metrics saved at: %s", metrics_path)
    logger.info("Model evaluation completed")

    return metrics


# --------------------------------------------------
# 12. Main execution
# --------------------------------------------------

if __name__ == "__main__":

    evaluate_model()
