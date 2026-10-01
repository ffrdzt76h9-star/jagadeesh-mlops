import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
)

from data_utils import load_clean_xy, split


def run_evaluation():
    print("Starting Model Evaluation...")

    # 1. Load test data and the best model
    X_test_final = np.load("data/processed/X_test_final.npy")
    y_test = np.load("data/processed/y_test.npy")
    model = joblib.load("models/best_model.pkl")
    try:
        with open("models/best_model_name.txt") as f:
            best_name = f.read().strip()
    except FileNotFoundError:
        best_name = type(model).__name__

    # 2. Predictions
    y_pred = model.predict(X_test_final)

    # 3. Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n--- Model Evaluation Report ({best_name}) ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("-------------------------------\n")

    # 4. Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["No Subscription", "Subscribed"]
    )
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig("artifacts/confusion_matrix.png", dpi=120)
    plt.close()

    # 5. Error analysis export (rebuild the same split to recover the original client rows)
    X, y = load_clean_xy()
    _, X_test_raw, _, _ = split(X, y)

    errors_df = X_test_raw.copy()
    errors_df["Actual_Subscribed"] = y_test
    errors_df["Predicted_Subscribed"] = y_pred

    # False negative = missed a client who would have subscribed; false positive = wasted call
    false_negatives = errors_df[(errors_df["Actual_Subscribed"] == 1) & (errors_df["Predicted_Subscribed"] == 0)]
    false_positives = errors_df[(errors_df["Actual_Subscribed"] == 0) & (errors_df["Predicted_Subscribed"] == 1)]

    false_negatives.to_csv("outputs/false_negatives.csv", index=False)
    false_positives.to_csv("outputs/false_positives.csv", index=False)

    print("Evaluation complete and error analysis files saved!")


if __name__ == "__main__":
    run_evaluation()
