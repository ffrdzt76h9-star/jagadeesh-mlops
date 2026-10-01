import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from model_zoo import get_models, fit_model


def run_tracked_training():
    print("[INFO] --- Starting MLflow Experiment Tracking ---")
    mlflow.set_experiment("Bank_Marketing_Prediction")

    X_train = np.load("data/processed/X_train_final.npy")
    X_test = np.load("data/processed/X_test_final.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    for name, model in get_models().items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_family", name)
            mlflow.log_params(model.get_params())

            fit_model(name, model, X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(sk_model=model, name="model")
            print(f"[INFO] {name:<22} F1={metrics['f1_score']:.4f} | ROC-AUC={metrics['roc_auc']:.4f}")

    print("[SUCCESS] All runs logged. View them with: mlflow ui")


if __name__ == "__main__":
    run_tracked_training()
