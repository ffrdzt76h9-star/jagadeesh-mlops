import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
)

from model_zoo import get_models, fit_model


def run_training():
    print("Starting Model Training (8-model comparison)...")

    # 1. Load processed data
    X_train = np.load("data/processed/X_train_final.npy")
    X_test = np.load("data/processed/X_test_final.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    # 2. Train every candidate and collect metrics
    results, trained, roc_data = {}, {}, {}
    for name, model in get_models().items():
        print(f"Training: {name}")
        fit_model(name, model, X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC_AUC": auc(fpr, tpr),
        }
        roc_data[name] = (fpr, tpr, results[name]["ROC_AUC"])
        trained[name] = model
        print(f"  {name:<22} F1={results[name]['F1']:.4f}  Recall={results[name]['Recall']:.4f}  ROC-AUC={results[name]['ROC_AUC']:.4f}")

    results_df = pd.DataFrame(results).T.sort_values(by="F1", ascending=False)
    results_df.to_csv("outputs/model_comparison.csv")

    # 3. Select + save the best model (by F1)
    best_name = results_df["F1"].idxmax()
    joblib.dump(trained[best_name], "models/best_model.pkl")
    with open("models/best_model_name.txt", "w") as f:
        f.write(best_name)
    print(f"\nBest model: {best_name}")

    # 4. Comparison plots
    plt.figure(figsize=(9, 6))
    for name, (fpr, tpr, a) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {a:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Comparison of ML Models")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("artifacts/roc_comparison.png", dpi=120)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(results_df.index, results_df["Accuracy"])
    plt.title("Accuracy Comparison of ML Models")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("artifacts/accuracy_comparison.png", dpi=120)
    plt.close()

    print("Model training complete and saved to disk!")


if __name__ == "__main__":
    run_training()
