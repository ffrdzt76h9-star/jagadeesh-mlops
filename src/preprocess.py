import os
import json
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from data_utils import load_clean_xy, split, column_types


def run_preprocessing():
    print("Starting Preprocessing Pipeline...")

    # 1. Load + clean (target creation, dedup, 'unknown' -> NaN, drop leaky columns)
    X, y = load_clean_xy()
    print(f"Clean dataset: {X.shape[0]} rows, {X.shape[1]} features")

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = split(X, y)
    X_train, X_test = X_train.copy(), X_test.copy()

    # 3. Separate column types
    num_cols, cat_cols = column_types(X_train)

    # 4. Impute using TRAIN statistics only: median (numeric), mode (categorical)
    for col in num_cols:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)
    for col in cat_cols:
        mode = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(mode)
        X_test[col] = X_test[col].fillna(mode)

    # 5. Scale & encode
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train[num_cols])
    x_test_scaled = scaler.transform(X_test[num_cols])

    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    x_train_encoded = ohe.fit_transform(X_train[cat_cols])
    x_test_encoded = ohe.transform(X_test[cat_cols])

    X_train_final = np.hstack((x_train_scaled, x_train_encoded))
    X_test_final = np.hstack((x_test_scaled, x_test_encoded))

    # 6. Save artifacts with explicit integer type casting
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.save("data/processed/X_train_final.npy", X_train_final)
    np.save("data/processed/X_test_final.npy", X_test_final)
    np.save("data/processed/y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save("data/processed/y_test.npy", y_test.to_numpy(dtype=np.int64))

    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(ohe, "models/ohe.pkl")

    metadata = {
        "dataset_name": "Bank Marketing - Term Deposit Subscription",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
    }
    with open("data/processed/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    run_preprocessing()
