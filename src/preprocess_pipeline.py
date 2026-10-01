import os
import json
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from data_utils import load_clean_xy, split, column_types


def run_preprocessing():
    print("[INFO] Starting Sklearn Pipeline Preprocessing...")

    # 1. Load + clean ('unknown' already converted to NaN for the pipeline to handle)
    X, y = load_clean_xy()

    # 2. Train-Test Split (done before transformations to prevent leakage)
    X_train, X_test, y_train, y_test = split(X, y)

    # 3. Separate column types
    num_cols, cat_cols = column_types(X_train)

    # 4. Modular scikit-learn pipelines
    # Numerical: median-impute, then scale
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical: mode-impute, then one-hot encode
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )

    # 5. Execute pipeline
    print("[INFO] Fitting and transforming training data...")
    X_train_final = preprocessor.fit_transform(X_train)

    print("[INFO] Transforming test data...")
    X_test_final = preprocessor.transform(X_test)

    # 6. Save artifacts
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.save("data/processed/X_train_final.npy", X_train_final)
    np.save("data/processed/X_test_final.npy", X_test_final)
    np.save("data/processed/y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save("data/processed/y_test.npy", y_test.to_numpy(dtype=np.int64))

    # Single preprocessor artifact instead of separate scaler/ohe
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    metadata = {
        "dataset_name": "Bank Marketing - Term Deposit Subscription",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
    }
    with open("data/processed/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("[SUCCESS] Preprocessing completed successfully! Pipeline saved as preprocessor.pkl.")


if __name__ == "__main__":
    run_preprocessing()
