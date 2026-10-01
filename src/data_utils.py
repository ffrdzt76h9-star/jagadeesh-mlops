"""Shared loading/cleaning helpers so every script sees exactly the same data and split."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/bank-additional-full.csv"
TARGET_COL = "target"

# Categorical columns that use the literal string "unknown" instead of a real null
UNKNOWN_COLS = ["job", "marital", "education", "default", "housing", "loan"]
# "duration" (last call length) is only known AFTER the call ends, so it leaks the
# outcome. It is dropped, along with the raw "y" column once the target exists.
LEAKY_COLS = ["y", "duration"]


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Create target, drop duplicates, treat 'unknown' as missing, drop leaky columns."""
    df = df.copy()
    df[TARGET_COL] = (df["y"] == "yes").astype(int)
    df = df.drop_duplicates()
    for col in UNKNOWN_COLS:
        df[col] = df[col].replace("unknown", np.nan)
    return df.drop(columns=LEAKY_COLS)


def load_clean_xy(path: str = RAW_PATH):
    df = clean(load_raw(path))
    return df.drop(columns=TARGET_COL), df[TARGET_COL]


def split(X, y, seed: int = 42):
    """The one train/test split used by the whole project (80/20, stratified)."""
    return train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)


def column_types(X: pd.DataFrame):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols
