"""Data loading, cleaning, stratified splitting and pipeline construction.

Leakage policy: nothing is fitted on data outside the training set. Scaling
lives *inside* a scikit-learn ``Pipeline`` so that, during cross-validation,
the scaler is re-fitted on each training fold only.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config

# scikit-learn >= 1.9 deprecates SVC(probability=True); it still works and the
# pinned version in requirements.txt keeps it stable, so silence the notice.
warnings.filterwarnings("ignore", message=".*probability.*deprecated.*",
                        category=FutureWarning)


def _from_sklearn_copy() -> pd.DataFrame:
    """Fallback: scikit-learn ships the same WDBC data (569 x 30)."""
    from sklearn.datasets import load_breast_cancer

    raw = load_breast_cancer(as_frame=True)
    rename = {}
    for col in raw.feature_names:
        if col.startswith("mean "):
            rename[col] = col[5:].replace(" ", "_") + "_mean"
        elif col.startswith("worst "):
            rename[col] = col[6:].replace(" ", "_") + "_worst"
        else:  # "<measurement> error"
            rename[col] = col[:-6].replace(" ", "_") + "_se"
    df = raw.data.rename(columns=rename)
    df[config.TARGET] = raw.target.map({0: "M", 1: "B"})  # sklearn: 0 = malignant
    return df


def load_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load WDBC from the local CSV; fall back to scikit-learn's identical copy."""
    path = Path(path) if path else config.DATA_PATH
    if path.exists():
        df = pd.read_csv(path)
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
        return df
    warnings.warn(
        f"{path} not found; using scikit-learn's bundled WDBC copy. "
        "Download the original UCI source with `python -m src.download_data`.",
        UserWarning,
        stacklevel=2,
    )
    return _from_sklearn_copy()


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Drop junk columns, check missing values / duplicates, encode the target.

    Returns ``(X, y, report)``. ``y`` is 1 for Malignant and 0 for Benign.
    """
    report: dict = {"raw_shape": list(df.shape)}
    df = df.copy()

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    id_cols = [c for c in df.columns if c.lower() == "id"]
    df = df.drop(columns=empty_cols + id_cols)
    report["dropped_columns"] = empty_cols + id_cols

    missing = df.isna().sum()
    report["missing_values"] = int(missing.sum())
    if missing.sum() > 0:
        raise ValueError(f"Unexpected missing values:\n{missing[missing > 0]}")

    report["duplicate_rows"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    unknown = set(df[config.TARGET].unique()) - set(config.LABEL_MAP)
    if unknown:
        raise ValueError(f"Unexpected diagnosis labels: {unknown}")
    y = df[config.TARGET].map(config.LABEL_MAP).astype(int)
    X = df[config.FEATURES].astype(float)

    report["clean_shape"] = list(X.shape)
    report["class_counts"] = {config.CLASS_NAMES[k]: int(v)
                              for k, v in y.value_counts().sort_index().items()}
    report["class_percentages"] = {
        config.CLASS_NAMES[k]: float(v * 100)
        for k, v in y.value_counts(normalize=True).sort_index().items()
    }
    report["target_excluded_from_features"] = config.TARGET not in X.columns
    return X, y, report


def split_data(X: pd.DataFrame, y: pd.Series, seed: int = config.SEED) -> dict:
    """Stratified train / validation / test split (60 / 20 / 20 by default)."""
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=seed)
    val_fraction = config.VAL_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_fraction, stratify=y_rest, random_state=seed)
    return {"X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test}


def build_pipeline(model_name: str, seed: int = config.SEED) -> Pipeline:
    """Scaler + classifier. All three models share the same pipeline shape."""
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=seed),
        "Random Forest": RandomForestClassifier(random_state=seed, n_jobs=-1),
        "SVM": SVC(kernel="rbf", probability=True, random_state=seed),
    }
    if model_name not in classifiers:
        raise KeyError(f"Unknown model '{model_name}'. Choose from {list(classifiers)}")
    return Pipeline([("scaler", StandardScaler()), ("clf", classifiers[model_name])])
