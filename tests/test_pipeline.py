"""Smoke tests: data handling, leakage safety and the saved artefacts."""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import InconsistentVersionWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, evaluate, preprocessing  # noqa: E402


@pytest.fixture(scope="module")
def data():
    X, y, report = preprocessing.clean_data(preprocessing.load_data())
    return X, y, report


def test_clean_data_shape_and_target(data):
    X, y, report = data
    assert X.shape == (569, 30)
    assert list(X.columns) == config.FEATURES
    assert set(y.unique()) == {0, 1}
    assert report["missing_values"] == 0 and report["duplicate_rows"] == 0
    assert report["target_excluded_from_features"] is True


def test_original_dataset_download_and_schema():
    if not config.DATA_PATH.exists():
        pytest.skip("download original UCI data with `python -m src.download_data`")
    raw = preprocessing.load_data()
    X, y, report = preprocessing.clean_data(raw)
    assert X.shape == (569, 30)
    assert list(X.columns) == config.FEATURES
    assert report["class_counts"] == {"Benign": 357, "Malignant": 212}
    assert set(raw[config.TARGET]) == {"B", "M"}


def test_clean_data_rejects_missing_measurements(data):
    raw = preprocessing.load_data().drop(columns=["id"]).copy()
    raw.loc[0, config.FEATURES[0]] = np.nan
    with pytest.raises(ValueError, match="Unexpected missing values"):
        preprocessing.clean_data(raw)


def test_clean_data_rejects_unknown_target_label():
    raw = preprocessing.load_data().copy()
    raw.loc[0, config.TARGET] = "X"
    with pytest.raises(ValueError, match="Unexpected diagnosis labels"):
        preprocessing.clean_data(raw)


def test_split_is_stratified_and_disjoint(data):
    X, y, _ = data
    s = preprocessing.split_data(X, y)
    idx = [set(s[f"X_{k}"].index) for k in ("train", "val", "test")]
    assert not (idx[0] & idx[1] or idx[0] & idx[2] or idx[1] & idx[2])
    assert sum(len(i) for i in idx) == len(X)
    for k in ("train", "val", "test"):
        assert abs(s[f"y_{k}"].mean() - y.mean()) < 0.03


@pytest.mark.parametrize("name", ["Logistic Regression", "Random Forest", "SVM"])
def test_scaler_is_fitted_on_training_data_only(data, name):
    X, y, _ = data
    s = preprocessing.split_data(X, y)
    pipe = preprocessing.build_pipeline(name).fit(s["X_train"], s["y_train"])
    scaler_mean = pipe.named_steps["scaler"].mean_
    assert np.allclose(scaler_mean, s["X_train"].mean().values)
    assert not np.allclose(scaler_mean, X.mean().values)


def test_threshold_selection_meets_recall_target():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(100), np.ones(60)].astype(int)
    p = np.r_[rng.beta(2, 8, 100), rng.beta(8, 2, 60)]
    thr, _ = evaluate.select_threshold(y, p, 0.95)
    assert evaluate.compute_metrics(y, p, thr)["recall"] >= 0.95


def test_saved_pipeline_predicts_and_matches_reported_metrics(data):
    if not config.MODEL_PATH.exists():
        pytest.skip("run `python -m src.train` first")
    X, y, _ = data
    s = preprocessing.split_data(X, y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipe = joblib.load(config.MODEL_PATH)
    assert not any(issubclass(item.category, InconsistentVersionWarning) for item in caught)
    meta = json.loads(config.METADATA_PATH.read_text())
    proba = pipe.predict_proba(s["X_test"])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    saved = json.loads((config.REPORTS_DIR / "metrics.json").read_text())["test"]
    now = evaluate.compute_metrics(s["y_test"], proba, meta["threshold"])
    assert now["recall"] == pytest.approx(saved["recall"])
    assert now["roc_auc"] == pytest.approx(saved["roc_auc"])


def test_pipeline_accepts_single_row_dataframe():
    if not config.MODEL_PATH.exists():
        pytest.skip("run `python -m src.train` first")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipe = joblib.load(config.MODEL_PATH)
    assert not any(issubclass(item.category, InconsistentVersionWarning) for item in caught)
    meta = json.loads(config.METADATA_PATH.read_text())
    row = pd.DataFrame([{f: meta["feature_stats"][f]["median"] for f in config.FEATURES}])
    assert pipe.predict_proba(row).shape == (1, 2)
