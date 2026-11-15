"""Explainability: permutation importance (always) and SHAP (optional bonus).

SHAP values are computed on the *probability of malignant*, so a bar of +0.12
means "this feature raised the malignant probability by 12 percentage points
relative to an average training case".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src import config


def permutation_table(pipeline, X_val, y_val, seed: int = config.SEED) -> pd.DataFrame:
    """Global importance on validation data: rise in log-loss when a feature is shuffled."""
    res = permutation_importance(pipeline, X_val, y_val, scoring="neg_log_loss",
                                 n_repeats=30, random_state=seed, n_jobs=1)
    df = pd.DataFrame({"feature": X_val.columns, "importance": res.importances_mean,
                       "std": res.importances_std})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def _explainer(pipeline, background: pd.DataFrame):
    import shap

    cols = list(background.columns)

    def predict_malignant(data):
        return pipeline.predict_proba(pd.DataFrame(data, columns=cols))[:, 1]

    masker = shap.maskers.Independent(background, max_samples=40)
    return shap.Explainer(predict_malignant, masker, algorithm="permutation")


def shap_values(pipeline, X: pd.DataFrame, background: pd.DataFrame,
                max_evals: int = 400):
    """Return a shap ``Explanation`` (values in probability units)."""
    return _explainer(pipeline, background)(X, max_evals=max_evals)


def local_contributions(pipeline, row: pd.DataFrame, background: pd.DataFrame):
    """Per-feature contribution for one input row -> (Series, base_value)."""
    exp = shap_values(pipeline, row, background)
    return pd.Series(exp.values[0], index=row.columns), float(exp.base_values[0])


def global_shap_plot(pipeline, X_val: pd.DataFrame, background: pd.DataFrame, path):
    """Mean |SHAP| bar chart over the validation set."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exp = shap_values(pipeline, X_val, background, max_evals=200)
    mean_abs = pd.Series(np.abs(exp.values).mean(axis=0), index=X_val.columns)
    top = mean_abs.sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.barh([config.display_name(f) for f in top.index], top.values, color="#B3261E")
    ax.set_xlabel("Mean |SHAP value| (change in malignant probability)")
    ax.set_title("SHAP: what drives predictions (validation set)", color="#12343B")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
