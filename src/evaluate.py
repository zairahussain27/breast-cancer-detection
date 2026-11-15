"""Metrics, threshold selection, plots, false-negative analysis and report.

Positive class = Malignant (1). Plots use one consistent palette.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # headless backend; must be set before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             fbeta_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from src import config  # noqa: E402

BENIGN_COLOR = "#1F7A5C"
MALIGNANT_COLOR = "#B3261E"
INK = "#12343B"
MODEL_COLORS = {"Logistic Regression": "#12343B", "Random Forest": "#C27C0E",
                "SVM": "#5B6FA8"}


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """All headline metrics at a given decision threshold."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "f1": f1_score(y_true, pred, zero_division=0),
        "f2": fbeta_score(y_true, pred, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def select_threshold(y_true, proba, target_recall: float = config.TARGET_RECALL):
    """Pick a threshold on *validation* data.

    Rule: among thresholds on a 0.01 grid whose malignant recall is at least
    ``target_recall``, take the one with the highest precision (ties -> the
    lower, i.e. more recall-friendly, threshold). If no threshold reaches the
    target, fall back to the threshold with the best recall.
    """
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    rows = [compute_metrics(y_true, proba, t) for t in grid]
    table = pd.DataFrame(rows)
    ok = table[table["recall"] >= target_recall]
    if ok.empty:
        best = table.sort_values(["recall", "precision"], ascending=False).iloc[0]
    else:
        best = ok.sort_values(["precision", "threshold"], ascending=[False, True]).iloc[0]
    return float(best["threshold"]), table


def bootstrap_ci(y_true, proba, threshold, n_boot: int = 1000,
                 seed: int = config.SEED) -> dict:
    """95 % percentile bootstrap intervals for the test metrics."""
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = compute_metrics(y_true[idx], proba[idx], threshold)
        for k in keys:
            draws[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in draws.items()}


# ----------------------------------------------------------------- plots
def _save(fig, name: str) -> str:
    config.VIZ_DIR.mkdir(parents=True, exist_ok=True)
    path = config.VIZ_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def plot_confusion_matrix(y_true, proba, threshold, title, name):
    pred = (np.asarray(proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    labels = ["Benign", "Malignant"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{title}\n(threshold = {threshold:.2f})", fontsize=11, color=INK)
    for i in range(2):
        for j in range(2):
            bad = (i, j) in [(1, 0), (0, 1)]
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=18,
                    fontweight="bold",
                    color=MALIGNANT_COLOR if bad and cm[i, j] else
                    ("white" if cm[i, j] > cm.max() / 2 else INK))
    ax.grid(False)
    return _save(fig, name)


def plot_roc_curves(y_true, proba_by_model: dict, title, name, markers=None):
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for model, proba in proba_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, lw=2, color=MODEL_COLORS.get(model, INK),
                label=f"{model} (AUC = {auc:.3f})")
    if markers:
        for model, thr in markers.items():
            m = compute_metrics(y_true, proba_by_model[model], thr)
            ax.scatter(1 - m["specificity"], m["recall"], s=60, zorder=5,
                       color=MODEL_COLORS.get(model, INK), edgecolor="white")
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (malignant recall)")
    ax.set_title(title, color=INK)
    ax.legend(loc="lower right", frameon=False)
    return _save(fig, name)


def plot_threshold_tradeoff(table: pd.DataFrame, chosen: float, name):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(table["threshold"], table["recall"], color=MALIGNANT_COLOR, lw=2,
            label="Malignant recall")
    ax.plot(table["threshold"], table["precision"], color=INK, lw=2, label="Precision")
    ax.plot(table["threshold"], table["specificity"], color=BENIGN_COLOR, lw=2,
            label="Specificity")
    ax.axvline(chosen, color="#666666", ls="--")
    ax.axhline(config.TARGET_RECALL, color=MALIGNANT_COLOR, ls=":", lw=1)
    ax.text(chosen + 0.01, 0.05, f"chosen = {chosen:.2f}", fontsize=9)
    ax.set_xlabel("Decision threshold (probability of malignant)")
    ax.set_ylabel("Validation score")
    ax.set_ylim(0, 1.03)
    ax.set_title("Threshold trade-off on the validation set", color=INK)
    ax.legend(loc="lower center", frameon=False, ncol=3)
    return _save(fig, name)


def plot_model_comparison(comparison: pd.DataFrame, name):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.8 / len(comparison)
    for i, (_, row) in enumerate(comparison.iterrows()):
        vals = [row[f"val_{m}"] for m in metrics]
        ax.bar(np.arange(len(metrics)) + i * width, vals, width,
               color=MODEL_COLORS.get(row["model"], INK), label=row["model"])
    ax.set_xticks(np.arange(len(metrics)) + width * (len(comparison) - 1) / 2,
                  ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.set_ylim(0.8, 1.01)
    ax.set_title("Validation performance at each model's tuned threshold", color=INK)
    ax.legend(frameon=False, loc="lower right", ncol=3)
    return _save(fig, name)


def plot_importance(importance: pd.DataFrame, name, top: int = 12):
    top_df = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.barh([config.display_name(f) for f in top_df["feature"]],
            top_df["importance"], color=INK)
    ax.set_xlabel("Increase in validation log-loss when the feature is shuffled")
    ax.set_title("Permutation importance (final model)", color=INK)
    return _save(fig, name)


# ------------------------------------------------- false negative analysis
def false_negative_analysis(X, y_true, proba, threshold, key_features) -> dict:
    """Describe missed malignant cases and the closest calls."""
    df = X.copy()
    df["actual"] = np.asarray(y_true)
    df["malignant_probability"] = np.asarray(proba)
    fn = df[(df["actual"] == 1) & (df["malignant_probability"] < threshold)]
    mal = df[df["actual"] == 1]
    near = mal.sort_values("malignant_probability").head(5)
    rows = []
    for f in key_features:
        rows.append({
            "feature": f,
            "benign_median": float(df.loc[df["actual"] == 0, f].median()),
            "malignant_median": float(mal[f].median()),
            "false_negative_median": float(fn[f].median()) if len(fn) else float("nan"),
        })
    return {"false_negatives": fn, "closest_calls": near,
            "profile": pd.DataFrame(rows)}


# ---------------------------------------------------------------- report
def _pct(x):
    return f"{100 * x:.1f}%"


def generate_report(results: dict) -> str:
    """Write reports/model_report.md from computed results only."""
    comp: pd.DataFrame = results["comparison"]
    final = results["final_model"]
    t = results["test_metrics"]
    ci = results["test_ci"]
    v = results["val_metrics_final"]
    fna = results["fn_analysis"]
    n_test = t["tn"] + t["fp"] + t["fn"] + t["tp"]
    n_mal = t["fn"] + t["tp"]

    lines = [
        "# Model report - Breast Cancer Detection",
        "",
        "_All numbers below are produced by `python -m src.train` "
        f"(random seed {config.SEED}). Re-running regenerates this file._",
        "",
        "## 1. Data and protocol",
        "",
        f"- Samples after cleaning: {results['data_report']['clean_shape'][0]} "
        f"({results['data_report']['class_counts']}).",
        f"- Stratified split: train {results['sizes']['train']}, validation "
        f"{results['sizes']['val']}, test {results['sizes']['test']}.",
        "- Scaling is inside each scikit-learn Pipeline, so it is fitted on training "
        "folds only (no leakage).",
        f"- Hyper-parameters tuned with GridSearchCV ({config.CV_FOLDS}-fold "
        "stratified CV, training set only), optimising F2 (recall-weighted).",
        f"- The decision threshold is chosen on the validation set: the highest-precision "
        f"threshold with malignant recall >= {config.TARGET_RECALL:.2f}.",
        "- The test set is used once, for the final model only.",
        "",
        "## 2. Model comparison (training CV and validation)",
        "",
        comp_to_markdown(comp),
        "",
        "![Validation comparison](../assets/visualizations/model_comparison.png)",
        "",
        "![Validation ROC curves](../assets/visualizations/roc_validation.png)",
        "",
        "Validation confusion matrices (each model at its own tuned threshold):",
        "",
    ]
    for m in comp["model"]:
        slug = m.lower().replace(" ", "_")
        lines.append(f"![{m}](../assets/visualizations/cm_val_{slug}.png)")
    lines += [
        "",
        f"**Selected model: {final}.** Rule: highest validation precision subject to "
        "recall >= target, ties broken by validation ROC-AUC. With only "
        f"{results['sizes']['val']} validation cases, differences between models are small "
        "and should not be over-interpreted.",
        "",
        "## 3. Threshold selection",
        "",
        f"Chosen threshold: **{results['threshold']:.2f}** (default would be 0.50). "
        f"Validation at this threshold: recall {_pct(v['recall'])}, precision "
        f"{_pct(v['precision'])}.",
        "",
        "![Threshold trade-off](../assets/visualizations/threshold_tradeoff.png)",
        "",
        "## 4. Final evaluation on the untouched test set",
        "",
        "| Metric | Value | 95% bootstrap CI |",
        "|---|---|---|",
    ]
    for k, label in [("accuracy", "Accuracy"), ("precision", "Precision"),
                     ("recall", "Recall (malignant)"), ("f1", "F1-score"),
                     ("roc_auc", "ROC-AUC")]:
        lines.append(f"| {label} | {t[k]:.3f} | {ci[k][0]:.3f} - {ci[k][1]:.3f} |")
    lines += [
        "",
        f"Confusion matrix (n = {n_test}): TN = {t['tn']}, FP = {t['fp']}, "
        f"FN = {t['fn']}, TP = {t['tp']}.",
        "",
        "![Test confusion matrix](../assets/visualizations/cm_test_final.png)",
        "",
        "![Test ROC curve](../assets/visualizations/roc_test_final.png)",
        "",
        f"The test set has only {n_mal} malignant cases, so each missed case changes "
        "recall by several percentage points; the bootstrap intervals show how wide "
        "the uncertainty really is.",
        "",
        "## 5. False-negative analysis (test set)",
        "",
    ]
    fn = fna["false_negatives"]
    if len(fn) == 0:
        lines.append("No malignant case was missed at the chosen threshold on the test "
                     "set. The five closest calls (lowest predicted malignant "
                     "probability among true malignant cases) were:")
    else:
        lines.append(f"{len(fn)} malignant case(s) were missed. Their predicted "
                     "malignant probabilities: "
                     + ", ".join(f"{p:.3f}" for p in fn["malignant_probability"]) + ".")
        lines.append("")
        lines.append("Closest calls (including the misses):")
    lines += ["", closest_to_markdown(fna["closest_calls"]), "",
              "Median values of the most important features:", "",
              profile_to_markdown(fna["profile"]), ""]
    lines += [
        "Interpretation: compare the missed-case column with the benign and malignant "
        "medians. Values nearer the benign median indicate an atypical malignant "
        "presentation that the model finds hard to separate. A false negative is the "
        "costly error in screening, which is why the threshold is tuned for recall "
        "rather than accuracy.",
        "",
        "## 6. Feature importance",
        "",
        "![Permutation importance](../assets/visualizations/permutation_importance.png)",
        "",
    ]
    if results.get("shap_ok"):
        lines += ["![SHAP summary](../assets/visualizations/shap_summary.png)", ""]
    lines += [
        "## 7. Limitations",
        "",
        "- 569 samples from one source; no external validation set.",
        "- Features are lab measurements from fine-needle-aspirate images, not raw images.",
        "- Threshold and model were selected on a small validation set.",
        "- Educational project - not a medical device.",
        "",
    ]
    path = config.REPORTS_DIR / "model_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def comp_to_markdown(comp: pd.DataFrame) -> str:
    cols = [("model", "Model"), ("cv_f2", "CV F2"), ("cv_roc_auc", "CV ROC-AUC"),
            ("threshold", "Threshold"), ("val_accuracy", "Val acc"),
            ("val_precision", "Val prec"), ("val_recall", "Val recall"),
            ("val_f1", "Val F1"), ("val_roc_auc", "Val ROC-AUC"), ("val_fn", "Val FN")]
    head = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "---|" * len(cols)
    body = []
    for _, r in comp.iterrows():
        cells = []
        for key, _ in cols:
            val = r[key]
            cells.append(val if isinstance(val, str) else
                         (f"{int(val)}" if key == "val_fn" else f"{val:.3f}"))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + body)


def closest_to_markdown(df: pd.DataFrame) -> str:
    rows = ["| Sample index | Malignant probability |", "|---|---|"]
    for idx, r in df.iterrows():
        rows.append(f"| {idx} | {r['malignant_probability']:.3f} |")
    return "\n".join(rows)


def profile_to_markdown(df: pd.DataFrame) -> str:
    rows = ["| Feature | Benign median | Malignant median | Missed-case median |",
            "|---|---|---|---|"]
    for _, r in df.iterrows():
        fnm = "n/a" if np.isnan(r["false_negative_median"]) else f"{r['false_negative_median']:.4g}"
        rows.append(f"| {config.display_name(r['feature'])} | {r['benign_median']:.4g} | "
                    f"{r['malignant_median']:.4g} | {fnm} |")
    return "\n".join(rows)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
