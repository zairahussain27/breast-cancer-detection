"""Train, tune, compare and save the breast cancer classifier.

Run from the project root:  ``python -m src.train``
"""
from __future__ import annotations

import os
import time
import warnings

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Joblib worker processes need the env var (the in-process filter is not inherited).
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
warnings.filterwarnings("ignore", category=FutureWarning)

from src import config, evaluate, explain, preprocessing  # noqa: E402

PARAM_GRIDS = {
    "Logistic Regression": {
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__class_weight": [None, "balanced"],
    },
    "Random Forest": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__class_weight": [None, "balanced"],
    },
    "SVM": {
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": ["scale", 0.001, 0.01, 0.1],
        "clf__class_weight": [None, "balanced"],
    },
}


def tune_model(name, X_train, y_train):
    """GridSearchCV on the training set only; F2 weights recall over precision."""
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    scoring = {"f2": make_scorer(fbeta_score, beta=2), "roc_auc": "roc_auc",
               "recall": "recall"}
    search = GridSearchCV(preprocessing.build_pipeline(name), PARAM_GRIDS[name],
                          scoring=scoring, refit="f2", cv=cv, n_jobs=1)
    search.fit(X_train, y_train)
    return search


def main() -> None:
    t0 = time.time()
    for d in (config.MODELS_DIR, config.REPORTS_DIR, config.VIZ_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. data ----------------------------------------------------------
    X, y, data_report = preprocessing.clean_data(preprocessing.load_data())
    s = preprocessing.split_data(X, y)
    X_train, X_val, X_test = s["X_train"], s["X_val"], s["X_test"]
    y_train, y_val, y_test = s["y_train"], s["y_val"], s["y_test"]
    print(f"Data: {data_report}")
    print(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # 2. tune each model, evaluate on VALIDATION only ------------------
    fitted, val_proba, rows = {}, {}, []
    for name in PARAM_GRIDS:
        print(f"\nTuning {name} ...")
        search = tune_model(name, X_train, y_train)
        best = search.best_estimator_
        fitted[name] = best
        proba = best.predict_proba(X_val)[:, 1]
        val_proba[name] = proba
        thr, _ = evaluate.select_threshold(y_val, proba)
        m = evaluate.compute_metrics(y_val, proba, thr)
        m05 = evaluate.compute_metrics(y_val, proba, 0.5)
        i = search.best_index_
        rows.append({
            "model": name,
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "cv_f2": search.cv_results_["mean_test_f2"][i],
            "cv_roc_auc": search.cv_results_["mean_test_roc_auc"][i],
            "cv_recall": search.cv_results_["mean_test_recall"][i],
            "threshold": thr,
            **{f"val_{k}": m[k] for k in ["accuracy", "precision", "recall", "f1",
                                          "roc_auc", "fn", "fp"]},
            "val_recall_at_0.5": m05["recall"], "val_fn_at_0.5": m05["fn"],
        })
        print(f"  best params: {rows[-1]['best_params']}")
        print(f"  CV F2={rows[-1]['cv_f2']:.4f}  val AUC={m['roc_auc']:.4f}  "
              f"thr={thr:.2f}  val recall={m['recall']:.4f}  val precision={m['precision']:.4f}")
        slug = name.lower().replace(" ", "_")
        evaluate.plot_confusion_matrix(y_val, proba, thr, f"{name} (validation)",
                                       f"cm_val_{slug}.png")

    comp = pd.DataFrame(rows)

    # 3. select model on validation (never on test) ---------------------
    eligible = comp[comp["val_recall"] >= config.TARGET_RECALL]
    pool = eligible if len(eligible) else comp
    final_name = pool.sort_values(["val_precision", "val_roc_auc"],
                                  ascending=False).iloc[0]["model"]
    threshold = float(comp.loc[comp["model"] == final_name, "threshold"].iloc[0])
    print(f"\nSelected model: {final_name} (threshold {threshold:.2f})")

    evaluate.plot_model_comparison(comp, "model_comparison.png")
    evaluate.plot_roc_curves(y_val, val_proba, "ROC curves - validation set",
                             "roc_validation.png",
                             markers={r.model: r.threshold for r in comp.itertuples()})
    _, thr_table = evaluate.select_threshold(y_val, val_proba[final_name])
    evaluate.plot_threshold_tradeoff(thr_table, threshold, "threshold_tradeoff.png")

    # 4. final model, TEST set used once -------------------------------
    final = fitted[final_name]
    test_proba = final.predict_proba(X_test)[:, 1]
    test_metrics = evaluate.compute_metrics(y_test, test_proba, threshold)
    val_final = evaluate.compute_metrics(y_val, val_proba[final_name], threshold)
    test_ci = evaluate.bootstrap_ci(y_test, test_proba, threshold)
    evaluate.plot_confusion_matrix(y_test, test_proba, threshold,
                                   f"{final_name} (test)", "cm_test_final.png")
    evaluate.plot_roc_curves(y_test, {final_name: test_proba},
                             "ROC curve - untouched test set", "roc_test_final.png",
                             markers={final_name: threshold})
    rounded = {k: round(float(v), 4) for k, v in test_metrics.items()
               if k not in ("tn", "fp", "fn", "tp")}
    counts = {k: test_metrics[k] for k in ("tn", "fp", "fn", "tp")}
    print("Test metrics:", rounded, counts)

    # 5. explainability ------------------------------------------------
    importance = explain.permutation_table(final, X_val, y_val)
    importance.to_csv(config.REPORTS_DIR / "feature_importance.csv", index=False)
    evaluate.plot_importance(importance, "permutation_importance.png")
    background = X_train.sample(n=min(40, len(X_train)), random_state=config.SEED)
    background.to_csv(config.BACKGROUND_PATH, index=False)
    shap_ok = False
    try:
        explain.global_shap_plot(final, X_val, background,
                                 config.VIZ_DIR / "shap_summary.png")
        shap_ok = True
    except Exception as exc:  # SHAP is an optional bonus
        print(f"[warn] SHAP summary skipped: {exc}")

    # 6. false-negative analysis (test) --------------------------------
    key_feats = list(importance["feature"].head(5))
    fn_analysis = evaluate.false_negative_analysis(X_test, y_test, test_proba,
                                                   threshold, key_feats)
    fn_analysis["false_negatives"].to_csv(config.REPORTS_DIR / "false_negatives_test.csv")

    # 7. persist ---------------------------------------------------------
    joblib.dump(final, config.MODEL_PATH)
    stats = {}
    for f in config.FEATURES:
        stats[f] = {
            "min": float(X[f].min()), "max": float(X[f].max()),
            "median": float(X_train[f].median()),
            "benign_median": float(X_train.loc[y_train == 0, f].median()),
            "malignant_median": float(X_train.loc[y_train == 1, f].median()),
        }
    metadata = {
        "model_name": final_name, "threshold": threshold, "seed": config.SEED,
        "features": config.FEATURES, "feature_stats": stats,
        "best_params": comp.loc[comp["model"] == final_name, "best_params"].iloc[0],
        "target_recall": config.TARGET_RECALL,
        "sklearn_version": sklearn.__version__,
        "split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "class_names": config.CLASS_NAMES, "shap_available": shap_ok,
        "data_source": "UCI WDBC (local original CSV)" if config.DATA_PATH.exists()
        else "scikit-learn bundled WDBC fallback",
        "python_version": __import__("platform").python_version(),
        "numpy_version": __import__("numpy").__version__,
        "pandas_version": pd.__version__,
    }
    evaluate.save_json(metadata, config.METADATA_PATH)

    comp_out = comp.copy()
    comp_out["best_params"] = comp_out["best_params"].astype(str)
    comp_out.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    pd.DataFrame({"y_true": y_test.values, "malignant_probability": test_proba,
                  "predicted": (test_proba >= threshold).astype(int)},
                 index=X_test.index).to_csv(config.REPORTS_DIR / "test_predictions.csv")
    val_df = pd.DataFrame({"y_true": y_val.values}, index=X_val.index)
    for name, p in val_proba.items():
        val_df[name] = p
    val_df.to_csv(config.REPORTS_DIR / "validation_predictions.csv")

    metrics_json = {
        "model": final_name, "threshold": threshold,
        "validation": {k: float(v) for k, v in val_final.items()},
        "test": {k: float(v) for k, v in test_metrics.items()},
        "test_bootstrap_95ci": test_ci,
        "data_report": data_report,
    }
    evaluate.save_json(metrics_json, config.REPORTS_DIR / "metrics.json")

    evaluate.generate_report({
        "comparison": comp_out, "final_model": final_name, "threshold": threshold,
        "test_metrics": test_metrics, "test_ci": test_ci, "val_metrics_final": val_final,
        "fn_analysis": fn_analysis, "data_report": data_report, "shap_ok": shap_ok,
        "sizes": metadata["split_sizes"],
    })
    print(f"\nSaved pipeline to {config.MODEL_PATH}\nDone in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
