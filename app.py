"""Streamlit app: Breast Cancer Detection System (WDBC).

Every number shown comes from files written by ``python -m src.train``
(metrics.json, model_comparison.csv, the saved pipeline). Nothing is hardcoded.
"""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import config

st.set_page_config(page_title="Breast Cancer Detection System", page_icon="\U0001fa7a",
                   layout="wide", initial_sidebar_state="expanded")

INK, RED, GREEN, AMBER = "#12343B", "#B3261E", "#1F7A5C", "#C27C0E"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,700&family=Public+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button, label {{ font-family: 'Public Sans', sans-serif; }}
h1, h2, h3 {{ font-family: 'Newsreader', Georgia, serif !important; color: {INK}; letter-spacing: -0.01em; }}
.block-container {{ padding-top: 2rem; max-width: 1180px; }}
.hero {{ background: {INK}; color: #F6F8F7; padding: 2.6rem 2.2rem; border-radius: 6px; margin-bottom: 1.4rem; }}
.hero h1 {{ color: #F6F8F7; font-size: clamp(1.9rem, 4vw, 3rem); margin: 0 0 .6rem 0; line-height: 1.1; }}
.hero p {{ color: #CFE0DC; font-size: 1.05rem; max-width: 46rem; margin: 0; line-height: 1.55; }}
.stat {{ border-left: 4px solid {INK}; padding: .2rem 0 .2rem .9rem; margin-bottom: .6rem; }}
.stat .v {{ font-family: 'Newsreader', serif; font-size: 2rem; font-weight: 700; color: {INK}; line-height: 1.1; }}
.stat .l {{ font-size: .86rem; color: #4a5f63; }}
.result {{ border-radius: 6px; padding: 1.4rem 1.6rem; color: white; margin: .6rem 0 1rem 0; }}
.result.mal {{ background: {RED}; }} .result.ben {{ background: {GREEN}; }}
.result .t {{ font-family: 'Newsreader', serif; font-size: 2.1rem; font-weight: 700; line-height: 1.1; }}
.result .s {{ opacity: .95; font-size: 1rem; margin-top: .3rem; }}
.meter {{ position: relative; height: 14px; border-radius: 7px; margin: 1.6rem 0 .3rem 0;
         background: linear-gradient(90deg, {GREEN}, #E7E3D4 50%, {RED}); }}
.meter .pin {{ position: absolute; top: -5px; width: 4px; height: 24px; background: {INK}; border-radius: 2px; }}
.meter .thr {{ position: absolute; top: -9px; width: 0; height: 0; border-left: 6px solid transparent;
              border-right: 6px solid transparent; border-top: 8px solid #555; }}
.note {{ background: #FFF6E5; border-left: 4px solid {AMBER}; padding: .8rem 1rem; border-radius: 4px; font-size: .92rem; }}
.small {{ font-size: .85rem; color: #4a5f63; }}
@media (max-width: 640px) {{ .hero {{ padding: 1.6rem 1.2rem; }} .result .t {{ font-size: 1.6rem; }} }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

DISCLAIMER = ("This tool is an educational machine-learning project. It is **not** a medical "
              "device and must not be used to diagnose, treat or rule out cancer. Always "
              "consult a qualified healthcare professional.")


# ------------------------------------------------------------------ loaders
@st.cache_resource
def load_artifacts():
    """Load compatible artifacts; return a user-readable error when unavailable."""
    if not (config.MODEL_PATH.exists() and config.METADATA_PATH.exists()):
        return None, None, "No trained model artifacts were found."
    try:
        meta = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
        import sklearn

        expected = meta.get("sklearn_version")
        if expected and sklearn.__version__ != expected:
            return None, meta, (
                f"This model was saved with scikit-learn {expected}, but the active "
                f"environment has {sklearn.__version__}. Activate the project .venv, "
                "install requirements, then launch with `python -m streamlit run app.py`."
            )
        return joblib.load(config.MODEL_PATH), meta, None
    except Exception as exc:
        return None, None, (
            f"Could not load the model in this Python environment ({type(exc).__name__}: "
            f"{exc}). Activate the project .venv, install requirements, then launch "
            "with `python -m streamlit run app.py`."
        )


@st.cache_data
def load_reports():
    out = {}
    mpath = config.REPORTS_DIR / "metrics.json"
    if mpath.exists():
        out["metrics"] = json.loads(mpath.read_text(encoding="utf-8"))
    for key, fname in [("comparison", "model_comparison.csv"),
                       ("importance", "feature_importance.csv")]:
        path = config.REPORTS_DIR / fname
        if path.exists():
            out[key] = pd.read_csv(path)
    return out


def show_image(name: str, caption: str | None = None):
    path = config.VIZ_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"{name} not found - run `python -m src.train` first.")


def stat(value: str, label: str):
    st.markdown(f'<div class="stat"><div class="v">{value}</div>'
                f'<div class="l">{label}</div></div>', unsafe_allow_html=True)


pipeline, meta, artifact_error = load_artifacts()
reports = load_reports()

# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("### Breast Cancer Detection System")
page = st.sidebar.radio("Navigate", ["Home", "Predict", "Model performance",
                                     "Feature insights", "About"],
                        label_visibility="collapsed")
st.sidebar.markdown(f'<div class="note"><b>Medical disclaimer.</b> {DISCLAIMER}</div>',
                    unsafe_allow_html=True)

if pipeline is None:
    st.error(artifact_error or "Model loading failed for an unknown reason.")
    st.stop()

FEATURES = meta["features"]
STATS = meta["feature_stats"]
THRESHOLD = meta["threshold"]


# --------------------------------------------------------------------- home
def page_home():
    st.markdown("""<div class="hero"><h1>Screening support for breast tumour
    measurements</h1><p>A machine-learning classifier trained on the Wisconsin Diagnostic
    Breast Cancer dataset. Enter the 30 cell-nucleus measurements of a fine-needle aspirate
    and get a benign / malignant estimate with its probability.</p></div>""",
                unsafe_allow_html=True)
    m = reports.get("metrics")
    if m:
        t = m["test"]
        c = st.columns(4)
        with c[0]:
            stat(f"{t['recall']:.1%}", "Malignant recall on held-out test set")
        with c[1]:
            stat(f"{t['precision']:.1%}", "Precision on held-out test set")
        with c[2]:
            stat(f"{t['roc_auc']:.3f}", "ROC-AUC on held-out test set")
        with c[3]:
            stat(meta["model_name"], f"Selected model, threshold {THRESHOLD:.2f}")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Project overview")
        sizes = meta["split_sizes"]
        st.markdown(f"""
- **Problem:** classify a breast tumour as malignant or benign from 30 numeric features.
- **Data:** Wisconsin Diagnostic Breast Cancer (WDBC), 569 cases (212 malignant, 357 benign).
- **Method:** three models (Logistic Regression, Random Forest, SVM) tuned with grid search
  inside leak-free pipelines; split into {sizes['train']} training, {sizes['val']} validation
  and {sizes['test']} test cases.
- **Safety first:** missing a malignant tumour is the costly error, so the decision
  threshold is set on validation data to reach at least {meta['target_recall']:.0%}
  malignant recall.
""")
    with right:
        st.subheader("Where to go next")
        st.markdown("**Predict** - enter measurements and see the result.  \n"
                    "**Model performance** - confusion matrices, ROC curves, model comparison.  \n"
                    "**Feature insights** - which measurements matter most.")
    st.markdown(f'<div class="note"><b>Medical disclaimer.</b> {DISCLAIMER}</div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------------ predict
def _init_inputs():
    for f in FEATURES:
        st.session_state.setdefault(f"in_{f}", float(STATS[f]["median"]))


def _load_preset(kind: str):
    for f in FEATURES:
        st.session_state[f"in_{f}"] = float(STATS[f][f"{kind}_median"])
    st.session_state.pop("result", None)


def _explain_in_words(contrib: pd.Series, values: pd.Series, top: int = 5):
    lines = []
    for f in contrib.abs().sort_values(ascending=False).head(top).index:
        pts = contrib[f] * 100
        direction = "towards malignant" if pts > 0 else "towards benign"
        s = STATS[f]
        lines.append(f"- **{config.display_name(f)}** = {values[f]:.4g} "
                     f"(typical benign {s['benign_median']:.4g}, typical malignant "
                     f"{s['malignant_median']:.4g}) pushes the estimate {direction} "
                     f"by about {abs(pts):.1f} percentage points.")
    return "\n".join(lines)


def page_predict():
    st.title("Predict")
    st.markdown("Enter the 30 measurements, or start from a typical case "
                "(the median of benign or malignant cases in the training data).")
    _init_inputs()
    b1, b2, _ = st.columns([1, 1, 2])
    b1.button("Load typical benign case", on_click=_load_preset, args=("benign",))
    b2.button("Load typical malignant case", on_click=_load_preset, args=("malignant",))

    tabs = st.tabs(["Mean values", "Standard error", "Worst values"])
    for tab, suffix in zip(tabs, config.SUFFIXES):
        with tab:
            cols = st.columns(3)
            feats = [f for f in FEATURES if f.endswith(f"_{suffix}")]
            for i, f in enumerate(feats):
                s = STATS[f]
                cols[i % 3].number_input(
                    config.display_name(f), min_value=0.0,
                    max_value=float(s["max"]) * 3, step=float(s["max"]) / 200,
                    format="%.5f", key=f"in_{f}",
                    help=f"Training range: {s['min']:.4g} to {s['max']:.4g}")

    values = pd.Series({f: st.session_state[f"in_{f}"] for f in FEATURES})
    outside = [config.display_name(f) for f in FEATURES
               if not STATS[f]["min"] <= values[f] <= STATS[f]["max"]]
    if outside:
        st.warning("Outside the range seen in training (predictions are less reliable): "
                   + ", ".join(outside[:6]) + ("..." if len(outside) > 6 else ""))

    if st.button("Run prediction", type="primary"):
        row = pd.DataFrame([values], columns=FEATURES)
        proba = float(pipeline.predict_proba(row)[0, 1])
        st.session_state["result"] = {"proba": proba, "row": row}

    res = st.session_state.get("result")
    if res:
        proba, row = res["proba"], res["row"]
        malignant = proba >= THRESHOLD
        conf = proba if malignant else 1 - proba
        label = "Malignant" if malignant else "Benign"
        css = "mal" if malignant else "ben"
        st.markdown(f"""<div class="result {css}"><div class="t">{label}</div>
        <div class="s">Probability of malignant: {proba:.1%} &nbsp;|&nbsp;
        confidence in this label: {conf:.1%}</div></div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="meter"><div class="pin" style="left:calc({proba*100:.1f}% - 2px)"></div>
        <div class="thr" style="left:calc({THRESHOLD*100:.1f}% - 6px)"></div></div>
        <div class="small">Dark bar = this case. Triangle = decision threshold
        ({THRESHOLD:.2f}); cases above it are labelled malignant.</div>""",
                    unsafe_allow_html=True)
        if abs(proba - THRESHOLD) < 0.10:
            st.markdown('<div class="note">This case is close to the decision boundary. '
                        'Treat the result with extra caution.</div>', unsafe_allow_html=True)
        st.caption("The threshold is deliberately below 0.5 so that fewer malignant cases "
                   "are missed; the cost is more false alarms.")

        st.subheader("Why this prediction?")
        bg = pd.read_csv(config.BACKGROUND_PATH) if config.BACKGROUND_PATH.exists() else None
        if bg is None:
            st.info("Background sample not found; explanation unavailable.")
        else:
            try:
                from src import explain
                with st.spinner("Computing SHAP explanation..."):
                    contrib, base = explain.local_contributions(pipeline, row, bg[FEATURES])
                top = contrib.reindex(contrib.abs().sort_values(ascending=False).head(8).index)
                fig, ax = plt.subplots(figsize=(6.4, 3.8))
                colors = [RED if v > 0 else GREEN for v in top.iloc[::-1]]
                top.iloc[::-1].plot.barh(ax=ax, color=colors)
                ax.set_yticklabels([config.display_name(f) for f in top.index[::-1]])
                ax.axvline(0, color="#444", lw=1)
                ax.set_xlabel("Effect on malignant probability (SHAP)")
                st.pyplot(fig, width="content")
                plt.close(fig)
                st.markdown(_explain_in_words(contrib, values))
                st.caption(f"Reference: an average training case scores about "
                           f"{base:.1%} malignant. Red bars raise the estimate, green "
                           f"bars lower it.")
            except Exception as exc:  # SHAP is optional
                st.info(f"SHAP explanation unavailable ({type(exc).__name__}).")
        st.markdown(f'<div class="note"><b>Medical disclaimer.</b> {DISCLAIMER}</div>',
                    unsafe_allow_html=True)


# -------------------------------------------------------------- performance
def page_performance():
    st.title("Model performance")
    m, comp = reports.get("metrics"), reports.get("comparison")
    if not m:
        st.info("Run `python -m src.train` to generate metrics.")
        return
    t, ci = m["test"], m["test_bootstrap_95ci"]
    st.subheader(f"Final model on the untouched test set - {m['model']}")
    cols = st.columns(5)
    for col, (k, lab) in zip(cols, [("accuracy", "Accuracy"), ("precision", "Precision"),
                                    ("recall", "Recall (malignant)"), ("f1", "F1-score"),
                                    ("roc_auc", "ROC-AUC")]):
        with col:
            stat(f"{t[k]:.3f}", f"{lab}  (95% CI {ci[k][0]:.2f}-{ci[k][1]:.2f})")
    st.markdown(f"Test confusion matrix: **{t['fn']:.0f}** missed malignant case(s), "
                f"**{t['fp']:.0f}** false alarm(s), out of "
                f"{t['tn'] + t['fp'] + t['fn'] + t['tp']:.0f} cases. The test set is small, "
                "so the intervals are wide.")
    c1, c2 = st.columns(2)
    with c1:
        show_image("cm_test_final.png")
    with c2:
        show_image("roc_test_final.png")

    st.subheader("Model comparison (training CV and validation set)")
    if comp is not None:
        show = comp[["model", "cv_f2", "cv_roc_auc", "threshold", "val_accuracy",
                     "val_precision", "val_recall", "val_f1", "val_roc_auc", "val_fn"]]
        st.dataframe(show.round(3), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        show_image("model_comparison.png")
    with c2:
        show_image("roc_validation.png")
    st.subheader("Why the threshold is not 0.5")
    c1, c2 = st.columns([3, 2])
    with c1:
        show_image("threshold_tradeoff.png")
    with c2:
        st.markdown(f"The threshold **{THRESHOLD:.2f}** is the highest-precision cut-off on "
                    f"the validation set that still catches at least "
                    f"{meta['target_recall']:.0%} of malignant cases. Lowering the "
                    "threshold finds more cancers but raises false alarms.")
    st.subheader("Validation confusion matrices")
    cols = st.columns(3)
    for col, name in zip(cols, ["logistic_regression", "random_forest", "svm"]):
        with col:
            show_image(f"cm_val_{name}.png")


# ------------------------------------------------------------------ insights
def page_insights():
    st.title("Feature insights")
    st.markdown("Each of the 30 features describes the cell nuclei in the sample: its "
                "**mean**, **standard error** and **worst** (mean of the three largest) "
                "value for ten measurements such as radius, texture and concavity.")
    imp = reports.get("importance")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Permutation importance")
        show_image("permutation_importance.png")
        st.caption("Shuffle one feature on validation data and see how much the model "
                   "gets worse. Correlated features share credit, so rankings are approximate.")
    with c2:
        st.subheader("SHAP")
        if meta.get("shap_available") and (config.VIZ_DIR / "shap_summary.png").exists():
            show_image("shap_summary.png")
            st.caption("Average size of each feature's effect on the malignant probability.")
        else:
            st.info("SHAP summary not generated.")
    if imp is not None:
        imp = imp.assign(feature=imp["feature"].map(config.display_name))
        st.dataframe(imp.round(4).head(15), hide_index=True, width="stretch")
    st.markdown("""
**Reading the features in plain language**
- *Radius, perimeter, area* describe nucleus size. They are strongly correlated with each other.
- *Concavity and concave points* describe how indented the nucleus outline is; irregular
  outlines are more typical of malignant cells.
- *Texture* is the variation in grey-scale values; *symmetry* and *fractal dimension*
  describe shape regularity and edge complexity.

These are statistical associations in this dataset, not causal medical explanations.
""")


# --------------------------------------------------------------------- about
def page_about():
    st.title("About")
    st.markdown(f"""
**Dataset.** Wisconsin Diagnostic Breast Cancer (Street, Wolberg & Mangasarian, 1993),
available from the UCI Machine Learning Repository and bundled with scikit-learn.

**Method.** Stratified train / validation / test split (seed {meta['seed']}); scaling inside
scikit-learn Pipelines to prevent leakage; GridSearchCV on training data; model and
threshold chosen on validation data; test set used once.

**Limitations.**
- 569 samples from a single source; no external validation.
- Inputs are measurements extracted from images, not the images themselves.
- Small validation/test sets give wide uncertainty on every metric.
- A probability from a model is not a clinical probability.
""")
    st.markdown(f'<div class="note"><b>Medical disclaimer.</b> {DISCLAIMER}</div>',
                unsafe_allow_html=True)


{"Home": page_home, "Predict": page_predict, "Model performance": page_performance,
 "Feature insights": page_insights, "About": page_about}[page]()

