# Breast Cancer Detection System

An end-to-end machine-learning project that classifies breast tumours as **malignant** or
**benign** from the 30 cell-nucleus measurements of the Wisconsin Diagnostic Breast Cancer
(WDBC) dataset, with a Streamlit app for predictions, performance review and explanations.

> **Disclaimer:** educational project only - **not a medical device**. It must not be used
> to diagnose, treat or rule out cancer. Always consult a qualified healthcare professional.

## Problem statement

Given 30 numeric features computed from a digitised fine-needle-aspirate image, predict whether
the tumour is malignant. In screening, a missed cancer (false negative) is far more harmful than
a false alarm, so the project prioritises **malignant-class recall**, selects a decision threshold
on validation data, and reports uncertainty honestly.

## Dataset

Wisconsin Diagnostic Breast Cancer (Street, Wolberg & Mangasarian, 1993), from the UCI Machine
Learning Repository (also on Kaggle and bundled in scikit-learn). 569 cases: 357 benign,
212 malignant; 30 features (mean, standard error and worst value of 10 nucleus measurements).
The raw CSV is git-ignored (see `data/README.md`); if it is absent the code falls back to
scikit-learn's identical copy (verified equal).

## Method in brief

| Step | Decision | Reason |
|---|---|---|
| Cleaning | Drop `id` and the empty `Unnamed: 32` column; no missing values or duplicates remain | `id` has no signal; the empty column came from a trailing comma |
| Split | Stratified 60 / 20 / 20 (train / validation / test), seed 42 | Test set stays untouched until the end |
| Leakage control | `StandardScaler` inside a `Pipeline`; GridSearchCV refits it per training fold | Scaler never sees validation or test rows |
| Models | Logistic Regression, Random Forest, SVM (RBF) | The three required models |
| Tuning | GridSearchCV, 5-fold stratified CV on train only, scoring F2 | F2 weights recall twice as much as precision |
| Model + threshold | Chosen on validation: highest precision subject to malignant recall >= 0.98 | Makes "prioritise recall" an explicit rule |
| Final check | Test set used once, for the final model only, with bootstrap 95% CIs | Honest estimate on small data |

## Installation

```bash
python --version                   # Python 3.12 is the verified project version
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"  # confirm it points inside .venv
python -m pip install -r requirements-dev.txt
```

SHAP is optional because it requires a large compiled dependency on Windows. The app and core
workflow provide permutation importance without it; install SHAP separately when the network is
stable:

```powershell
python -m pip install --index-url https://pypi.org/simple --timeout 120 --retries 10 shap==0.50.0
```

Download the original UCI dataset to the ignored local `data/` directory:

```bash
python -m src.download_data
```

The official UCI archive contains 569 records, 30 floating point measurements, an integer
identifier, diagnosis labels `M` and `B`, and no missing values in its measurements. Diagnosis
is removed before model fitting (`M` maps to positive class 1, `B` to class 0); the identifier is
also dropped. The dataset has 212 malignant (37.3%) and 357 benign (62.7%) records. If the CSV is
absent, code warns and uses scikit-learn's bundled copy; the project's preferred source remains
the original UCI archive. The downloaded 569-by-30 feature matrix and labels were compared with
scikit-learn's copy and match exactly. Downloaded data is git-ignored.

## Reproduce the results

```bash
python -m src.train        # tunes models, writes models/, reports/, assets/visualizations/ (~1.5 min)
python -m pytest -v        # automated tests
```

## Run the notebook

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/breast_cancer_analysis.ipynb
```

The notebook performs the EDA, demonstrates the leakage safeguards, and loads the real results
written by `src/train.py` (it asserts that it reproduces the stored test metrics). The command
above runs nbconvert from the active interpreter.

## Run the Streamlit app

```bash
python -m streamlit run app.py
```

Using `python -m streamlit` ensures Streamlit comes from the active interpreter. If pip reports
that it cannot resolve `pypi.org`, package installation has not completed; restore network/DNS
access and rerun the install before launching the application. Do not use a global `streamlit`
or `jupyter` executable as a fallback: model artifacts must load with the pinned scikit-learn
version in the project environment.

Pages: **Home**, **Predict** (all 30 inputs, probability, optional SHAP explanation), **Model
performance**, **Feature insights**, **About**. The app only reads the saved pipeline and report
files; every number it shows comes from real training output.

## Results

Model comparison (training CV, then validation set at each model's own tuned threshold):

| Model | CV F2 | CV ROC-AUC | Threshold | Val acc | Val prec | Val recall | Val F1 | Val ROC-AUC | Val FN |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.962 | 0.993 | 0.23 | 0.886 | 0.768 | 1.000 | 0.869 | 0.995 | 0 |
| Random Forest | 0.947 | 0.988 | 0.09 | 0.877 | 0.754 | 1.000 | 0.860 | 0.993 | 0 |
| **SVM** (selected) | 0.964 | 0.993 | 0.18 | 0.912 | 0.811 | 1.000 | 0.896 | 0.997 | 0 |

Final model (SVM, threshold 0.18) on the **untouched test set** (114 cases, 42 malignant):

| Metric | Value | 95% bootstrap CI |
|---|---|---|
| Accuracy | 0.930 | 0.877 - 0.974 |
| Precision | 0.854 | 0.733 - 0.948 |
| Recall (malignant) | 0.976 | 0.921 - 1.000 |
| F1-score | 0.911 | 0.833 - 0.966 |
| ROC-AUC | 0.995 | 0.984 - 1.000 |

Confusion matrix: TN 65, FP 7, **FN 1**, TP 41. The full report with ROC curves, confusion
matrices and false-negative analysis is in [`reports/model_report.md`](reports/model_report.md).
These tables are copied from `reports/`; re-running `src.train` regenerates the source files.

### Findings

- All three models separate the classes very well (validation ROC-AUC 0.993-0.997). The
  differences between them are small and not statistically meaningful with 114 validation cases.
- **The recall target costs precision.** On validation, the default 0.5 threshold gave the SVM
  1 missed malignant case and 2 false alarms; the tuned threshold of 0.18 gave 0 missed and
  10 false alarms. With about 43 malignant validation cases, a target of 0.98 effectively
  requires perfect recall (42/43 = 0.977), which is why the threshold ended up low.
- **One malignant test case was still missed**, with a predicted malignant probability of 0.07,
  far below the threshold, so no reasonable cut-off would catch it. Its texture, symmetry and
  standard-error radius resemble benign medians (see the report's false-negative table).
- Radius, perimeter and area are almost perfectly collinear (r > 0.97), so feature-importance
  rankings are shared among correlated features and should be read as approximate.

## Project structure

```
app.py                     Streamlit application
src/config.py              paths, seed, split ratios, feature names
src/preprocessing.py       loading, cleaning, stratified split, pipelines
src/train.py               GridSearchCV, comparison, threshold, final test, saving
src/evaluate.py            metrics, plots, bootstrap CIs, false-negative analysis, report
src/explain.py             permutation importance and SHAP
notebooks/                 executed EDA and analysis notebook
models/                    fitted pipeline (.joblib), metadata, SHAP background sample
reports/                   metrics.json, comparison CSV, model_report.md, predictions
assets/visualizations/     all figures
tests/                     pytest smoke tests
```

## Artifact compatibility

The checked-in pipeline was produced with the exact scikit-learn version pinned in
`requirements.txt`. Loading joblib files across scikit-learn versions is unsupported; after
changing that pin, retrain with `python -m src.train` and validate the newly generated artifact.
This repository contains no deployment configuration or deployment claim.

## Limitations

- 569 samples from a single source; no external or prospective validation.
- Inputs are pre-computed image measurements, not the images themselves.
- Small validation and test sets give wide uncertainty (see the confidence intervals).
- The model and threshold were chosen on a small validation set.
- SHAP and permutation importance describe associations in this dataset, not medical causation.
- `SVC(probability=True)` is deprecated in scikit-learn 1.9 and slated for removal in 1.11;
  the version pin keeps this stable, and migrating to `CalibratedClassifierCV` is future work.

## Disclaimer

This software is provided for education and research demonstration only. It is not a medical
device, has not been clinically validated, and must not be used for medical decisions.

## References

Street, W. N., Wolberg, W. H. & Mangasarian, O. L. (1993). *Nuclear feature extraction for breast
tumor diagnosis.* IS&T/SPIE International Symposium on Electronic Imaging.


