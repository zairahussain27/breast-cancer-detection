# Local validation report

Validated locally on 2026-11-15 with Python 3.12.0 and the system interpreter. No GitHub remote,
commit, or deployment was created.

| Requirement | Status | Evidence |
|---|---|---|
| WDBC dataset | PASS | Original UCI archive downloaded; 569 rows, 30 numeric features, B=357 and M=212; no missing values or duplicate feature rows. Feature values and labels match scikit-learn's bundled copy exactly. |
| Preprocessing | PASS | Target excluded from 30 inputs; B maps to 0 and M to 1; stratified disjoint 60/20/20 split; scaler is fitted inside each training pipeline/fold. |
| EDA | PASS | Notebook includes overview, missingness, class chart, distributions, boxplots, correlations and class comparisons; saved figures are in `assets/visualizations/`. |
| Three ML models | PASS | Logistic Regression, Random Forest and RBF SVM trained and compared. |
| Hyperparameter tuning | PASS | 5-fold stratified GridSearchCV optimizing F2 on training data only. |
| Test evaluation | PASS | Test set used after selection; metrics and predictions saved under `reports/`. SVM: accuracy .930, precision .854, malignant recall .976, F1 .911, ROC-AUC .995; TN 65, FP 7, FN 1, TP 41. |
| Threshold selection | PASS | Threshold .18 selected on validation to maximize precision subject to recall >= .98; threshold trade-off figure saved. |
| Explainable AI | PARTIAL | Permutation importance is generated from validation data. SHAP is an optional dependency separated into `requirements-explain.txt`; it was not available during final training. The old image is hidden unless current metadata confirms SHAP output. |
| Streamlit application | PASS | Server launched at localhost:8501 and `/_stcore/health` returned `ok`; Streamlit AppTest opened Predict, submitted all default inputs and displayed the probability result with no app exceptions. |
| Jupyter Notebook | PASS | Executed top to bottom using nbconvert; execution succeeded. Windows emitted one pyzmq Proactor event-loop runtime warning. |
| Automated tests | PASS | `python -m pytest -v`: 11 passed, 0 failed, 0 skipped; no test warnings. |
| README | PASS | Includes source, download, install, training, notebook, app, evaluation, limitations and medical disclaimer; removed placeholder repository/deployment instructions. |
| Reproducibility | PARTIAL | Fixed seeds, data downloader, version pins, single-worker tuning, and artifact metadata are present. The provided PowerShell transcript confirms the clean `.venv` install failed because DNS could not resolve pypi.org; validation was completed with the system interpreter. |

## Artifact metrics

The final model is an RBF SVM with threshold 0.18. Test metrics are calculated from the untouched
114-row test split. The confusion matrix has one false negative; this demonstrates why the
experimental model cannot rule out disease. Bootstrap intervals are in `reports/metrics.json`.

## Remaining validation

- Restore DNS/network access to `pypi.org`, install `requirements-dev.txt` into `.venv`, then rerun commands with `python -m ...` from that environment.
- When downloads complete reliably, install `requirements-explain.txt`, retrain to generate current SHAP output, and verify the SHAP plot and metadata.
