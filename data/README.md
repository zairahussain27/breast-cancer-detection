# Dataset

The project uses the original Wisconsin Diagnostic Breast Cancer (WDBC) dataset from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic).
Download and validate it with `python -m src.download_data`; the resulting
`Wisconsin_Breast.csv` is intentionally git-ignored.

The source archive provides 569 cases with 30 numeric measurements, diagnosis (`M` malignant,
`B` benign), and an identifier. It has 212 malignant and 357 benign cases. The original numeric
matrix and labels were compared to scikit-learn's bundled WDBC data and matched exactly. The 30 numeric
measurements have no missing values or duplicate feature rows. The CSV contains only source
values and descriptive column names; training removes identifier and diagnosis from model inputs.

When the original file is unavailable, `src.preprocessing.load_data()` warns and falls back to
the matching WDBC copy included with scikit-learn. The project documents that fallback so the
pipeline remains runnable offline.
