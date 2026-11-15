"""Central configuration: paths, random seed, split ratios and feature names.

Every path is built relative to the project root with ``pathlib`` so the code
works unchanged on a laptop, in CI, or on Streamlit Community Cloud.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Wisconsin_Breast.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
VIZ_DIR = ROOT / "assets" / "visualizations"

MODEL_PATH = MODELS_DIR / "breast_cancer_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
BACKGROUND_PATH = MODELS_DIR / "background_sample.csv"

SEED = 42
TEST_SIZE = 0.20   # 20 % of all data, held out until the very end
VAL_SIZE = 0.20    # 20 % of all data, used for model selection + threshold
CV_FOLDS = 5
TARGET_RECALL = 0.98  # malignant recall the decision threshold must reach on validation

TARGET = "diagnosis"
LABEL_MAP = {"B": 0, "M": 1}          # Malignant is the positive class
CLASS_NAMES = {0: "Benign", 1: "Malignant"}

MEASUREMENTS = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry",
    "fractal_dimension",
]
SUFFIXES = ["mean", "se", "worst"]
FEATURES = [f"{m}_{s}" for s in SUFFIXES for m in MEASUREMENTS]

_SUFFIX_LABEL = {"mean": "mean", "se": "standard error", "worst": "worst"}


def display_name(feature: str) -> str:
    """Readable label, e.g. ``concave_points_worst`` -> ``Concave points (worst)``."""
    base, suffix = feature.rsplit("_", 1)
    return f"{base.replace('_', ' ').capitalize()} ({_SUFFIX_LABEL[suffix]})"
