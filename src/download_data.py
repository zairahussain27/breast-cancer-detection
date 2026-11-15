"""Download and validate the original UCI WDBC dataset."""
from __future__ import annotations

from io import BytesIO
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd

from src import config

UCI_ARCHIVE_URL = (
    "https://archive.ics.uci.edu/static/public/17/"
    "breast+cancer+wisconsin+diagnostic.zip"
)
RAW_COLUMNS = ["id", "diagnosis", *config.FEATURES]


def download_dataset(destination=config.DATA_PATH) -> None:
    """Fetch official UCI archive, validate expected WDBC structure and save CSV."""
    request = Request(UCI_ARCHIVE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        archive_bytes = response.read()
    with ZipFile(BytesIO(archive_bytes)) as archive:
        frame = pd.read_csv(archive.open("wdbc.data"), header=None)

    if frame.shape != (569, 32):
        raise ValueError(f"Expected original UCI WDBC shape (569, 32), got {frame.shape}")
    frame.columns = RAW_COLUMNS
    if set(frame["diagnosis"]) != {"B", "M"}:
        raise ValueError("Unexpected diagnosis labels in UCI WDBC archive")
    if frame[config.FEATURES].isna().any().any():
        raise ValueError("UCI WDBC numeric features contain missing values")

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"Saved {len(frame)} records and {len(config.FEATURES)} features to {destination}")


if __name__ == "__main__":
    download_dataset()
