"""Download a stratified subset of the official competition using Kaggle credentials."""
import argparse
import csv
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split

COMPETITION = "airbus-ship-detection"
LABEL_FILE = "train_ship_segmentations_v2.csv"


def download(root, count, workers, seed):
    images = root / "train_v2"
    images.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    label_file = root / LABEL_FILE
    if not label_file.is_file():
        api.competition_download_file(COMPETITION, LABEL_FILE, str(root), unzip=True)
    with label_file.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    labels = {}
    for row in rows:
        name = row["ImageId"]
        if Path(name).name != name or not name or "\\" in name:
            raise ValueError(f"Invalid image name: {name!r}")
        encoded = (row["EncodedPixels"] or "").strip().lower()
        labels[name] = labels.get(name, False) or encoded not in {"", "nan"}
    names = sorted(labels)
    selected, _ = train_test_split(
        names, train_size=count, random_state=seed,
        stratify=[labels[name] for name in names],
    )
    local = threading.local()

    def fetch(name):
        if (images / name).is_file():
            return
        if not hasattr(local, "api"):
            local.api = KaggleApi()
            local.api.authenticate()
        local.api.competition_download_file(
            COMPETITION, f"train_v2/{name}", str(root), quiet=True, unzip=True,
        )
        if not (images / name).is_file():
            raise FileNotFoundError(images / name)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, _ in enumerate(pool.map(fetch, selected), 1):
            if index % 25 == 0 or index == count:
                print(f"Downloaded {index}/{count} images", flush=True)
    subset = set(selected)
    with (root / "subset_labels.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("ImageId", "EncodedPixels"))
        writer.writeheader()
        writer.writerows(row for row in rows if row["ImageId"] in subset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/airbus"))
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    download(args.root, args.count, args.workers, args.seed)
