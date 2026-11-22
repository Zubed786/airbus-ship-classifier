"""Evaluate saved Kaggle checkpoints and a fixed equal-weight ensemble."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

INPUT = Path("/kaggle/input")
DATA = INPUT / "competitions/airbus-ship-detection"
OUTPUT = Path("/kaggle/working")
REPO = Path("/tmp/airbus-ship-classifier")
SOURCE = "airbus-ship-classifier-15000-image-benchmark"
NAMES = ("convnextv2", "dinov3")
PILOT_SOURCE = "airbus-ship-classifier-benchmark"
ENSEMBLE_SOURCE = "airbus-ship-classifier-ensemble"


def find_output(source, filename="split.csv"):
    roots = [INPUT / "notebooks/mdzubed" / source, INPUT / source, INPUT / "mdzubed" / source]
    for root in roots:
        if (root / filename).is_file():
            return root
    raise FileNotFoundError(f"No Kaggle output {filename} for {source}")


def test_ids_digest(rows):
    return hashlib.sha256("\n".join(row["image"] for row in rows).encode()).hexdigest()


def load_splits(root, previous):
    original = (root / "split.csv").read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    for name in NAMES:
        metrics = json.loads((root / name / "metrics.json").read_text())
        if digest != metrics["split_sha256"]:
            raise ValueError(f"Split fingerprint differs from the {name} training run")
    rows = list(csv.DictReader(original.decode().splitlines()))
    excluded = {row["image"] for row in rows}
    labels = {}
    with (DATA / "train_ship_segmentations_v2.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["ImageId"]
            if name not in excluded:
                labels[name] = labels.get(name, False) or (row["EncodedPixels"] or "").strip().lower() not in {"", "nan"}
    candidates = [{"image": name, "label": int(label)} for name, label in sorted(labels.items())]
    fresh, _ = train_test_split(
        candidates, train_size=3000, random_state=2026,
        stratify=[row["label"] for row in candidates],
    )
    validation = [row for row in rows if row["split"] == "val"]
    if previous is not None:
        if previous["source_split_sha256"] != digest or previous["fresh_test_ids_sha256"] != test_ids_digest(fresh):
            raise ValueError("The earlier evaluation split could not be reproduced")
        pilot_root = find_output(PILOT_SOURCE)
        with (pilot_root / "split.csv").open(newline="") as stream:
            used = {row["image"] for row in csv.DictReader(stream)}
        used.update(row["image"] for row in fresh)
        fresh = [row for row in candidates if row["image"] not in used]
    return validation, fresh, digest


def threshold_for(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[f1.argmax()])


def metrics_for(labels, scores, threshold):
    predictions = scores >= threshold
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, predictions),
        "average_precision": average_precision_score(labels, scores),
        "f1": f1_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-unused", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
    root = find_output(SOURCE)
    previous = None
    if args.all_unused:
        earlier = find_output(ENSEMBLE_SOURCE, "ensemble-metrics.json")
        previous = json.loads((earlier / "ensemble-metrics.json").read_text())
    validation, fresh, digest = load_splits(root, previous)
    print(f"Prepared {len(validation)} validation and {len(fresh)} fresh test images", flush=True)
    subprocess.run(["git", "clone", "https://github.com/Zubed786/airbus-ship-classifier.git", str(REPO)], check=True)
    subprocess.run(["git", "-C", str(REPO), "checkout", "--detach", "2ca89b5"], check=True)
    sys.path.insert(0, str(REPO))
    from train import BACKBONES, Ships, evaluate

    scores = {}
    for name in NAMES:
        model = timm.create_model(BACKBONES[name], pretrained=False, num_classes=1)
        config = timm.data.resolve_model_data_config(model)
        model.load_state_dict(torch.load(root / name / "best.pt", map_location="cpu", weights_only=True))
        model.to("cuda")
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(), transforms.Normalize(config["mean"], config["std"]),
        ])
        scores[name] = []
        partitions = [("test", fresh)] if previous else [("validation", validation), ("test", fresh)]
        for partition, rows in partitions:
            loader = DataLoader(
                Ships(rows, DATA / "train_v2", transform),
                batch_size=32, num_workers=2, pin_memory=True, shuffle=False,
            )
            print(f"Scoring {name} on {len(rows):,} {partition} images", flush=True)
            labels, predictions = evaluate(model, loader, "cuda")
            if labels != [float(row["label"]) for row in rows]:
                raise ValueError("Predictions and labels are not aligned")
            scores[name].append(np.asarray(predictions))
        print(f"Scored {name}", flush=True)

    scores["ensemble"] = np.mean([scores[name] for name in NAMES], axis=0)
    validation_labels = [int(row["label"]) for row in validation]
    test_labels = [int(row["label"]) for row in fresh]
    results = {
        "test_scope": "all-unused" if args.all_unused else "stratified-sample",
        "source_split_sha256": digest,
        "fresh_test_ids_sha256": test_ids_digest(fresh),
        "validation_size": len(validation),
        "test_size": len(fresh),
        "test_positives": sum(test_labels),
        "ensemble_weights": dict.fromkeys(NAMES, 0.5),
        "metrics": {
            name: metrics_for(
                test_labels, predictions[-1],
                previous["metrics"][name]["threshold"] if previous else threshold_for(validation_labels, predictions[0]),
            )
            for name, predictions in scores.items()
        },
    }
    (OUTPUT / "ensemble-metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print("ENSEMBLE_RESULT " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
