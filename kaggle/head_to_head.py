"""Compare the reconstructed CS229 SIM-CNN with the saved ensemble."""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

REPO = Path("/tmp/airbus-ship-classifier")
ROOT = Path("/kaggle/input/competitions/airbus-ship-detection")
OUTPUT = Path("/kaggle/working")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = "cuda"
    subprocess.run(["git", "clone", "https://github.com/Zubed786/airbus-ship-classifier.git", str(REPO)], check=True)
    subprocess.run(["git", "-C", str(REPO), "checkout", "--detach", "60f9b21"], check=True)
    sys.path.insert(0, str(REPO))
    from train import BACKBONES, Ships, baseline, evaluate
    sys.path.insert(0, str(REPO / "kaggle"))
    from ensemble import SOURCE, find_output

    source = find_output(SOURCE)
    split = (source / "split.csv").read_bytes()
    fingerprint = hashlib.sha256(split).hexdigest()
    rows = list(csv.DictReader(split.decode().splitlines()))
    partitions = {name: [row for row in rows if row["split"] == name] for name in ("train", "val", "test")}
    assert [len(partitions[name]) for name in partitions] == [9000, 3000, 3000]
    assert len({row["image"] for row in rows}) == len(rows)

    def loader(partition, size, mean=None, std=None):
        steps = [transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC)]
        if partition == "train":
            steps += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
                transforms.ColorJitter(brightness=0.15),
            ]
        steps.append(transforms.ToTensor())
        if mean is not None:
            steps.append(transforms.Normalize(mean, std))
        return DataLoader(
            Ships(partitions[partition], ROOT / "train_v2", transforms.Compose(steps)),
            batch_size=64, num_workers=2, shuffle=partition == "train",
            pin_memory=True, persistent_workers=True,
        )

    model = baseline().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    train_loader = loader("train", 256)
    val_loader = loader("val", 256)
    best_loss = float("inf")
    best_epoch = 0
    checkpoint = OUTPUT / "paper-cnn-best.pt"
    for epoch in range(1, 31):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device)).flatten(), labels.to(device, dtype=torch.float32))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            total = sum(
                criterion(model(images.to(device)).flatten(), labels.to(device, dtype=torch.float32)).item() * len(labels)
                for images, labels in val_loader
            )
        validation_loss = total / len(val_loader.dataset)
        scheduler.step(validation_loss)
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            torch.save(model.state_dict(), checkpoint)
        print(json.dumps({"epoch": epoch, "validation_loss": validation_loss}), flush=True)

    def score(model, val, test):
        return [np.asarray(evaluate(model, batch, device)[1]) for batch in (val, test)]

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    baseline_scores = score(model, val_loader, loader("test", 256))
    del model, train_loader, val_loader
    predictions = []
    for name in BACKBONES:
        recorded = json.loads((source / name / "metrics.json").read_text())
        if recorded["split_sha256"] != fingerprint:
            raise ValueError(f"Checkpoint split mismatch for {name}")
        model = timm.create_model(BACKBONES[name], pretrained=False, num_classes=1)
        config = timm.data.resolve_model_data_config(model)
        model.load_state_dict(torch.load(source / name / "best.pt", weights_only=True, map_location="cpu"))
        predictions.append(score(
            model.to(device), loader("val", 224, config["mean"], config["std"]),
            loader("test", 224, config["mean"], config["std"]),
        ))
        del model

    val_true, test_true = (np.asarray([int(row["label"]) for row in partitions[name]]) for name in ("val", "test"))
    result = {"split_sha256": fingerprint, "split_sizes": {k: len(v) for k, v in partitions.items()},
              "test_positives": int(test_true.sum()), "paper_cnn_epochs": 30, "paper_cnn_best_epoch": best_epoch,
              "threshold_rule": "Maximize validation accuracy on 0.1 to 0.9 in steps of 0.1, ties nearest 0.5",
              "metrics": {}}
    correct = {}
    for name, (val, test) in {"paper_cnn_reconstruction": baseline_scores, "ensemble": np.mean(predictions, axis=0)}.items():
        threshold = max((i / 10 for i in range(1, 10)), key=lambda t: (accuracy_score(val_true, val >= t), -abs(t-0.5)))
        predicted = test >= threshold
        correct[name] = predicted == test_true
        tn, fp, fn, tp = confusion_matrix(test_true, predicted).ravel()
        result["metrics"][name] = {
            "threshold": threshold, "accuracy": accuracy_score(test_true, predicted),
            "average_precision": average_precision_score(test_true, test),
            "f1": f1_score(test_true, predicted), "precision": precision_score(test_true, predicted, zero_division=0),
            "recall": recall_score(test_true, predicted, zero_division=0), "specificity": float(tn/(tn+fp)),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        }
    a, b = correct["ensemble"], correct["paper_cnn_reconstruction"]
    from scipy.stats import binomtest
    wins, losses = int((a & ~b).sum()), int((b & ~a).sum())
    result["paired_accuracy"] = {"ensemble_only_correct": wins, "cnn_only_correct": losses,
                                 "mcnemar_exact_p": binomtest(wins, wins + losses).pvalue if wins + losses else 1.0}
    (OUTPUT / "head-to-head.json").write_text(json.dumps(result, indent=2) + "\n")
    print("HEAD_TO_HEAD " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
