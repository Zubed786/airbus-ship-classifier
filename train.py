"""Train and evaluate binary ship classifiers on a common split."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import timm
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

BACKBONES = {
    "convnextv2": "convnextv2_atto.fcmae_ft_in1k",
    "dinov3": "vit_small_patch16_dinov3.lvd1689m",
}


class Ships(Dataset):
    def __init__(self, rows, folder, transform):
        self.rows = rows
        self.folder = folder
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.folder / row["image"]) as image:
            return self.transform(image.convert("RGB")), float(row["label"])


def baseline():
    layers = []
    for channels_in, channels_out in zip((3, 16, 32, 64), (16, 32, 64, 128)):
        layers.extend([
            nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out), nn.ReLU(), nn.MaxPool2d(2),
        ])
    return nn.Sequential(
        *layers, nn.AdaptiveAvgPool2d((16, 16)), nn.Flatten(),
        nn.Linear(32768, 128), nn.ReLU(), nn.Linear(128, 1),
    )


def evaluate(model, loader, device):
    model.eval()
    labels, scores = [], []
    with torch.inference_mode():
        for images, targets in loader:
            scores.extend(model(images.to(device)).flatten().sigmoid().cpu().tolist())
            labels.extend(targets.tolist())
    return labels, scores


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=False)
    split_bytes = args.split.read_bytes()
    rows = list(csv.DictReader(split_bytes.decode("utf-8").splitlines()))
    if len({row["image"] for row in rows}) != len(rows):
        raise ValueError("The split contains duplicate image IDs")
    if args.model == "cnn":
        model = baseline()
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        model = timm.create_model(
            BACKBONES[args.model], pretrained=not args.random_init, num_classes=1,
        )
        config = timm.data.resolve_model_data_config(model)
        mean, std = config["mean"], config["std"]
    size = args.size or (256 if args.model == "cnn" else 224)
    if args.model == "dinov3" and size % 16:
        raise ValueError("DINOv3 requires an image size divisible by 16")
    freeze = args.model != "cnn" and not args.finetune
    if freeze:
        model.requires_grad_(False)
        model.get_classifier().requires_grad_(True)

    loaders = {}
    for split in ("train", "val", "test"):
        subset = [row for row in rows if row["split"] == split]
        if {row["label"] for row in subset} != {"0", "1"}:
            raise ValueError(f"{split} must contain both labels")
        steps = [transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC)]
        if split == "train":
            steps += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        loaders[split] = DataLoader(
            Ships(subset, args.images, transforms.Compose(steps)),
            batch_size=args.batch_size, shuffle=split == "train",
            num_workers=args.workers, pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )

    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr, weight_decay=0.01,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    best_ap = -1.0
    checkpoint = args.output / "best.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(not freeze)
        if freeze:
            model.get_classifier().train()
        total_loss = 0.0
        for images, labels in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device)).flatten()
            loss = loss_fn(logits, labels.to(device, dtype=torch.float32))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.numel()
        labels, scores = evaluate(model, loaders["val"], device)
        ap = average_precision_score(labels, scores)
        record = {"epoch": epoch, "loss": total_loss / len(loaders["train"].dataset),
                  "validation_average_precision": ap}
        history.append(record)
        print(json.dumps(record), flush=True)
        if ap > best_ap:
            best_ap = ap
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    labels, scores = evaluate(model, loaders["val"], device)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1]).clip(min=1e-12)
    threshold = float(thresholds[f1.argmax()])
    labels, scores = evaluate(model, loaders["test"], device)
    predictions = [int(score >= threshold) for score in scores]
    results = {
        "model": args.model, "backbone": BACKBONES.get(args.model),
        "pretrained": args.model != "cnn" and not args.random_init,
        "frozen_backbone": freeze, "best_epoch": best_epoch,
        "validation_f1_threshold": threshold, "config": vars(args),
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(), "device": str(device),
        "test_size": len(labels), "test_positives": int(sum(labels)),
        "test_majority_accuracy": max(sum(labels), len(labels) - sum(labels)) / len(labels),
        "test_accuracy": accuracy_score(labels, predictions),
        "test_average_precision": average_precision_score(labels, scores),
        "test_f1": f1_score(labels, predictions, zero_division=0),
        "test_precision": precision_score(labels, predictions, zero_division=0),
        "test_recall": recall_score(labels, predictions, zero_division=0),
        "test_confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "history": history,
    }
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, default=str), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("data/split.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("cnn", *BACKBONES), required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--size", type=int)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--random-init", action="store_true", help="Skip pretrained downloads for smoke tests")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads, args.size or 224) < 1:
        parser.error("Epochs, batch size, threads and image size must be positive")
    train(args)
