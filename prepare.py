"""Build one binary label per image and save a reproducible stratified split."""
import argparse
import csv
from pathlib import Path

from sklearn.model_selection import train_test_split


def prepare(labels_file, images, output, limit, seed):
    labels = {}
    with labels_file.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["ImageId"]
            if Path(name).name != name or not name or "\\" in name:
                raise ValueError(f"Invalid image name: {name!r}")
            encoded = (row["EncodedPixels"] or "").strip().lower()
            labels[name] = labels.get(name, False) or encoded not in {"", "nan"}

    rows = [(name, int(label)) for name, label in sorted(labels.items())]
    if limit and limit < len(rows):
        rows, _ = train_test_split(
            rows, train_size=limit, stratify=[label for _, label in rows],
            random_state=seed,
        )
    missing = next((name for name, _ in rows if not (images / name).is_file()), None)
    if missing:
        raise FileNotFoundError(images / missing)
    remaining, test = train_test_split(
        rows, test_size=0.2, stratify=[label for _, label in rows], random_state=seed,
    )
    train, val = train_test_split(
        remaining, test_size=0.25, stratify=[label for _, label in remaining],
        random_state=seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "label", "split"])
        for split, subset in (("train", train), ("val", val), ("test", test)):
            writer.writerows((name, label, split) for name, label in subset)
            print(f"{split}: {len(subset)} images, {sum(label for _, label in subset)} positive")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/split.csv"))
    parser.add_argument("--limit", type=int, default=15000, help="0 uses all images")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.labels, args.images, args.output, args.limit, args.seed)
