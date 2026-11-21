import subprocess
import sys
from pathlib import Path

import torch

WORK = Path("/kaggle/working")
SOURCE = Path("/kaggle/input/competitions/airbus-ship-detection")
REPO = Path("/tmp/airbus-ship-classifier")


def run(*args):
    subprocess.run(args, cwd=WORK, check=True)


if not torch.cuda.is_available():
    raise RuntimeError("A CUDA GPU is required for this run")
print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
run("git", "clone", "https://github.com/Zubed786/airbus-ship-classifier.git", str(REPO))
run("git", "-C", str(REPO), "checkout", "--detach", "2ca89b504d3fb640f46000e7a7e706c0a6fb27c7")
run(sys.executable, "-m", "pip", "install", "--quiet", "timm>=1.0.24,<2")
labels = SOURCE / "train_ship_segmentations_v2.csv"
print(f"Dataset labels: {labels}", flush=True)
images = labels.parent / "train_v2"
if not images.is_dir():
    images = next(path for path in SOURCE.rglob("train_v2") if path.is_dir())
split = WORK / "split.csv"
run(sys.executable, str(REPO / "prepare.py"), "--labels", str(labels),
    "--images", str(images), "--limit", "15000", "--output", str(split))
for model in ("cnn", "convnextv2", "dinov3"):
    run(sys.executable, str(REPO / "train.py"), "--model", model,
        "--images", str(images), "--split", str(split), "--epochs", "10", "--workers", "2",
        "--output", str(WORK / model))
