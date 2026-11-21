# Airbus ship presence classification

A follow-up to the Stanford CS229 project by Junwen Zheng, Ying Chen, and Zhengqing Zhou. Given a satellite image, predict whether it contains at least one ship. Each image receives one binary label derived from the Airbus mask annotations.

| Model | Training |
| --- | --- |
| Four-layer CNN | Train from scratch using a baseline inspired by the CS229 architecture |
| ConvNeXt V2 Atto | Train the classification head on a pretrained backbone |
| DINOv3 ViT-S/16 | Train the classification head on a pretrained backbone |

The backbone options also support full fine-tuning with `--finetune`. Start with a smaller learning rate such as `--lr 0.0001` for that comparison.

## Reproduce the experiment

Install the dependencies with Python 3.11. For GPU runs, select the appropriate PyTorch build from the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

```powershell
python -m pip install -r requirements.txt
```

Get `train_v2` and `train_ship_segmentations_v2.csv` from the [Airbus Ship Detection Challenge](https://www.kaggle.com/competitions/airbus-ship-detection/data) and place them under `data/`. Kaggle controls access to the competition data. Pretrained runs download weights through timm.

```powershell
python prepare.py --labels data/train_ship_segmentations_v2.csv --images data/train_v2 --limit 15000
python train.py --model cnn --images data/train_v2 --output runs/cnn
python train.py --model convnextv2 --images data/train_v2 --output runs/convnextv2
python train.py --model dinov3 --images data/train_v2 --output runs/dinov3
```

The default split uses 60% training, 20% validation, and 20% test, stratified by label after combining mask rows by image ID. Each model selects the epoch by validation average precision and chooses a threshold by validation F1. Each run saves the selected checkpoint and a metrics JSON file with the split fingerprint, confusion matrix, majority-class baseline, accuracy, average precision, precision, recall, and F1. Compare runs with the same split fingerprint.

For a smaller experiment, authenticate with the Kaggle CLI and accept the competition rules. The subset helper downloads images individually and keeps the class proportions from the label CSV.

```powershell
python -m pip install "kaggle>=2.2.5,<3"
python download_subset.py --count 500
python prepare.py --labels data/airbus/subset_labels.csv --images data/airbus/train_v2
```

Use `--images data/airbus/train_v2` when training on this subset.

## Results

The original report lists 94% test accuracy for its simple CNN and 90% for the DenseNet-169 transfer baseline. Those figures come from its own split. This experiment measures changes against its four-layer CNN on the shared split. The split groups annotations by image ID and does not account for overlapping geographic coverage across image IDs.

No Airbus benchmark results are available yet.

## References

- [Original Stanford CS229 project](https://github.com/cs229ShipDetection/CS229project-Airbus-Ship-Detection)
- [ConvNeXt V2 paper](https://arxiv.org/abs/2301.00808)
- [DINOv3 reference implementation](https://github.com/facebookresearch/dinov3)
