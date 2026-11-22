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

The [500-image pilot on Kaggle](https://www.kaggle.com/code/mdzubed/airbus-ship-classifier-benchmark/versions) trained each model for 10 epochs on a Tesla T4. It used 300 training images, 100 validation images, and 100 held-out test images. There were 22 positives in the test split. Predicting the majority class would score 78% accuracy.

| Model | Accuracy | Average precision | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Four-layer CNN](results/kaggle-500/cnn.json) | 77% | 0.557 | 0.566 | 0.484 | 0.682 |
| [ConvNeXt V2 Atto](results/kaggle-500/convnextv2.json) | 93% | 0.900 | 0.837 | 0.857 | 0.818 |
| [DINOv3 ViT-S/16](results/kaggle-500/dinov3.json) | 85% | 0.735 | 0.727 | 0.606 | 0.909 |

ConvNeXt V2 led in average precision and F1. DINOv3 found 20 of 22 positive images and produced 13 false positives. The small test set limits the precision of these estimates.

### 15,000-image Kaggle benchmark

The [private Kaggle GPU run](https://www.kaggle.com/code/mdzubed/airbus-ship-classifier-15000-image-benchmark) completed. It uses a stratified split of 9,000 training, 3,000 validation, and 3,000 held-out images. Each model runs for 10 epochs. The ConvNeXt V2 and DINOv3 backbones are frozen. The [Kaggle script](kaggle/benchmark.py) pins the training code revision and attaches to the competition dataset mounted by Kaggle.

The test split includes 663 positive images. Predicting the majority class would score 77.9% accuracy.

| Model | Accuracy | Average precision | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Four-layer CNN](results/kaggle-15000/cnn.json) | 87.0% | 0.798 | 0.718 | 0.689 | 0.750 |
| [ConvNeXt V2 Atto](results/kaggle-15000/convnextv2.json) | 94.3% | 0.946 | 0.871 | 0.866 | 0.876 |
| [DINOv3 ViT-S/16](results/kaggle-15000/dinov3.json) | 94.1% | 0.950 | 0.872 | 0.835 | 0.911 |

DINOv3 ViT-S/16 had the highest average precision (0.950). The CNN baseline scored 0.798 on the same split.

### ConvNeXt V2 and DINOv3 ensemble

The ensemble averages the two saved models' ship probabilities with equal weights. Each individual model and the ensemble chose its threshold by F1 on the original 3,000 validation images. The comparison below uses the same fresh 3,000-image test sample for all three, excluding every image in the original 15,000-image split by image ID. The test sample has 663 positives.

| Model | Accuracy | Average precision | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt V2 Atto | 94.60% | 0.9431 | 0.8763 | 0.8872 | 0.8658 |
| DINOv3 ViT-S/16 | 94.57% | 0.9490 | 0.8808 | 0.8551 | 0.9080 |
| Equal-weight ensemble | 95.10% | 0.9559 | 0.8915 | 0.8728 | 0.9110 |

The ensemble had the highest accuracy, average precision, F1, and recall on this sample. ConvNeXt V2 had the highest precision and produced 73 false positives versus 88 for the ensemble. Ensemble inference requires both backbones.

[Evaluation script](kaggle/ensemble.py) · [Metrics JSON](results/ensemble-3000.json) · [Private Kaggle run](https://www.kaggle.com/code/mdzubed/airbus-ship-classifier-ensemble)

### Full held-out evaluation

The [private Kaggle GPU run](https://www.kaggle.com/code/mdzubed/airbus-ship-classifier-full-test) evaluates both saved models and their fixed 50/50 ensemble on 174,556 previously unused labeled images, including 38,578 ship-positive images. It excludes every image ID used in the 500-image pilot, the 15,000-image benchmark, or the earlier ensemble test. It reuses the earlier validation thresholds, and the checkpoints are unchanged.

The [evaluation script](kaggle/ensemble.py) accepts `--all-unused` for this run. Results are pending.

## References

- [Original Stanford CS229 project](https://github.com/cs229ShipDetection/CS229project-Airbus-Ship-Detection)
- [ConvNeXt V2 paper](https://arxiv.org/abs/2301.00808)
- [DINOv3 reference implementation](https://github.com/facebookresearch/dinov3)
