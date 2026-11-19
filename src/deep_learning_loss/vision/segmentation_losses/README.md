# segmentation_losses

## Overview

Region losses for semantic segmentation. Each implements [`SegmentationLoss`](../segmentation_loss.py).

## Components

| Component | Description |
| --------- | ----------- |
| [`dice.py`](dice.py) | `DiceLoss`: Default overlap loss for segmentation; robust to class imbalance, usually added to cross-entropy. |
| [`focal_tversky.py`](focal_tversky.py) | `FocalTverskyLoss`: Tversky loss that emphasizes poorly segmented classes; for very small, rare structures. |
| [`jaccard.py`](jaccard.py) | `JaccardLoss`: Soft IoU loss; optimizes the intersection over union that segmentation benchmarks report. |
| [`lovasz_softmax.py`](lovasz_softmax.py) | `LovaszSoftmaxLoss`: Convex surrogate of the IoU; optimizes the benchmark metric directly, usually after pretraining with cross-entropy. |
| [`tversky.py`](tversky.py) | `TverskyLoss`: Dice variant weighting false negatives over false positives; favors recall of small structures. |
