# soft_label_losses

## Overview

Losses of class logits against target class probabilities, for Mixup, CutMix and noisy or pseudo labels. Each implements [`SoftLabelClassificationLoss`](../soft_label_loss.py).

## Components

| Component | Description |
| --------- | ----------- |
| [`cross_entropy.py`](cross_entropy.py) | `SoftLabelCrossEntropyLoss`: Default loss for soft targets; the cross-entropy that Mixup, CutMix and pseudo-labeling train with. |
| [`focal.py`](focal.py) | `SoftLabelFocalLoss`: Focal loss for soft targets; down-weights well-classified classes under Mixup or label noise. |
