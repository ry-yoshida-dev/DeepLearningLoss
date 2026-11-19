# multiclass_losses

## Overview

Losses of class logits against the true class index, for multiclass classification. Each implements [`ClassificationLoss`](../multiclass_loss.py).

## Components

| Component | Description |
| --------- | ----------- |
| [`cross_entropy.py`](cross_entropy.py) | `CrossEntropyLoss`: Default multiclass classification loss, with optional label smoothing. |
| [`focal.py`](focal.py) | `FocalLoss`: Cross-entropy that down-weights easy samples; for class imbalance. |
| [`generalized_cross_entropy.py`](generalized_cross_entropy.py) | `GeneralizedCrossEntropyLoss`: Cross-entropy robust to noisy labels, between cross-entropy and MAE. |
| [`hinge.py`](hinge.py) | `MulticlassHingeLoss`: SVM-style margin loss on class scores. |
| [`poly1.py`](poly1.py) | `Poly1Loss`: Cross-entropy plus a linear term; a drop-in replacement that often improves accuracy. |
