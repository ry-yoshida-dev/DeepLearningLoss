# binary_losses

## Overview

Losses of independent yes/no logits, for binary and multi-label classification. Each implements [`BinaryClassificationLoss`](../binary_loss.py).

## Components

| Component | Description |
| --------- | ----------- |
| [`asymmetric.py`](asymmetric.py) | `AsymmetricLoss`: Multi-label loss that discards easy negatives; for samples with few positive labels among many. |
| [`cross_entropy.py`](cross_entropy.py) | `BinaryCrossEntropyLoss`: Default binary and multi-label classification loss, with an optional positive weight. |
| [`focal.py`](focal.py) | `BinaryFocalLoss`: Binary cross-entropy that down-weights easy samples; for dense detection and class imbalance. |
