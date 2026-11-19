# ordinal_losses

## Overview

Losses of threshold logits against ordered ranks, for ordinal regression. Each implements [`OrdinalRegressionLoss`](../ordinal_loss.py).

## Components

| Component | Description |
| --------- | ----------- |
| [`coral.py`](coral.py) | `CORALLoss`: Default ordinal loss; one binary task per threshold, consistent when the logits share their weights. |
| [`corn.py`](corn.py) | `CORNLoss`: Conditional ordinal loss; consistent ranks with any output layer, often more accurate than CORAL. |
