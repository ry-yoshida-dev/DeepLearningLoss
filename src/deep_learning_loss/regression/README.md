# regression

## Overview

Losses of networks that predict values: the values themselves, a distribution over them, or an ordered rank.

## Components

| Component | Description |
| --------- | ----------- |
| [`point_loss.py`](point_loss.py) | `PointLoss`: errors between predicted and target values. |
| [point_losses/](point_losses/README.md) | `PointLoss` implementations. |
| [`distribution_loss.py`](distribution_loss.py) | `DistributionLoss`: scoring rules of predicted distributions. |
| [distribution_losses/](distribution_losses/README.md) | `DistributionLoss` implementations. |
| [`ordinal_loss.py`](ordinal_loss.py) | `OrdinalRegressionLoss`: threshold logits against ordered ranks. |
| [ordinal_losses/](ordinal_losses/README.md) | `OrdinalRegressionLoss` implementations. |
