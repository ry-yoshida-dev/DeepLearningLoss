# regression

## Overview

Unit tests for `deep_learning_loss.regression`.

## Components

| Component | Description |
| --------- | ----------- |
| [`test_point_losses.py`](test_point_losses.py) | `PointLoss` implementations. |
| [`test_distribution_losses.py`](test_distribution_losses.py) | `DistributionLoss` implementations against `torch.distributions`, and masks over out-of-support targets. |
| [`test_ordinal_losses.py`](test_ordinal_losses.py) | `OrdinalRegressionLoss` implementations and rank decoding. |
