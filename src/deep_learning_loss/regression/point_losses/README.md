# point_losses

## Overview

Losses between predicted values and targets of the same shape. Each implements [`PointLoss`](../point_loss.py).

## Components

| Component | Description |
| --------- | ----------- |
| [`charbonnier.py`](charbonnier.py) | `CharbonnierLoss`: Differentiable absolute error; common in image restoration such as super-resolution. |
| [`huber.py`](huber.py) | `HuberLoss`: MSE for small errors and MAE for large ones; robust to outliers with a tunable threshold. |
| [`log_cosh.py`](log_cosh.py) | `LogCoshLoss`: Smooth loss between MSE and MAE; robust to outliers without a threshold to tune. |
| [`mae.py`](mae.py) | `MAELoss`: Mean absolute error; robust to outliers, predicting the conditional median. |
| [`mape.py`](mape.py) | `MAPELoss`: Absolute percentage error; a scale-free error for targets far from zero. |
| [`mse.py`](mse.py) | `MSELoss`: Default regression loss; predicts the conditional mean, and large errors dominate. |
| [`msle.py`](msle.py) | `MSLELoss`: Squared error of log values; penalizes relative errors of non-negative targets spanning magnitudes. |
| [`pinball.py`](pinball.py) | `PinballLoss`: Quantile loss; trains a prediction of one quantile, such as a bound of a prediction interval. |
| [`smape.py`](smape.py) | `SMAPELoss`: Symmetric percentage error in [0, 2]; a bounded, scale-free error for forecasting. |
