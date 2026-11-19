# image_losses

## Overview

Structural losses between predicted and target images, for restoration and dense prediction. Each implements [`ImageLoss`](../image_loss.py).

## Components

| Component | Description |
| --------- | ----------- |
| [`gradient_difference.py`](gradient_difference.py) | `GradientDifferenceLoss`: Penalizes differences between image gradient magnitudes; sharpens edges in frame prediction and depth estimation. |
| [`ssim.py`](ssim.py) | `SSIMLoss`: Structural similarity loss; matches local luminance, contrast and structure as human vision perceives them. |
