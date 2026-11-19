# vision

## Overview

Losses specific to images: region overlap of segmentation maps, overlap of bounding boxes, and local image structure.

## Components

| Component | Description |
| --------- | ----------- |
| [`segmentation_loss.py`](segmentation_loss.py) | `SegmentationLoss`: class probability regions against a class map. |
| [segmentation_losses/](segmentation_losses/README.md) | `SegmentationLoss` implementations. |
| [`box_regression_loss.py`](box_regression_loss.py) | `BoxRegressionLoss`: overlap of predicted and target bounding boxes. |
| [box_regression_losses/](box_regression_losses/README.md) | `BoxRegressionLoss` implementations. |
| [`image_loss.py`](image_loss.py) | `ImageLoss`: local structure of predicted and target images. |
| [image_losses/](image_losses/README.md) | `ImageLoss` implementations. |
