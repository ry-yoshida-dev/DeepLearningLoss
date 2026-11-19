# vision

## Overview

Unit tests for `deep_learning_loss.vision`.

## Components

| Component | Description |
| --------- | ----------- |
| [`test_segmentation_losses.py`](test_segmentation_losses.py) | `SegmentationLoss` implementations, binary and masked segmentation, and Lovasz-Softmax against a reference. |
| [`test_box_regression_losses.py`](test_box_regression_losses.py) | `BoxRegressionLoss` implementations against hand-computed overlaps. |
| [`test_image_losses.py`](test_image_losses.py) | `ImageLoss` implementations. |
