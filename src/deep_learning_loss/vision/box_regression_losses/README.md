# box_regression_losses

## Overview

IoU-based losses between predicted and target bounding boxes, for object detection. Each implements [`BoxRegressionLoss`](../box_regression_loss.py).

## Components

| Component | Description |
| --------- | ----------- |
| [`ciou.py`](ciou.py) | `CIoULoss`: IoU loss with center-distance and aspect-ratio penalties; the usual box loss of YOLO-style detectors. |
| [`diou.py`](diou.py) | `DIoULoss`: IoU loss with a center-distance penalty; converges faster than GIoU, especially for nested boxes. |
| [`giou.py`](giou.py) | `GIoULoss`: Default box regression loss; IoU loss with a gradient even for boxes that do not overlap. |
| [`iou.py`](iou.py) | `IoULoss`: Plain IoU loss; scale-invariant, but without gradient for boxes that do not overlap. |
