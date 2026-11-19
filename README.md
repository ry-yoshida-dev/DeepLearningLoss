# DeepLearningLoss

## Overview

DeepLearningLoss provides typed PyTorch loss functions, validated at construction and on every call, and tells you which losses suit a learning task. Losses are grouped into families by what they compare, so the losses of one family are interchangeable behind the same method.

| Family | Compares | Typical tasks |
| ------ | -------- | ------------- |
| [Point](src/deep_learning_loss/regression/point_losses/README.md) | Predicted values with targets of the same shape | Regression, forecasting, reconstruction |
| [Distribution](src/deep_learning_loss/regression/distribution_losses/README.md) | Predicted distribution parameters or quantiles with targets | Regression with uncertainty, counts, bounded values |
| [Binary classification](src/deep_learning_loss/classification/binary_losses/README.md) | Logits with 0/1 (or soft) targets of the same shape | Binary and multi-label classification |
| [Classification](src/deep_learning_loss/classification/multiclass_losses/README.md) | Class logits with class indices | Multiclass classification, per-pixel segmentation |
| [Soft-label classification](src/deep_learning_loss/classification/soft_label_losses/README.md) | Class logits with target class probabilities | Mixup and CutMix, noisy or pseudo labels |
| [Ordinal regression](src/deep_learning_loss/regression/ordinal_losses/README.md) | Threshold logits with ordered ranks | Ratings, grades, severity levels |
| [Segmentation](src/deep_learning_loss/vision/segmentation_losses/README.md) | Class logits with a class map, by region | Semantic segmentation |
| [Box regression](src/deep_learning_loss/vision/box_regression_losses/README.md) | Predicted with target bounding boxes, by overlap | Object detection |
| [Image](src/deep_learning_loss/vision/image_losses/README.md) | Predicted with target images, by local structure | Image restoration, depth estimation |
| [Sequence alignment](src/deep_learning_loss/classification/ctc_loss.py) | Frame logits with shorter label sequences | Speech and handwriting recognition |
| [Distillation](src/deep_learning_loss/distillation_losses/README.md) | Student logits with teacher logits | Knowledge distillation |
| [Paired embedding](src/deep_learning_loss/embedding/paired_losses/README.md) | Two batches of embeddings whose rows correspond | Self-supervised and cross-modal learning |
| [Labeled embedding](src/deep_learning_loss/embedding/labeled_losses/README.md) | Embeddings with class labels | Metric learning, retrieval |
| [Proxy embedding](src/deep_learning_loss/embedding/proxy_losses/README.md) | Embeddings with labels and learnable class proxies | Face recognition, retrieval |
| [Ranking](src/deep_learning_loss/ranking_losses/README.md) | Item scores with graded relevance | Learning to rank |
| [Latent prior](src/deep_learning_loss/generative/latent_kl_divergence_loss.py) | A Gaussian posterior with the standard normal prior | Variational autoencoders |
| [Adversarial](src/deep_learning_loss/generative/adversarial_losses/README.md) | Discriminator scores of real and generated samples | GANs |
| [Gradient penalty](src/deep_learning_loss/generative/gradient_penalties/README.md) | Discriminator input gradients with a target norm | GANs |

## Installation

```bash
pip install .
```

## Usage

Find the losses that suit a task, the usual default first:

```python
from deep_learning_loss import DiceLoss, LearningTask

print(LearningTask.MULTI_LABEL_CLASSIFICATION.describe())
# MULTI_LABEL_CLASSIFICATION:
# - BinaryCrossEntropyLoss: Default binary and multi-label classification loss, with an optional positive weight.
# - AsymmetricLoss: Multi-label loss that discards easy negatives; for samples with few positive labels among many.
# - BinaryFocalLoss: Binary cross-entropy that down-weights easy samples; for dense detection and class imbalance.

LearningTask.find(DiceLoss)  # [LearningTask.SEMANTIC_SEGMENTATION]
```

Compute a loss:

```python
import torch

from deep_learning_loss import CrossEntropyLoss, DiceLoss, GaussianNLLLoss, GIoULoss, HuberLoss

predictions, targets = torch.randn(32, 24), torch.randn(32, 24)
loss = HuberLoss(delta=0.5).compute(predictions, targets)
loss = HuberLoss().compute_masked(predictions, targets, is_included=~targets.isnan())

logits, indices = torch.randn(8, 3, 64, 64), torch.randint(0, 3, (8, 64, 64))
loss = CrossEntropyLoss().compute(logits, indices, class_weights=torch.tensor([0.2, 1.0, 3.0]))
loss = loss + DiceLoss().compute(logits, indices)

parameters = torch.randn(32, 24, 2)  # (mean, raw scale) for each target value
loss = GaussianNLLLoss().compute(parameters, targets)
point_forecast = GaussianNLLLoss().compute_mean(parameters)

target_boxes = torch.tensor([[10.0, 20.0, 50.0, 80.0], [0.0, 0.0, 30.0, 30.0]])  # (x_min, y_min, x_max, y_max)
loss = GIoULoss().compute_masked(target_boxes + 5.0, target_boxes, is_included=torch.tensor([True, False]))
```
