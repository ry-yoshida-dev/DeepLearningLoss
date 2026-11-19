# deep_learning_loss

## Overview

Loss functions grouped by learning domain, then into families by what they compare, and the guide from learning task to losses.

## Components

| Component | Description |
| --------- | ----------- |
| [`loss.py`](loss.py) | `Loss`: root of every family, with a one-line summary of each loss. |
| [`elementwise_loss.py`](elementwise_loss.py) | `ElementwiseLoss`: losses defined value by value, with mean and masked mean. |
| [`learning_task.py`](learning_task.py) | `LearningTask`: learning tasks and the losses that suit each. |
| [regression/](regression/README.md) | Losses of predicted values, distributions and ordered ranks. |
| [classification/](classification/README.md) | Losses of class logits against binary, class-index, soft or sequence targets. |
| [vision/](vision/README.md) | Losses of segmentation maps, bounding boxes and images. |
| [embedding/](embedding/README.md) | Losses of embeddings, paired, labeled or against class proxies. |
| [generative/](generative/README.md) | Losses and penalties for GANs and VAEs. |
| [`distillation_loss.py`](distillation_loss.py) | `DistillationLoss`: divergence between student and teacher distributions. |
| [distillation_losses/](distillation_losses/README.md) | `DistillationLoss` implementations. |
| [`ranking_loss.py`](ranking_loss.py) | `RankingLoss`: item scores against graded relevance. |
| [ranking_losses/](ranking_losses/README.md) | `RankingLoss` implementations. |
