# tests

## Overview

Unit tests for `deep_learning_loss`, laid out like the package: values against reference formulas or PyTorch, masking, gradients at edge cases, and input validation.

## Components

| Component | Description |
| --------- | ----------- |
| [regression/](regression/README.md) | Tests of `deep_learning_loss.regression`. |
| [classification/](classification/README.md) | Tests of `deep_learning_loss.classification`. |
| [vision/](vision/README.md) | Tests of `deep_learning_loss.vision`. |
| [embedding/](embedding/README.md) | Tests of `deep_learning_loss.embedding`. |
| [generative/](generative/README.md) | Tests of `deep_learning_loss.generative`. |
| [`test_distillation_losses.py`](test_distillation_losses.py) | `DistillationLoss` implementations. |
| [`test_ranking_losses.py`](test_ranking_losses.py) | `RankingLoss` implementations and padded lists. |
| [`test_learning_task.py`](test_learning_task.py) | `LearningTask` covers every loss, and every loss constructs and summarizes. |
