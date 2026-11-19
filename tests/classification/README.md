# classification

## Overview

Unit tests for `deep_learning_loss.classification`.

## Components

| Component | Description |
| --------- | ----------- |
| [`test_binary_losses.py`](test_binary_losses.py) | `BinaryClassificationLoss` implementations. |
| [`test_multiclass_losses.py`](test_multiclass_losses.py) | `ClassificationLoss` implementations, class weights, and ignored positions. |
| [`test_soft_label_losses.py`](test_soft_label_losses.py) | `SoftLabelClassificationLoss` implementations against their hard-label counterparts. |
| [`test_ctc_loss.py`](test_ctc_loss.py) | `CTCLoss` against PyTorch and impossible alignments. |
