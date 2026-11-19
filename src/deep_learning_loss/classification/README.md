# classification

## Overview

Losses of class logits, against independent yes/no targets, class indices, class probabilities or label sequences.

## Components

| Component | Description |
| --------- | ----------- |
| [`binary_loss.py`](binary_loss.py) | `BinaryClassificationLoss`: independent yes/no logits against targets. |
| [binary_losses/](binary_losses/README.md) | `BinaryClassificationLoss` implementations. |
| [`multiclass_loss.py`](multiclass_loss.py) | `ClassificationLoss`: class logits against class indices. |
| [multiclass_losses/](multiclass_losses/README.md) | `ClassificationLoss` implementations. |
| [`soft_label_loss.py`](soft_label_loss.py) | `SoftLabelClassificationLoss`: class logits against target class probabilities. |
| [soft_label_losses/](soft_label_losses/README.md) | `SoftLabelClassificationLoss` implementations. |
| [`ctc_loss.py`](ctc_loss.py) | `CTCLoss`: frame logits against shorter label sequences, without alignment. |
