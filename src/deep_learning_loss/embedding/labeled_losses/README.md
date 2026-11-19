# labeled_losses

## Overview

Losses of embeddings with class labels, for metric learning. Each implements [`LabeledEmbeddingLoss`](../labeled_loss.py).

## Components

| Component | Description |
| --------- | ----------- |
| [`contrastive.py`](contrastive.py) | `ContrastiveLoss`: Pairwise margin loss; the classic loss for Siamese verification networks. |
| [`multi_similarity.py`](multi_similarity.py) | `MultiSimilarityLoss`: Pair-weighting loss with hard-pair mining; strong metric learning for retrieval with many classes. |
| [`supervised_contrastive.py`](supervised_contrastive.py) | `SupervisedContrastiveLoss`: Default metric learning loss; contrasts each embedding with all others of its label. |
| [`triplet.py`](triplet.py) | `TripletLoss`: Batch-hard triplet loss; for retrieval and re-identification. |
