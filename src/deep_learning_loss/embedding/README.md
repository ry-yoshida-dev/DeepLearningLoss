# embedding

## Overview

Losses that shape an embedding space, for self-supervised, metric and cross-modal learning.

## Components

| Component | Description |
| --------- | ----------- |
| [`paired_loss.py`](paired_loss.py) | `PairedEmbeddingLoss`: two batches of corresponding embeddings. |
| [paired_losses/](paired_losses/README.md) | `PairedEmbeddingLoss` implementations. |
| [`labeled_loss.py`](labeled_loss.py) | `LabeledEmbeddingLoss`: embeddings with class labels. |
| [labeled_losses/](labeled_losses/README.md) | `LabeledEmbeddingLoss` implementations. |
| [`proxy_loss.py`](proxy_loss.py) | `ProxyEmbeddingLoss`: labeled embeddings against learnable class proxies. |
| [proxy_losses/](proxy_losses/README.md) | `ProxyEmbeddingLoss` implementations. |
