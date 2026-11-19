# proxy_losses

## Overview

Losses of labeled embeddings against learnable class proxies, for face recognition and retrieval. Each implements [`ProxyEmbeddingLoss`](../proxy_loss.py).

## Components

| Component | Description |
| --------- | ----------- |
| [`anchor.py`](anchor.py) | `ProxyAnchorLoss`: Proxy loss that weighs each sample by its hardness; fast-converging metric learning for image retrieval. |
| [`arc_face.py`](arc_face.py) | `ArcFaceLoss`: Default proxy loss; additive angular margin softmax, the standard for face recognition. |
| [`cos_face.py`](cos_face.py) | `CosFaceLoss`: Additive cosine margin softmax; simpler than ArcFace and as strong for face recognition. |
