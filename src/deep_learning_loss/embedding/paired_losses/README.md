# paired_losses

## Overview

Losses of two batches of corresponding embeddings, for self-supervised and cross-modal learning. Each implements [`PairedEmbeddingLoss`](../paired_loss.py).

## Components

| Component | Description |
| --------- | ----------- |
| [`barlow_twins.py`](barlow_twins.py) | `BarlowTwinsLoss`: Cross-correlation loss; self-supervised learning without negatives, working with small batches. |
| [`info_nce.py`](info_nce.py) | `InfoNCELoss`: Default contrastive loss for paired views; the CLIP loss for aligning two modalities. |
| [`negative_cosine_similarity.py`](negative_cosine_similarity.py) | `NegativeCosineSimilarityLoss`: Negative cosine similarity of pairs; the BYOL and SimSiam loss, with a predictor and stop-gradient. |
| [`nt_xent.py`](nt_xent.py) | `NTXentLoss`: SimCLR contrastive loss; self-supervised learning from two augmentations of each sample. |
| [`sig_lip.py`](sig_lip.py) | `SigLIPLoss`: Sigmoid pairwise loss for aligning two modalities; scales to large batches better than the CLIP softmax. |
| [`vicreg.py`](vicreg.py) | `VICRegLoss`: Variance-invariance-covariance loss; self-supervised learning without negatives or large batches. |
