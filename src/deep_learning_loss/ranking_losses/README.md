# ranking_losses

## Overview

Losses of item scores against graded relevance, for learning to rank. Each implements [`RankingLoss`](../ranking_loss.py).

## Components

| Component | Description |
| --------- | ----------- |
| [`approx_ndcg.py`](approx_ndcg.py) | `ApproxNDCGLoss`: Differentiable NDCG; optimizes the ranking metric directly, weighing the top of the list most. |
| [`lambda_rank.py`](lambda_rank.py) | `LambdaRankLoss`: Pairwise loss weighted by the NDCG change of each swap; the loss behind LambdaMART, focusing on the top. |
| [`list_mle.py`](list_mle.py) | `ListMLELoss`: Listwise likelihood of the full order by relevance; for when the whole order matters. |
| [`list_net.py`](list_net.py) | `ListNetLoss`: Default listwise ranking loss; matches the softmax of the scores to that of the relevance. |
| [`rank_net.py`](rank_net.py) | `RankNetLoss`: Pairwise logistic ranking loss; a simple baseline that orders every pair of items. |
