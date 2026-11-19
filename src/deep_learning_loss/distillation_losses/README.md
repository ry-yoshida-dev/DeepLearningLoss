# distillation_losses

## Overview

Divergences between student and teacher class distributions, for knowledge distillation. Each implements [`DistillationLoss`](../distillation_loss.py).

## Components

| Component | Description |
| --------- | ----------- |
| [`jensen_shannon_divergence.py`](jensen_shannon_divergence.py) | `JensenShannonDivergenceLoss`: Symmetric, bounded distillation loss; also suits mutual learning between peers. |
| [`kl_divergence.py`](kl_divergence.py) | `KLDivergenceLoss`: Default distillation loss; the student covers every class the teacher deems plausible. |
| [`reverse_kl_divergence.py`](reverse_kl_divergence.py) | `ReverseKLDivergenceLoss`: Mode-seeking distillation; a small student concentrates on the teacher's main modes. |
