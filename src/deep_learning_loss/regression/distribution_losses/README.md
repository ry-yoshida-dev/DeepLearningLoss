# distribution_losses

## Overview

Proper scoring rules, mostly negative log-likelihoods, of distributions whose parameters or quantiles the network predicts. Each implements [`DistributionLoss`](../distribution_loss.py).

## Components

| Component | Description |
| --------- | ----------- |
| [`beta.py`](beta.py) | `BetaNLLLoss`: Likelihood of values strictly between 0 and 1, such as rates, proportions or probabilities. |
| [`gamma.py`](gamma.py) | `GammaNLLLoss`: Likelihood of positive, right-skewed values, such as durations, amounts or rainfall. |
| [`gaussian.py`](gaussian.py) | `GaussianNLLLoss`: Default regression with uncertainty; predicts a mean and a standard deviation for each value. |
| [`gaussian_crps.py`](gaussian_crps.py) | `GaussianCRPSLoss`: Continuous ranked probability score of a Gaussian; regression with uncertainty, robust where the likelihood is not. |
| [`gaussian_mixture.py`](gaussian_mixture.py) | `GaussianMixtureNLLLoss`: Mixture density network likelihood; regression whose targets have several plausible values. |
| [`laplace.py`](laplace.py) | `LaplaceNLLLoss`: Regression with uncertainty, robust to outliers; the likelihood counterpart of the MAE. |
| [`log_normal.py`](log_normal.py) | `LogNormalNLLLoss`: Likelihood of positive values spanning orders of magnitude, such as prices, incomes or latencies. |
| [`negative_binomial.py`](negative_binomial.py) | `NegativeBinomialNLLLoss`: Likelihood of overdispersed counts, whose variance exceeds their mean, such as intermittent demand. |
| [`poisson.py`](poisson.py) | `PoissonNLLLoss`: Default likelihood of counts, such as demand or events, whose variance equals their mean. |
| [`quantile.py`](quantile.py) | `QuantileLoss`: Pinball loss averaged over several quantile levels; distribution-free forecasts and prediction intervals. |
| [`student_t.py`](student_t.py) | `StudentTNLLLoss`: Heavy-tailed regression with uncertainty; a Gaussian likelihood robust to outliers. |
| [`zero_inflated_negative_binomial.py`](zero_inflated_negative_binomial.py) | `ZeroInflatedNegativeBinomialNLLLoss`: Likelihood of overdispersed counts with excess zeros, such as single-cell gene expression or intermittent demand. |
| [`zero_inflated_poisson.py`](zero_inflated_poisson.py) | `ZeroInflatedPoissonNLLLoss`: Likelihood of counts with excess zeros, such as sparse sales or rare events, around a Poisson. |
