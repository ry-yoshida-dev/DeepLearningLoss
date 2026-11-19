import pytest
import torch
import torch.nn.functional as functional
from torch import distributions

from deep_learning_loss import (
    BetaNLLLoss,
    DistributionLoss,
    GammaNLLLoss,
    GaussianCRPSLoss,
    GaussianMixtureNLLLoss,
    GaussianNLLLoss,
    LaplaceNLLLoss,
    LogNormalNLLLoss,
    NegativeBinomialNLLLoss,
    PoissonNLLLoss,
    QuantileLoss,
    StudentTNLLLoss,
    ZeroInflatedNegativeBinomialNLLLoss,
    ZeroInflatedPoissonNLLLoss,
)

PARAMETERS: torch.Tensor = torch.tensor([[0.5, 0.2], [-1.0, -0.7], [2.0, 1.5]])
TARGETS: torch.Tensor = torch.tensor([0.0, -2.0, 3.5])
COUNTS: torch.Tensor = torch.tensor([0.0, 2.0, 7.0])
SCALES: torch.Tensor = functional.softplus(PARAMETERS[:, 1]) + 1e-6


def test_continuous_likelihoods_match_torch_distributions() -> None:
    cases: list[tuple[DistributionLoss, distributions.Distribution]] = [
        (GaussianNLLLoss(), distributions.Normal(PARAMETERS[:, 0], SCALES)),
        (LaplaceNLLLoss(), distributions.Laplace(PARAMETERS[:, 0], SCALES)),
        (StudentTNLLLoss(degrees_of_freedom=4.0), distributions.StudentT(4.0, PARAMETERS[:, 0], SCALES)),
    ]
    for loss, distribution in cases:
        expected: torch.Tensor = -distribution.log_prob(TARGETS)
        assert torch.allclose(loss.compute_elementwise(PARAMETERS, TARGETS), expected, atol=1e-5)


def test_count_likelihoods_match_torch_distributions() -> None:
    log_rates = PARAMETERS[:, :1]
    poisson: torch.Tensor = -distributions.Poisson(log_rates[:, 0].exp()).log_prob(COUNTS)
    assert torch.allclose(PoissonNLLLoss().compute_elementwise(log_rates, COUNTS), poisson, atol=1e-5)

    means = PARAMETERS[:, 0].exp()
    total_counts = 1.0 / (functional.softplus(PARAMETERS[:, 1]) + 1e-6)
    negative_binomial = distributions.NegativeBinomial(total_counts, logits=(means / total_counts).log())
    expected: torch.Tensor = -negative_binomial.log_prob(COUNTS)
    assert torch.allclose(NegativeBinomialNLLLoss().compute_elementwise(PARAMETERS, COUNTS), expected, atol=1e-4)


def test_means_decode_parameters() -> None:
    assert torch.equal(GaussianNLLLoss().compute_mean(PARAMETERS), PARAMETERS[:, 0])
    assert torch.allclose(PoissonNLLLoss().compute_mean(PARAMETERS[:, :1]), PARAMETERS[:, 0].exp())
    assert torch.allclose(NegativeBinomialNLLLoss().compute_mean(PARAMETERS), PARAMETERS[:, 0].exp())


def test_distribution_losses_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="last axis of size 2"):
        GaussianNLLLoss().compute(PARAMETERS[:, :1], TARGETS)
    with pytest.raises(ValueError, match="targets shape"):
        GaussianNLLLoss().compute(PARAMETERS, TARGETS[:2])
    with pytest.raises(ValueError, match="last axis of size 1"):
        PoissonNLLLoss().compute_mean(PARAMETERS)


def test_distribution_losses_mask_values() -> None:
    is_included = torch.tensor([True, False, True])
    masked = GaussianNLLLoss().compute_masked(PARAMETERS, TARGETS, is_included)
    elementwise = GaussianNLLLoss().compute_elementwise(PARAMETERS, TARGETS)
    assert masked.item() == pytest.approx(elementwise[is_included].mean().item())


def test_distribution_losses_reject_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="minimum_scale"):
        GaussianNLLLoss(minimum_scale=0.0)
    with pytest.raises(ValueError, match="degrees_of_freedom"):
        StudentTNLLLoss(degrees_of_freedom=1.0)
    with pytest.raises(ValueError, match="minimum_dispersion"):
        NegativeBinomialNLLLoss(minimum_dispersion=0.0)


def test_positive_and_bounded_likelihoods_match_torch_distributions() -> None:
    positive_targets = torch.tensor([0.5, 2.0, 7.0])
    shapes = functional.softplus(PARAMETERS[:, 1]) + 1e-6
    gamma = distributions.Gamma(shapes, shapes / PARAMETERS[:, 0].exp())
    assert torch.allclose(
        GammaNLLLoss().compute_elementwise(PARAMETERS, positive_targets), -gamma.log_prob(positive_targets), atol=1e-5
    )

    log_normal = distributions.LogNormal(PARAMETERS[:, 0], SCALES)
    expected = -log_normal.log_prob(positive_targets)
    assert torch.allclose(LogNormalNLLLoss().compute_elementwise(PARAMETERS, positive_targets), expected, atol=1e-5)
    assert torch.allclose(LogNormalNLLLoss().compute_mean(PARAMETERS), log_normal.mean, atol=1e-5)

    proportions = torch.tensor([0.1, 0.5, 0.95])
    means = torch.sigmoid(PARAMETERS[:, 0])
    concentrations = functional.softplus(PARAMETERS[:, 1]) + 1e-6
    beta = distributions.Beta(means * concentrations, (1.0 - means) * concentrations)
    assert torch.allclose(
        BetaNLLLoss().compute_elementwise(PARAMETERS, proportions), -beta.log_prob(proportions), atol=1e-4
    )


def test_gaussian_mixture_matches_torch_mixture() -> None:
    predictions = torch.randn(5, 6)
    targets = torch.randn(5)
    logits, means, raw_scales = predictions[:, :2], predictions[:, 2:4], predictions[:, 4:]
    mixture = distributions.MixtureSameFamily(
        distributions.Categorical(logits=logits),
        distributions.Normal(means, functional.softplus(raw_scales) + 1e-6),
    )
    loss = GaussianMixtureNLLLoss(components=2)
    assert torch.allclose(loss.compute_elementwise(predictions, targets), -mixture.log_prob(targets), atol=1e-5)
    assert torch.allclose(loss.compute_mean(predictions), mixture.mean, atol=1e-5)


def test_zero_inflated_likelihoods_mix_a_point_mass_at_zero() -> None:
    predictions = torch.tensor([[0.5, -1.0, 0.3], [1.2, 0.4, -0.5]])
    counts = torch.tensor([0.0, 3.0])
    zero_probabilities = torch.sigmoid(predictions[:, 2])

    poisson = distributions.Poisson(predictions[:, 0].exp())
    poisson_densities = poisson.log_prob(counts).exp()
    inflated = torch.where(counts == 0, zero_probabilities, 0.0) + (1.0 - zero_probabilities) * poisson_densities
    zero_inflated_poisson = ZeroInflatedPoissonNLLLoss().compute_elementwise(predictions[:, [0, 2]], counts)
    assert torch.allclose(zero_inflated_poisson, -inflated.log(), atol=1e-5)

    total_counts = 1.0 / (functional.softplus(predictions[:, 1]) + 1e-6)
    negative_binomial = distributions.NegativeBinomial(
        total_counts, logits=(predictions[:, 0].exp() / total_counts).log()
    )
    densities = negative_binomial.log_prob(counts).exp()
    inflated = torch.where(counts == 0, zero_probabilities, 0.0) + (1.0 - zero_probabilities) * densities
    assert torch.allclose(
        ZeroInflatedNegativeBinomialNLLLoss().compute_elementwise(predictions, counts), -inflated.log(), atol=1e-5
    )
    expected_means = (1.0 - zero_probabilities) * predictions[:, 0].exp()
    assert torch.allclose(ZeroInflatedNegativeBinomialNLLLoss().compute_mean(predictions), expected_means)


def test_gaussian_crps_matches_numerical_integration() -> None:
    grid = torch.linspace(-30.0, 30.0, 600_001, dtype=torch.float64)
    parameters = PARAMETERS.double()
    scales = functional.softplus(parameters[:, 1]) + 1e-6
    for index in range(3):
        cumulative = distributions.Normal(parameters[index, 0], scales[index]).cdf(grid)
        step = (grid >= TARGETS[index].double()).double()
        expected = torch.trapezoid((cumulative - step).pow(2), grid)
        loss = GaussianCRPSLoss().compute_elementwise(
            parameters[index : index + 1], TARGETS[index : index + 1].double()
        )
        # The trapezoid rule misses up to half a grid step at the jump of the step function.
        assert loss.item() == pytest.approx(expected.item(), abs=1e-4)


def test_quantile_loss_averages_pinball_losses_and_integrates_the_mean() -> None:
    loss = QuantileLoss(levels=(0.1, 0.5, 0.9))
    predictions = torch.tensor([[1.0, 2.0, 3.0]])
    targets = torch.tensor([2.5])
    # Errors 1.5, 0.5, -0.5 at levels 0.1, 0.5, 0.9.
    expected = (0.1 * 1.5 + 0.5 * 0.5 + 0.1 * 0.5) / 3.0
    assert loss.compute(predictions, targets).item() == pytest.approx(expected)
    assert loss.compute_mean(predictions).item() == pytest.approx(0.3 * 1.0 + 0.4 * 2.0 + 0.3 * 3.0)


def test_masking_replaces_out_of_support_targets() -> None:
    predictions = torch.zeros(2, 2, requires_grad=True)
    is_included = torch.tensor([True, False])
    for loss, targets in [
        (BetaNLLLoss(), torch.tensor([0.3, float("nan")])),
        (GammaNLLLoss(), torch.tensor([1.5, -1.0])),
        (LogNormalNLLLoss(), torch.tensor([1.5, 0.0])),
    ]:
        value = loss.compute_masked(predictions, targets, is_included)
        (gradient,) = torch.autograd.grad(value, predictions)
        assert torch.isfinite(value) and torch.isfinite(gradient).all()
        assert value.item() == pytest.approx(loss.compute_elementwise(predictions, targets)[0].item())


def test_negative_binomial_stays_finite_for_large_means() -> None:
    predictions = torch.tensor([[100.0, 0.0]], requires_grad=True)
    value = NegativeBinomialNLLLoss().compute(predictions, torch.tensor([3.0]))
    (gradient,) = torch.autograd.grad(value, predictions)
    assert torch.isfinite(value) and torch.isfinite(gradient).all()


def test_new_distribution_losses_reject_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="levels"):
        QuantileLoss(levels=())
    with pytest.raises(ValueError, match="increasing"):
        QuantileLoss(levels=(0.5, 0.1))
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        QuantileLoss(levels=(0.0, 0.5))
    with pytest.raises(ValueError, match="components"):
        GaussianMixtureNLLLoss(components=0)
    with pytest.raises(ValueError, match="minimum_shape"):
        GammaNLLLoss(minimum_shape=0.0)
    with pytest.raises(ValueError, match="minimum_concentration"):
        BetaNLLLoss(minimum_concentration=0.0)
    with pytest.raises(ValueError, match="last axis of size 3"):
        QuantileLoss().compute(PARAMETERS, TARGETS)
