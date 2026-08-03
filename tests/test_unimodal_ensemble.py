import torch

from scripts.python_new.evaluate_unimodal_ensemble import (
    ensemble_probabilities,
    ensemble_with_gold_probabilities,
)


def test_unimodal_ensemble_uses_equal_weight_probability_mean():
    left = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    right = torch.tensor([[0.0, 1.0], [3.0, 0.0]])

    results = ensemble_probabilities(left, right)

    expected = 0.5 * (torch.softmax(left, 1) + torch.softmax(right, 1))
    assert torch.allclose(results["ume_probability_mean"], expected)
    assert torch.allclose(results["unimodal_left"], torch.softmax(left, 1))
    assert torch.allclose(results["unimodal_right"], torch.softmax(right, 1))
    assert torch.allclose(
        results["ume_logit_mean_sensitivity"],
        torch.softmax(0.5 * (left + right), 1),
    )


def test_ume_plus_gold_uses_equal_weights_without_validation_selection():
    left = torch.tensor([[2.0, 0.0]])
    right = torch.tensor([[0.0, 1.0]])
    gold = torch.tensor([[1.5, -0.5]])

    results = ensemble_with_gold_probabilities(left, right, gold)

    expected_probabilities = (
        torch.softmax(left, 1)
        + torch.softmax(right, 1)
        + torch.softmax(gold, 1)
    ) / 3.0
    assert torch.allclose(
        results["ume_plus_gold_probability_mean"], expected_probabilities
    )
    assert torch.allclose(
        results["ume_plus_gold_logit_mean_sensitivity"],
        torch.softmax((left + right + gold) / 3.0, 1),
    )
