import pytest

from pdf_forensics.application.ml_shared.sample_weighting import apply_sample_weight


def test_none_weights_returns_items_unchanged() -> None:
    assert apply_sample_weight(["a", "b", "c"], None) == ["a", "b", "c"]


def test_weight_one_is_a_no_op() -> None:
    assert apply_sample_weight(["a", "b"], [1.0, 1.0]) == ["a", "b"]


def test_integer_weights_duplicate_items() -> None:
    assert apply_sample_weight(["a", "b"], [3.0, 1.0]) == ["a", "a", "a", "b"]


def test_fractional_weights_round_to_nearest() -> None:
    assert apply_sample_weight(["a", "b"], [2.4, 2.6]) == ["a", "a", "b", "b", "b"]


def test_zero_weight_excludes_item() -> None:
    assert apply_sample_weight(["a", "b"], [0.0, 1.0]) == ["b"]


def test_negative_weight_excludes_item() -> None:
    assert apply_sample_weight(["a", "b"], [-5.0, 1.0]) == ["b"]


def test_mismatched_length_raises() -> None:
    with pytest.raises(ValueError, match="sample_weight has"):
        apply_sample_weight(["a", "b"], [1.0])
