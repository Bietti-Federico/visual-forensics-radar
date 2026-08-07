import pytest

from pdf_forensics.domain.risk.risk_weights import RiskWeights


def test_defaults_sum_to_one() -> None:
    weights = RiskWeights()
    total = (
        weights.rule_engine
        + weights.ml_probability
        + weights.anomaly_detection
        + weights.structural
        + weights.metadata
        + weights.entity_consistency
        + weights.signature_integrity
    )
    assert total == pytest.approx(1.0)


def test_valid_custom_weights_construct_fine() -> None:
    weights = RiskWeights(
        rule_engine=0.4,
        ml_probability=0.2,
        anomaly_detection=0.15,
        structural=0.05,
        metadata=0.05,
        entity_consistency=0.1,
        signature_integrity=0.05,
    )
    assert weights.rule_engine == 0.4


def test_weights_not_summing_to_one_raise() -> None:
    with pytest.raises(ValueError, match="must sum to 1.0"):
        RiskWeights(
            rule_engine=0.5,
            ml_probability=0.5,
            anomaly_detection=0.5,
            structural=0.5,
            metadata=0.5,
            entity_consistency=0.5,
            signature_integrity=0.5,
        )


def test_without_ml_probability_sums_to_one_and_zeroes_ml() -> None:
    rescaled = RiskWeights().without_ml_probability()

    assert rescaled.ml_probability == 0.0
    total = (
        rescaled.rule_engine
        + rescaled.ml_probability
        + rescaled.anomaly_detection
        + rescaled.structural
        + rescaled.metadata
        + rescaled.entity_consistency
        + rescaled.signature_integrity
    )
    assert total == pytest.approx(1.0)


def test_without_ml_probability_preserves_relative_proportions() -> None:
    rescaled = RiskWeights().without_ml_probability()

    # rule_engine and anomaly_detection had equal-ish weight before rescale
    # (0.24 vs 0.16); the ratio between any two non-ml components must be
    # unchanged by a uniform rescale.
    original = RiskWeights()
    assert rescaled.rule_engine / rescaled.anomaly_detection == pytest.approx(
        original.rule_engine / original.anomaly_detection
    )


def test_without_ml_probability_raises_if_ml_is_the_entire_weight() -> None:
    weights = RiskWeights(
        rule_engine=0.0,
        ml_probability=1.0,
        anomaly_detection=0.0,
        structural=0.0,
        metadata=0.0,
        entity_consistency=0.0,
        signature_integrity=0.0,
    )
    with pytest.raises(ValueError, match="entire weight"):
        weights.without_ml_probability()
