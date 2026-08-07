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
