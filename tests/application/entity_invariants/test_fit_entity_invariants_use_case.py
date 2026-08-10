from pdf_forensics.application.entity_invariants.fit_entity_invariants_use_case import (
    HISTOGRAM_KEY_SEPARATOR,
    FitEntityInvariantsUseCase,
)
from pdf_forensics.domain.entity_invariants.learned_invariant import LearnedInvariant
from pdf_forensics.domain.features.enums import FeatureCategory, FeatureType
from pdf_forensics.domain.features.feature import Feature
from pdf_forensics.domain.features.feature_set import FeatureSet


def _feature(name: str, value: object, value_type: FeatureType) -> Feature:
    return Feature(
        name=name,
        value=value,
        value_type=value_type,
        description="test feature",
        source="test",
        confidence=1.0,
        category=FeatureCategory.GENERAL,
        schema_version="1.0.0",
    )


def _feature_set(**typed_values: tuple[object, FeatureType]) -> FeatureSet:
    return FeatureSet(
        features=[
            _feature(name, value, value_type) for name, (value, value_type) in typed_values.items()
        ]
    )


def test_boolean_feature_constant_across_samples_is_learned() -> None:
    feature_sets = [_feature_set(has_acroform=(True, FeatureType.BOOLEAN)) for _ in range(4)]

    invariants = FitEntityInvariantsUseCase().execute(feature_sets)

    assert LearnedInvariant("has_acroform", True) in invariants


def test_boolean_feature_that_varies_is_not_learned() -> None:
    feature_sets = [
        _feature_set(has_acroform=(True, FeatureType.BOOLEAN)),
        _feature_set(has_acroform=(False, FeatureType.BOOLEAN)),
        _feature_set(has_acroform=(True, FeatureType.BOOLEAN)),
        _feature_set(has_acroform=(True, FeatureType.BOOLEAN)),
    ]

    invariants = FitEntityInvariantsUseCase().execute(feature_sets)

    assert all(inv.feature_name != "has_acroform" for inv in invariants)


def test_histogram_key_constant_across_samples_is_learned_as_derived_feature() -> None:
    feature_sets = [
        _feature_set(filter_histogram=({"DCTDecode": 2, "FlateDecode": 5}, FeatureType.DICT))
        for _ in range(4)
    ]

    invariants = FitEntityInvariantsUseCase().execute(feature_sets)

    assert LearnedInvariant(f"filter_histogram{HISTOGRAM_KEY_SEPARATOR}DCTDecode", 2) in invariants
    assert (
        LearnedInvariant(f"filter_histogram{HISTOGRAM_KEY_SEPARATOR}FlateDecode", 5) in invariants
    )


def test_histogram_key_absent_in_some_samples_counts_as_zero_not_skipped() -> None:
    feature_sets = [
        _feature_set(filter_histogram=({"DCTDecode": 0}, FeatureType.DICT)),
        _feature_set(filter_histogram=({}, FeatureType.DICT)),
        _feature_set(filter_histogram=({"DCTDecode": 0}, FeatureType.DICT)),
        _feature_set(filter_histogram=({}, FeatureType.DICT)),
    ]

    invariants = FitEntityInvariantsUseCase().execute(feature_sets)

    assert LearnedInvariant(f"filter_histogram{HISTOGRAM_KEY_SEPARATOR}DCTDecode", 0) in invariants


def test_histogram_key_that_varies_is_not_learned() -> None:
    feature_sets = [
        _feature_set(filter_histogram=({"DCTDecode": 1}, FeatureType.DICT)),
        _feature_set(filter_histogram=({"DCTDecode": 2}, FeatureType.DICT)),
        _feature_set(filter_histogram=({"DCTDecode": 1}, FeatureType.DICT)),
        _feature_set(filter_histogram=({"DCTDecode": 1}, FeatureType.DICT)),
    ]

    invariants = FitEntityInvariantsUseCase().execute(feature_sets)

    assert not invariants


def test_continuous_scalar_features_are_never_mined_even_if_coincidentally_constant() -> None:
    feature_sets = [
        _feature_set(
            file_size_bytes=(217950, FeatureType.INTEGER),
            producer=("Acme PDF", FeatureType.STRING),
            ratio=(1.5, FeatureType.FLOAT),
        )
        for _ in range(4)
    ]

    invariants = FitEntityInvariantsUseCase().execute(feature_sets)

    assert invariants == ()


def test_empty_batch_yields_no_invariants() -> None:
    assert FitEntityInvariantsUseCase().execute([]) == ()
