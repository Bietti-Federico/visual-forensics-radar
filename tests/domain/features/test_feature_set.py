import pytest

from pdf_forensics.domain.features.enums import FeatureCategory, FeatureType
from pdf_forensics.domain.features.feature import Feature
from pdf_forensics.domain.features.feature_set import FeatureSet


def _feature(name: str, value, category: FeatureCategory = FeatureCategory.GENERAL) -> Feature:
    return Feature(
        name=name,
        value=value,
        value_type=FeatureType.INTEGER,
        description="test feature",
        source="test",
        confidence=1.0,
        category=category,
        schema_version="1.0.0",
    )


def test_by_category_filters_correctly() -> None:
    a = _feature("a", 1, FeatureCategory.GENERAL)
    b = _feature("b", 2, FeatureCategory.METADATA)
    feature_set = FeatureSet(features=[a, b])

    assert feature_set.by_category(FeatureCategory.GENERAL) == [a]
    assert feature_set.by_category(FeatureCategory.METADATA) == [b]
    assert feature_set.by_category(FeatureCategory.TRAILER) == []


def test_by_name_returns_matching_feature_or_none() -> None:
    a = _feature("a", 1)
    feature_set = FeatureSet(features=[a])

    assert feature_set.by_name("a") is a
    assert feature_set.by_name("missing") is None


def test_to_dict_flattens_name_to_value() -> None:
    feature_set = FeatureSet(features=[_feature("a", 1), _feature("b", 2)])
    assert feature_set.to_dict() == {"a": 1, "b": 2}


def test_to_dict_raises_on_duplicate_names() -> None:
    feature_set = FeatureSet(features=[_feature("a", 1), _feature("a", 2)])
    with pytest.raises(ValueError, match="Duplicate feature name"):
        feature_set.to_dict()


def test_len_and_iter() -> None:
    features = [_feature("a", 1), _feature("b", 2)]
    feature_set = FeatureSet(features=features)
    assert len(feature_set) == 2
    assert list(feature_set) == features


def test_empty_feature_set_defaults() -> None:
    feature_set = FeatureSet()
    assert len(feature_set) == 0
    assert feature_set.to_dict() == {}
