from pdf_forensics.application.ml_shared.feature_vectorizer import select_numeric_features
from pdf_forensics.domain.features.enums import FeatureCategory, FeatureType
from pdf_forensics.domain.features.feature import Feature
from pdf_forensics.domain.features.feature_set import FeatureSet


def _feature(name: str, value, value_type: FeatureType) -> Feature:
    return Feature(
        name=name,
        value=value,
        value_type=value_type,
        description="d",
        source="s",
        confidence=1.0,
        category=FeatureCategory.GENERAL,
        schema_version="1.0.0",
    )


def test_keeps_integer_float_and_boolean_features() -> None:
    feature_set = FeatureSet(
        features=[
            _feature("general.object_count", 5, FeatureType.INTEGER),
            _feature("streams.avg_size", 1.5, FeatureType.FLOAT),
            _feature("security.is_encrypted", True, FeatureType.BOOLEAN),
        ]
    )
    vector = select_numeric_features(feature_set)
    assert vector == {
        "general.object_count": 5.0,
        "streams.avg_size": 1.5,
        "security.is_encrypted": 1.0,
    }


def test_excludes_string_dict_and_list_features() -> None:
    feature_set = FeatureSet(
        features=[
            _feature("metadata.producer", "Acme", FeatureType.STRING),
            _feature("objects.type_histogram", {"dictionary": 2}, FeatureType.DICT),
        ]
    )
    assert select_numeric_features(feature_set) == {}


def test_omits_none_valued_numeric_feature_instead_of_coercing_to_zero() -> None:
    feature_set = FeatureSet(
        features=[
            _feature("catalog.page_count", None, FeatureType.INTEGER),
            _feature("general.object_count", 3, FeatureType.INTEGER),
        ]
    )
    vector = select_numeric_features(feature_set)
    assert "catalog.page_count" not in vector
    assert vector["general.object_count"] == 3.0


def test_false_boolean_is_kept_as_zero() -> None:
    feature_set = FeatureSet(
        features=[_feature("security.is_encrypted", False, FeatureType.BOOLEAN)]
    )
    assert select_numeric_features(feature_set) == {"security.is_encrypted": 0.0}
