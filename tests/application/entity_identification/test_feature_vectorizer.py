from pdf_forensics.application.entity_identification.feature_vectorizer import (
    select_entity_features,
)
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


def test_includes_numeric_features_and_producer_creator_strings() -> None:
    feature_set = FeatureSet(
        features=[
            _feature("general.object_count", 5, FeatureType.INTEGER),
            _feature("metadata.producer", "Acme PDF", FeatureType.STRING),
            _feature("metadata.creator", "Acme Writer", FeatureType.STRING),
        ]
    )
    vector = select_entity_features(feature_set)
    assert vector == {
        "general.object_count": 5.0,
        "metadata.producer": "Acme PDF",
        "metadata.creator": "Acme Writer",
    }


def test_excludes_other_string_features_and_none_valued_producer() -> None:
    feature_set = FeatureSet(
        features=[
            _feature("metadata.title", "Some Title", FeatureType.STRING),
            _feature("metadata.producer", None, FeatureType.STRING),
        ]
    )
    assert select_entity_features(feature_set) == {}
