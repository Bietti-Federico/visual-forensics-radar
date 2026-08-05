"""Feature extractor plugins.

Explicitly wired (not auto-registered): `default_feature_extractors()` below
is the one place to touch when adding, removing, or swapping an extractor.

Deferred to later modules — none of these are attempted here, and none are
represented by placeholder features: font/`/Resources`/`/XObject` tree-walking,
image feature extraction, XMP semantic parsing, digital signature validation,
linearization detection, and any page-tree walk beyond the `/Pages/Count` key.
"""

from __future__ import annotations

from pdf_forensics.application.feature_extraction.ports import FeatureExtractorPlugin
from pdf_forensics.plugins.features.catalog_extractor import CatalogFeatureExtractor
from pdf_forensics.plugins.features.general_extractor import GeneralFeatureExtractor
from pdf_forensics.plugins.features.incremental_updates_extractor import (
    IncrementalUpdatesFeatureExtractor,
)
from pdf_forensics.plugins.features.metadata_extractor import MetadataFeatureExtractor
from pdf_forensics.plugins.features.objects_extractor import ObjectsFeatureExtractor
from pdf_forensics.plugins.features.security_extractor import SecurityFeatureExtractor
from pdf_forensics.plugins.features.statistics_extractor import StatisticsFeatureExtractor
from pdf_forensics.plugins.features.streams_extractor import StreamsFeatureExtractor
from pdf_forensics.plugins.features.trailer_extractor import TrailerFeatureExtractor
from pdf_forensics.plugins.features.xref_extractor import XrefFeatureExtractor


def default_feature_extractors() -> tuple[FeatureExtractorPlugin, ...]:
    return (
        GeneralFeatureExtractor(),
        MetadataFeatureExtractor(),
        TrailerFeatureExtractor(),
        CatalogFeatureExtractor(),
        XrefFeatureExtractor(),
        IncrementalUpdatesFeatureExtractor(),
        ObjectsFeatureExtractor(),
        StreamsFeatureExtractor(),
        SecurityFeatureExtractor(),
        StatisticsFeatureExtractor(),
    )


__all__ = ["default_feature_extractors"]
