"""Entity classifier plugins.

Explicitly wired (not auto-registered): `default_entity_classifiers()` below
is the one place to touch when adding, removing, or swapping a classifier.

Deliberately just one classifier this iteration — the training corpus (a
handful of real documents per entity, plus their transformations/synthetic
variants) is far too small to justify an ensemble the way Module 6's six
models are; a single RandomForest is honest about what this dataset size can
actually support.
"""

from __future__ import annotations

from pdf_forensics.application.entity_identification.ports import EntityClassifierPlugin
from pdf_forensics.plugins.entity_identification.random_forest_entity_classifier import (
    RandomForestEntityClassifier,
)


def default_entity_classifiers() -> tuple[EntityClassifierPlugin, ...]:
    return (RandomForestEntityClassifier(),)


__all__ = ["default_entity_classifiers"]
