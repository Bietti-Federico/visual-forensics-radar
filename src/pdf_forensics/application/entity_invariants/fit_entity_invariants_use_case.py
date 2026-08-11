"""Mines `LearnedInvariant`s automatically from a batch of genuine training
documents, replacing what used to be a hand-written, hand-maintained
per-entity dict (`plugins/rules/entity_template_mismatch_rule.py`, retired).

Deliberately restricted to two feature shapes: boolean structural flags,
checked directly, and per-key counts derived from dict-valued histogram
features (e.g. `streams.filter_histogram`), flattened into one synthetic
feature per key seen (`streams.filter_histogram::DCTDecode`). See
`docs/DOCUMENTACION.md` for why continuous scalar features (byte counts,
object counts, page counts, ...) are excluded from mining — those vary with
a document's genuine content and would turn a coincidental match in a small
fitting batch into constant false positives once the corpus grows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pdf_forensics.domain.entity_invariants.learned_invariant import LearnedInvariant
from pdf_forensics.domain.features.enums import FeatureType
from pdf_forensics.domain.features.feature import Feature
from pdf_forensics.domain.features.feature_set import FeatureSet

#: Separator between a dict-valued feature's name and one of its keys in a
#: synthesized invariant feature name, e.g. `streams.filter_histogram::DCTDecode`.
HISTOGRAM_KEY_SEPARATOR = "::"


class FitEntityInvariantsUseCase:
    def execute(self, feature_sets: Sequence[FeatureSet]) -> tuple[LearnedInvariant, ...]:
        if not feature_sets:
            return ()
        # `FeatureSet.by_name` is a linear scan; looking a name up per
        # document per candidate feature (as the mining loops below do) would
        # be O(names * documents * features). Indexing each document's
        # features by name once, up front, makes every lookup below O(1).
        by_document = [
            {feature.name: feature for feature in feature_set} for feature_set in feature_sets
        ]
        return tuple(
            self._mine_boolean_invariants(by_document)
            + self._mine_histogram_invariants(by_document)
        )

    def _mine_boolean_invariants(
        self, by_document: Sequence[Mapping[str, Feature]]
    ) -> list[LearnedInvariant]:
        boolean_names = sorted(
            {
                feature.name
                for feature in by_document[0].values()
                if feature.value_type is FeatureType.BOOLEAN
            }
        )
        invariants: list[LearnedInvariant] = []
        for name in boolean_names:
            values: list[bool] = []
            for document in by_document:
                feature = document.get(name)
                if feature is None or not isinstance(feature.value, bool):
                    values = []
                    break
                values.append(feature.value)
            if values and len(set(values)) == 1:
                invariants.append(LearnedInvariant(name, values[0]))
        return invariants

    def _mine_histogram_invariants(
        self, by_document: Sequence[Mapping[str, Feature]]
    ) -> list[LearnedInvariant]:
        dict_names = sorted(
            {
                feature.name
                for feature in by_document[0].values()
                if feature.value_type is FeatureType.DICT
            }
        )
        invariants: list[LearnedInvariant] = []
        for name in dict_names:
            histograms: list[Mapping[str, Any]] = []
            for document in by_document:
                feature = document.get(name)
                if feature is None or not isinstance(feature.value, Mapping):
                    histograms = []
                    break
                histograms.append(feature.value)
            if not histograms:
                continue

            # A key present in some documents but absent in others is still
            # a candidate invariant — absence means a count of 0, not "no data".
            all_keys = sorted({key for histogram in histograms for key in histogram})
            for key in all_keys:
                counts = {histogram.get(key, 0) for histogram in histograms}
                if len(counts) == 1:
                    invariants.append(
                        LearnedInvariant(f"{name}{HISTOGRAM_KEY_SEPARATOR}{key}", counts.pop())
                    )
        return invariants
