"""Flags when /Root/Version differs from the file header's %PDF-x.y version.

This is a real, spec-sanctioned override mechanism (ISO 32000-1 §7.5.2) that
lets a generator declare a later effective version without rewriting the
header — purely informational, not inherently suspicious.
"""

from __future__ import annotations

from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.domain.rules.rule_finding import RuleFinding


class PdfVersionMismatchRule:
    rule_id = "pdf_version_mismatch"

    def evaluate(self, feature_set: FeatureSet) -> RuleFinding | None:
        header_feature = feature_set.by_name("general.pdf_version")
        catalog_feature = feature_set.by_name("catalog.version")
        if header_feature is None or catalog_feature is None:
            return None

        header_version = header_feature.value
        catalog_version = catalog_feature.value
        if not header_version or not catalog_version or header_version == catalog_version:
            return None

        return RuleFinding(
            rule_id=self.rule_id,
            severity=AnomalySeverity.INFO,
            confidence=1.0,
            explanation=(
                f"El header declara versión de PDF {header_version!r} pero /Root/Version "
                f"declara {catalog_version!r}."
            ),
            references=("ISO 32000-1 §7.5.2",),
        )
