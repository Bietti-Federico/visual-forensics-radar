from pdf_forensics.application.feature_extraction.extract_features_use_case import (
    FeatureExtractionUseCase,
)
from pdf_forensics.application.rule_engine.evaluate_rules_use_case import EvaluateRulesUseCase
from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.infrastructure.parsing.document_parser import PdfDocumentParser
from pdf_forensics.plugins.features import default_feature_extractors
from pdf_forensics.plugins.rules import default_rules
from tests.fixtures.pdf_builder import PdfBuilder


def test_full_pipeline_flags_missing_info_and_duplicate_object_id() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog >>")
    off1_dup = builder.add_object(1, 0, "<< /Type /Catalog /Extra true >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (1, off1_dup, 0, "n")],
        size=2,
        root_ref="1 0 R",
    )
    document = PdfDocumentParser().parse(builder.build())
    feature_set = FeatureExtractionUseCase(default_feature_extractors()).execute(document)

    report = EvaluateRulesUseCase(default_rules()).execute(feature_set)

    rule_ids = {finding.rule_id for finding in report}
    assert "missing_info_dictionary" in rule_ids
    assert "duplicate_object_ids" in rule_ids
    assert len(report.by_severity(AnomalySeverity.WARNING)) >= 2


def test_clean_document_triggers_no_findings() -> None:
    builder = PdfBuilder()
    off_info = builder.add_object(
        2, 0, "<< /Title (Report) /CreationDate (D:20260101120000Z) /ModDate (D:20260102120000Z) >>"
    )
    off1 = builder.add_object(1, 0, "<< /Type /Catalog >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (2, off_info, 0, "n")],
        size=3,
        root_ref="1 0 R",
        extra_trailer="/Info 2 0 R",
    )
    document = PdfDocumentParser().parse(builder.build())
    feature_set = FeatureExtractionUseCase(default_feature_extractors()).execute(document)

    report = EvaluateRulesUseCase(default_rules()).execute(feature_set)

    assert len(report) == 0
