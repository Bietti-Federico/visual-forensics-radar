from pdf_forensics.application.anomaly_detection.detect_anomalies_use_case import (
    DetectAnomaliesUseCase,
)
from pdf_forensics.application.anomaly_detection.fit_anomaly_detectors_use_case import (
    FitAnomalyDetectorsUseCase,
)
from pdf_forensics.application.feature_extraction.extract_features_use_case import (
    FeatureExtractionUseCase,
)
from pdf_forensics.infrastructure.parsing.document_parser import PdfDocumentParser
from pdf_forensics.plugins.anomaly_detection import default_detectors
from pdf_forensics.plugins.features import default_feature_extractors
from tests.fixtures.pdf_builder import PdfBuilder


def _feature_set_for(builder: PdfBuilder):
    document = PdfDocumentParser().parse(builder.build())
    return FeatureExtractionUseCase(default_feature_extractors()).execute(document)


def _normal_document(extra_pages: int) -> PdfBuilder:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    off2 = builder.add_object(2, 0, f"<< /Type /Pages /Kids [] /Count {extra_pages} >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (2, off2, 0, "n")], size=3, root_ref="1 0 R"
    )
    return builder


def _many_revisions_document() -> PdfBuilder:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    off2 = builder.add_object(2, 0, "<< /Type /Pages /Kids [] /Count 0 >>")
    prev = builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (2, off2, 0, "n")], size=3, root_ref="1 0 R"
    )
    for _ in range(6):
        off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R /Touched true >>")
        prev = builder.add_classic_xref_and_trailer(
            [(1, off1, 0, "n")], size=3, root_ref="1 0 R", prev=prev
        )
    return builder


def test_document_with_many_revisions_scores_higher_on_isolation_forest() -> None:
    normal_feature_sets = [_feature_set_for(_normal_document(i)) for i in range(15)]
    weird_feature_set = _feature_set_for(_many_revisions_document())

    detectors = default_detectors()
    FitAnomalyDetectorsUseCase(detectors).execute(normal_feature_sets)

    detect = DetectAnomaliesUseCase(detectors)
    normal_report = detect.execute(normal_feature_sets[0])
    weird_report = detect.execute(weird_feature_set)

    normal_iforest = normal_report.by_detector("isolation_forest")
    weird_iforest = weird_report.by_detector("isolation_forest")
    assert normal_iforest is not None
    assert weird_iforest is not None
    assert weird_iforest.score > normal_iforest.score

    assert len(weird_report) == 4  # all four default detectors reported a score


def test_sample_weight_lets_a_duplicated_document_look_normal() -> None:
    normal_feature_sets = [_feature_set_for(_normal_document(i)) for i in range(15)]
    weird_feature_set = _feature_set_for(_many_revisions_document())
    feature_sets = normal_feature_sets + [weird_feature_set]

    unweighted_detectors = default_detectors()
    FitAnomalyDetectorsUseCase(unweighted_detectors).execute(feature_sets)
    unweighted_score = (
        DetectAnomaliesUseCase(unweighted_detectors)
        .execute(weird_feature_set)
        .by_detector("isolation_forest")
        .score
    )

    weighted_detectors = default_detectors()
    sample_weight = [1.0] * len(normal_feature_sets) + [40.0]
    FitAnomalyDetectorsUseCase(weighted_detectors).execute(
        feature_sets, sample_weight=sample_weight
    )
    weighted_score = (
        DetectAnomaliesUseCase(weighted_detectors)
        .execute(weird_feature_set)
        .by_detector("isolation_forest")
        .score
    )

    # Duplicated 40x, the "weird" document is no longer isolated relative to
    # its own copies — it should look markedly less anomalous than when it
    # was fit alongside the normal batch only once.
    assert weighted_score < unweighted_score
