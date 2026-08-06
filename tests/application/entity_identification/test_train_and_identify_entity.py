from pdf_forensics.application.entity_identification.identify_entity_use_case import (
    IdentifyEntityUseCase,
)
from pdf_forensics.application.entity_identification.train_entity_classifier_use_case import (
    TrainEntityClassifierUseCase,
)
from pdf_forensics.application.feature_extraction.extract_features_use_case import (
    FeatureExtractionUseCase,
)
from pdf_forensics.infrastructure.parsing.document_parser import PdfDocumentParser
from pdf_forensics.plugins.entity_identification import default_entity_classifiers
from pdf_forensics.plugins.features import default_feature_extractors
from tests.fixtures.pdf_builder import PdfBuilder


def _feature_set_for(builder: PdfBuilder):
    document = PdfDocumentParser().parse(builder.build())
    return FeatureExtractionUseCase(default_feature_extractors()).execute(document)


def _document(producer: str, variant: int) -> PdfBuilder:
    builder = PdfBuilder()
    off_info = builder.add_object(4, 0, f"<< /Producer ({producer}) >>")
    off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    off2 = builder.add_object(2, 0, f"<< /Type /Pages /Kids [] /Count {variant} >>")
    builder.add_classic_xref_and_trailer(
        [
            (0, 0, 65535, "f"),
            (1, off1, 0, "n"),
            (2, off2, 0, "n"),
            (4, off_info, 0, "n"),
        ],
        size=5,
        root_ref="1 0 R",
        extra_trailer="/Info 4 0 R",
    )
    return builder


def test_train_and_identify_recognizes_entity_by_producer() -> None:
    feature_sets = []
    entity_labels = []
    for entity, producer in (
        ("ANSES", "iTextSharp 5.5.13.4"),
        ("LA_RIOJA", "mPDF 5.7"),
        ("JUJUY", "iTextSharp 5.5.8"),
    ):
        for variant in range(6):
            feature_sets.append(_feature_set_for(_document(producer, variant)))
            entity_labels.append(entity)

    classifiers = default_entity_classifiers()
    TrainEntityClassifierUseCase(classifiers).execute(feature_sets, entity_labels)

    held_out = _feature_set_for(_document("mPDF 5.7", 99))
    report = IdentifyEntityUseCase(classifiers).execute(held_out)

    assert len(report) == 1
    prediction = report.by_classifier("random_forest_entity")
    assert prediction is not None
    assert prediction.predicted_entity == "LA_RIOJA"
    assert prediction.confidence > 0.5
    assert set(prediction.probabilities) == {"ANSES", "LA_RIOJA", "JUJUY"}


def test_sample_weight_shifts_confidence_toward_upweighted_entity() -> None:
    feature_sets = [_feature_set_for(_document("Ambiguous Producer", i)) for i in range(6)]
    entity_labels = ["ANSES"] * 3 + ["LA_RIOJA"] * 3

    baseline_classifiers = default_entity_classifiers()
    TrainEntityClassifierUseCase(baseline_classifiers).execute(feature_sets, entity_labels)

    weighted_classifiers = default_entity_classifiers()
    sample_weight = [1.0, 1.0, 1.0, 20.0, 20.0, 20.0]
    TrainEntityClassifierUseCase(weighted_classifiers).execute(
        feature_sets, entity_labels, sample_weight=sample_weight
    )

    held_out = _feature_set_for(_document("Ambiguous Producer", 50))
    baseline_confidence = (
        IdentifyEntityUseCase(baseline_classifiers)
        .execute(held_out)
        .by_classifier("random_forest_entity")
        .probabilities["LA_RIOJA"]
    )
    weighted_confidence = (
        IdentifyEntityUseCase(weighted_classifiers)
        .execute(held_out)
        .by_classifier("random_forest_entity")
        .probabilities["LA_RIOJA"]
    )

    assert weighted_confidence > baseline_confidence
