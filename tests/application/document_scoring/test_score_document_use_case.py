from pdf_forensics.application.document_scoring.score_document_use_case import (
    ScoreDocumentUseCase,
)
from pdf_forensics.application.entity_identification.train_entity_classifier_use_case import (
    TrainEntityClassifierUseCase,
)
from pdf_forensics.application.feature_extraction.extract_features_use_case import (
    FeatureExtractionUseCase,
)
from pdf_forensics.application.ml_ensemble.train_models_use_case import TrainModelsUseCase
from pdf_forensics.application.model_persistence.entity_model_bundle import EntityModelBundle
from pdf_forensics.domain.entity_invariants.learned_invariant import LearnedInvariant
from pdf_forensics.infrastructure.parsing.document_parser import PdfDocumentParser
from pdf_forensics.plugins.entity_identification import default_entity_classifiers
from pdf_forensics.plugins.features import default_feature_extractors
from pdf_forensics.plugins.ml_ensemble import default_models
from tests.fixtures.pdf_builder import PdfBuilder


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


def _feature_set_for(builder: PdfBuilder):
    document = PdfDocumentParser().parse(builder.build())
    return FeatureExtractionUseCase(default_feature_extractors()).execute(document)


def _trained_entity_classifiers():
    feature_sets = [_feature_set_for(_document("Test Producer", i)) for i in range(6)]
    labels = ["TEST_ENTITY"] * 6
    classifiers = default_entity_classifiers()
    TrainEntityClassifierUseCase(classifiers).execute(feature_sets, labels)
    return classifiers


def test_full_weights_used_when_entity_ml_ensemble_ready() -> None:
    feature_sets = [_feature_set_for(_document("Test Producer", i)) for i in range(10)]
    labels = [True] * 5 + [False] * 5
    models = default_models()
    TrainModelsUseCase(models).execute(feature_sets, labels)

    bundle = EntityModelBundle(
        detectors=(),
        models=models,
        invariants=(),
        genuine_count=5,
        confirmed_fraud_count=3,
        ml_ensemble_ready=True,
    )
    use_case = ScoreDocumentUseCase(_trained_entity_classifiers(), {"TEST_ENTITY": bundle})

    result = use_case.execute(_document("Test Producer", 50).build())

    ml_component = next(c for c in result.risk_report.components if c.name == "ml_probability")
    assert ml_component.weight > 0.0


def test_reduced_weights_used_when_entity_ml_ensemble_not_ready() -> None:
    bundle = EntityModelBundle(
        detectors=(),
        models=(),
        invariants=(),
        genuine_count=1,
        confirmed_fraud_count=0,
        ml_ensemble_ready=False,
    )
    use_case = ScoreDocumentUseCase(_trained_entity_classifiers(), {"TEST_ENTITY": bundle})

    result = use_case.execute(_document("Test Producer", 50).build())

    ml_component = next(c for c in result.risk_report.components if c.name == "ml_probability")
    assert ml_component.weight == 0.0
    assert ml_component.score == 0.0


def test_no_entity_classifiers_still_scores_without_crashing() -> None:
    use_case = ScoreDocumentUseCase([], {})

    result = use_case.execute(_document("Unknown Producer", 1).build())

    assert len(result.entity_report) == 0
    assert result.risk_report.risk_score >= 0


def test_learned_invariant_violation_surfaces_as_rule_finding() -> None:
    # None of these test documents have an /AcroForm entry, so an invariant
    # claiming they always should is guaranteed to be violated.
    bundle = EntityModelBundle(
        detectors=(),
        models=(),
        invariants=(LearnedInvariant("catalog.has_acroform", True),),
        genuine_count=6,
        confirmed_fraud_count=0,
        ml_ensemble_ready=False,
    )
    use_case = ScoreDocumentUseCase(_trained_entity_classifiers(), {"TEST_ENTITY": bundle})

    result = use_case.execute(_document("Test Producer", 50).build())

    assert any("formulario o firma digital" in reason for reason in result.explanation.reasons)


def test_unrecognized_entity_falls_back_to_empty_bundle() -> None:
    bundle = EntityModelBundle(
        detectors=(),
        models=(),
        invariants=(),
        genuine_count=5,
        confirmed_fraud_count=5,
        ml_ensemble_ready=True,
    )
    # Bundle exists only for a DIFFERENT entity than what gets predicted.
    use_case = ScoreDocumentUseCase(_trained_entity_classifiers(), {"SOME_OTHER_ENTITY": bundle})

    result = use_case.execute(_document("Test Producer", 50).build())

    ml_component = next(c for c in result.risk_report.components if c.name == "ml_probability")
    assert ml_component.weight == 0.0
