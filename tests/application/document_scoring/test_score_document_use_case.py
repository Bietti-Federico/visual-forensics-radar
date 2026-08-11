import pytest

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


def test_entity_override_replaces_the_classifier_prediction_entirely() -> None:
    bundle = EntityModelBundle(
        detectors=(),
        models=(),
        invariants=(),
        genuine_count=5,
        confirmed_fraud_count=5,
        ml_ensemble_ready=True,
    )
    # The classifier would predict TEST_ENTITY; override points at a
    # different entity that actually has a bundle configured for it.
    use_case = ScoreDocumentUseCase(_trained_entity_classifiers(), {"OVERRIDE_ENTITY": bundle})

    result = use_case.execute(
        _document("Test Producer", 50).build(), entity_override="OVERRIDE_ENTITY"
    )

    assert len(result.entity_report) == 1
    prediction = result.entity_report.predictions[0]
    assert prediction.predicted_entity == "OVERRIDE_ENTITY"
    # The classifier never saw OVERRIDE_ENTITY as a class, so it has no
    # structural confidence for it — 0.0, not an artificial 1.0. See
    # test_entity_consistency_reflects_classifiers_real_confidence_even_when_overridden
    # for the case where the override matches the classifier's own guess.
    assert prediction.confidence == 0.0
    assert prediction.classifier_id == "manual_override"

    ml_component = next(c for c in result.risk_report.components if c.name == "ml_probability")
    assert ml_component.weight > 0.0


def test_entity_consistency_reflects_classifiers_real_confidence_even_when_overridden() -> None:
    # Overriding to the SAME entity the classifier already predicted
    # should not artificially erase whatever structural confidence (or
    # lack of it) the classifier actually computed for that entity —
    # a document confirmed to be entity X that still doesn't structurally
    # resemble X is itself a real signal, not something to discard.
    classifiers = _trained_entity_classifiers()

    unforced_result = ScoreDocumentUseCase(classifiers, {}).execute(
        _document("Test Producer", 50).build()
    )

    forced_result = ScoreDocumentUseCase(classifiers, {}).execute(
        _document("Test Producer", 50).build(), entity_override="TEST_ENTITY"
    )

    entity_consistency_unforced = next(
        c for c in unforced_result.risk_report.components if c.name == "entity_consistency"
    )
    entity_consistency_forced = next(
        c for c in forced_result.risk_report.components if c.name == "entity_consistency"
    )
    assert entity_consistency_forced.score == pytest.approx(entity_consistency_unforced.score)


def test_entity_override_forces_invariant_check_past_the_confidence_gate() -> None:
    # A learned invariant that the classifier's own (low) confidence would
    # normally gate out of consideration should still fire once a human
    # has confirmed the entity — that confirmation is exactly what removes
    # "not confident which entity this is" as a reason to skip the check.
    bundle = EntityModelBundle(
        detectors=(),
        models=(),
        invariants=(LearnedInvariant("catalog.has_acroform", True),),
        genuine_count=6,
        confirmed_fraud_count=0,
        ml_ensemble_ready=False,
    )
    # No classifiers at all: the "auto" prediction list is empty, so a
    # plain (non-overridden) call has no entity to even gate on.
    use_case = ScoreDocumentUseCase([], {"TEST_ENTITY": bundle})

    result = use_case.execute(_document("Test Producer", 50).build(), entity_override="TEST_ENTITY")

    assert any("formulario o firma digital" in reason for reason in result.explanation.reasons)
