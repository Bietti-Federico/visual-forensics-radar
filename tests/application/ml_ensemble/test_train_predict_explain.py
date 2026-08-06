from pdf_forensics.application.feature_extraction.extract_features_use_case import (
    FeatureExtractionUseCase,
)
from pdf_forensics.application.ml_ensemble.explain_prediction_use_case import (
    ExplainPredictionUseCase,
)
from pdf_forensics.application.ml_ensemble.predict_use_case import PredictUseCase
from pdf_forensics.application.ml_ensemble.train_models_use_case import TrainModelsUseCase
from pdf_forensics.infrastructure.parsing.document_parser import PdfDocumentParser
from pdf_forensics.plugins.features import default_feature_extractors
from pdf_forensics.plugins.ml_ensemble import default_models
from tests.fixtures.pdf_builder import PdfBuilder


def _feature_set_for(builder: PdfBuilder):
    document = PdfDocumentParser().parse(builder.build())
    return FeatureExtractionUseCase(default_feature_extractors()).execute(document)


def _single_revision_document(variant: int) -> PdfBuilder:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    off2 = builder.add_object(2, 0, f"<< /Type /Pages /Kids [] /Count {variant} >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (2, off2, 0, "n")], size=3, root_ref="1 0 R"
    )
    return builder


def _many_revision_document(variant: int) -> PdfBuilder:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    off2 = builder.add_object(2, 0, f"<< /Type /Pages /Kids [] /Count {variant} >>")
    prev = builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (2, off2, 0, "n")], size=3, root_ref="1 0 R"
    )
    for _ in range(5):
        off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R /Touched true >>")
        prev = builder.add_classic_xref_and_trailer(
            [(1, off1, 0, "n")], size=3, root_ref="1 0 R", prev=prev
        )
    return builder


def test_train_predict_and_explain_end_to_end() -> None:
    original_feature_sets = [_feature_set_for(_single_revision_document(i)) for i in range(8)]
    manipulated_feature_sets = [_feature_set_for(_many_revision_document(i)) for i in range(8)]

    feature_sets = original_feature_sets + manipulated_feature_sets
    labels = [True] * len(original_feature_sets) + [False] * len(manipulated_feature_sets)

    models = default_models()
    TrainModelsUseCase(models).execute(feature_sets, labels)

    held_out_original = _feature_set_for(_single_revision_document(99))
    held_out_manipulated = _feature_set_for(_many_revision_document(99))

    predict = PredictUseCase(models)
    original_report = predict.execute(held_out_original)
    manipulated_report = predict.execute(held_out_manipulated)

    assert len(original_report) == 6
    assert len(manipulated_report) == 6

    random_forest_original = original_report.by_model("random_forest")
    random_forest_manipulated = manipulated_report.by_model("random_forest")
    assert random_forest_original is not None
    assert random_forest_manipulated is not None
    assert random_forest_manipulated.probability > random_forest_original.probability

    explanations = ExplainPredictionUseCase(models).execute(held_out_manipulated)
    explained_model_ids = {explanation.model_id for explanation in explanations}
    # The two ensembles don't support SHAP; the four base models do.
    assert explained_model_ids == {"random_forest", "extra_trees", "logistic_regression", "xgboost"}


def test_sample_weight_shifts_prediction_toward_upweighted_class() -> None:
    original_feature_sets = [_feature_set_for(_single_revision_document(i)) for i in range(8)]
    manipulated_feature_sets = [_feature_set_for(_many_revision_document(i)) for i in range(8)]
    feature_sets = original_feature_sets + manipulated_feature_sets
    labels = [True] * len(original_feature_sets) + [False] * len(manipulated_feature_sets)

    baseline_models = default_models()
    TrainModelsUseCase(baseline_models).execute(feature_sets, labels)

    weighted_models = default_models()
    sample_weight = [1.0] * len(original_feature_sets) + [50.0] * len(manipulated_feature_sets)
    TrainModelsUseCase(weighted_models).execute(feature_sets, labels, sample_weight=sample_weight)

    held_out_original = _feature_set_for(_single_revision_document(99))
    baseline_probability = (
        PredictUseCase(baseline_models)
        .execute(held_out_original)
        .by_model("random_forest")
        .probability
    )
    weighted_probability = (
        PredictUseCase(weighted_models)
        .execute(held_out_original)
        .by_model("random_forest")
        .probability
    )

    # Massively up-weighting the manipulated class should make the model
    # more confident that even a genuinely original held-out document looks
    # manipulated, relative to the unweighted baseline.
    assert weighted_probability > baseline_probability
