from pdf_forensics.application.feature_extraction.extract_features_use_case import (
    FeatureExtractionUseCase,
)
from pdf_forensics.application.fingerprinting.generate_fingerprint_use_case import (
    GenerateFingerprintUseCase,
)
from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.infrastructure.parsing.document_parser import PdfDocumentParser
from pdf_forensics.plugins.features import default_feature_extractors
from tests.fixtures.pdf_builder import PdfBuilder


def _build_and_extract(builder: PdfBuilder):
    document = PdfDocumentParser().parse(builder.build())
    feature_set = FeatureExtractionUseCase(default_feature_extractors()).execute(document)
    return document, feature_set


def _basic_builder(page_body: str = "<< /Type /Page /Parent 2 0 R >>") -> PdfBuilder:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    off2 = builder.add_object(2, 0, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    off3 = builder.add_object(3, 0, page_body)
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (2, off2, 0, "n"), (3, off3, 0, "n")],
        size=4,
        root_ref="1 0 R",
    )
    return builder


def test_fingerprint_is_deterministic() -> None:
    document, feature_set = _build_and_extract(_basic_builder())
    use_case = GenerateFingerprintUseCase()

    first = use_case.execute(document, feature_set)
    second = use_case.execute(document, feature_set)

    assert first == second


def test_font_hash_is_always_none() -> None:
    document, feature_set = _build_and_extract(_basic_builder())
    fingerprint = GenerateFingerprintUseCase().execute(document, feature_set)
    assert fingerprint.font_hash is None
    assert fingerprint.algorithm == "sha256"


def test_generator_and_producer_come_from_info_dict_not_feature_names() -> None:
    builder = PdfBuilder()
    off_info = builder.add_object(4, 0, "<< /Creator (Acme Writer) /Producer (Acme PDF) >>")
    off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    off2 = builder.add_object(2, 0, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    off3 = builder.add_object(3, 0, "<< /Type /Page /Parent 2 0 R >>")
    builder.add_classic_xref_and_trailer(
        [
            (0, 0, 65535, "f"),
            (1, off1, 0, "n"),
            (2, off2, 0, "n"),
            (3, off3, 0, "n"),
            (4, off_info, 0, "n"),
        ],
        size=5,
        root_ref="1 0 R",
        extra_trailer="/Info 4 0 R",
    )
    document, feature_set = _build_and_extract(builder)

    fingerprint = GenerateFingerprintUseCase().execute(document, feature_set)

    assert fingerprint.generator == "Acme Writer"
    assert fingerprint.producer == "Acme PDF"
    assert fingerprint.pdf_version == "1.7"


def test_generator_and_producer_none_without_info_dict() -> None:
    document, feature_set = _build_and_extract(_basic_builder())
    fingerprint = GenerateFingerprintUseCase().execute(document, feature_set)
    assert fingerprint.generator is None
    assert fingerprint.producer is None


def test_structure_hash_ignores_byte_offsets() -> None:
    # Two documents with the exact same object-graph shape but different padding
    # before the objects, so byte offsets differ, should still structurally match.
    document_a, feature_set_a = _build_and_extract(_basic_builder())

    builder_b = PdfBuilder()
    builder_b.add_raw(b"%padding to shift every offset\n")
    off1 = builder_b.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    off2 = builder_b.add_object(2, 0, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    off3 = builder_b.add_object(3, 0, "<< /Type /Page /Parent 2 0 R >>")
    builder_b.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (2, off2, 0, "n"), (3, off3, 0, "n")],
        size=4,
        root_ref="1 0 R",
    )
    document_b, feature_set_b = _build_and_extract(builder_b)

    use_case = GenerateFingerprintUseCase()
    fingerprint_a = use_case.execute(document_a, feature_set_a)
    fingerprint_b = use_case.execute(document_b, feature_set_b)

    assert fingerprint_a.structure_hash == fingerprint_b.structure_hash
    assert fingerprint_a.xref_hash != fingerprint_b.xref_hash  # offsets really did change


def test_metadata_hash_changes_with_metadata() -> None:
    def build_with_title(title: str) -> tuple:
        builder = PdfBuilder()
        off_info = builder.add_object(4, 0, f"<< /Title ({title}) >>")
        off1 = builder.add_object(1, 0, "<< /Type /Catalog >>")
        builder.add_classic_xref_and_trailer(
            [(0, 0, 65535, "f"), (1, off1, 0, "n"), (4, off_info, 0, "n")],
            size=5,
            root_ref="1 0 R",
            extra_trailer="/Info 4 0 R",
        )
        return _build_and_extract(builder)

    document_a, feature_set_a = build_with_title("First")
    document_b, feature_set_b = build_with_title("Second")

    use_case = GenerateFingerprintUseCase()
    fingerprint_a = use_case.execute(document_a, feature_set_a)
    fingerprint_b = use_case.execute(document_b, feature_set_b)

    assert fingerprint_a.metadata_hash != fingerprint_b.metadata_hash


def test_page_tree_hash_uses_sentinel_when_pages_unresolvable() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 9 0 R >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n")], size=2, root_ref="1 0 R"
    )
    document, feature_set = _build_and_extract(builder)

    fingerprint = GenerateFingerprintUseCase().execute(document, feature_set)

    other_document, other_feature_set = _build_and_extract(builder)
    other_fingerprint = GenerateFingerprintUseCase().execute(other_document, other_feature_set)

    # Deterministic even though /Pages never resolves.
    assert fingerprint.page_tree_hash == other_fingerprint.page_tree_hash


def test_feature_hash_matches_recomputed_feature_set_hash() -> None:
    import hashlib
    import json

    document, feature_set = _build_and_extract(_basic_builder())
    fingerprint = GenerateFingerprintUseCase().execute(document, feature_set)

    expected = hashlib.sha256(
        json.dumps(feature_set.to_dict(), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    assert fingerprint.feature_hash == expected


def test_json_fallback_converts_tuples_and_stringifies_everything_else() -> None:
    fallback = GenerateFingerprintUseCase._json_fallback
    assert fallback((1, 2, 3)) == [1, 2, 3]
    assert fallback(object()).startswith("<object object")


def test_full_pipeline_integration() -> None:
    document = PdfDocumentParser().parse(_basic_builder().build())
    feature_set: FeatureSet = FeatureExtractionUseCase(default_feature_extractors()).execute(
        document
    )
    fingerprint = GenerateFingerprintUseCase().execute(document, feature_set)

    assert fingerprint.pdf_version == "1.7"
    assert len(fingerprint.xref_hash) == 64  # sha256 hex digest length
    assert len(fingerprint.feature_hash) == 64
