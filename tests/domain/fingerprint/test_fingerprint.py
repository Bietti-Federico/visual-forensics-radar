from pdf_forensics.domain.fingerprint.fingerprint import PdfFingerprint


def _fingerprint(**overrides: object) -> PdfFingerprint:
    defaults: dict[str, object] = {
        "generator": "Acme Creator",
        "producer": "Acme Producer",
        "pdf_version": "1.7",
        "xref_hash": "aaa",
        "structure_hash": "bbb",
        "metadata_hash": "ccc",
        "font_hash": None,
        "page_tree_hash": "ddd",
        "feature_hash": "eee",
        "algorithm": "sha256",
        "schema_version": "1.0.0",
    }
    defaults.update(overrides)
    return PdfFingerprint(**defaults)  # type: ignore[arg-type]


def test_to_dict_shape() -> None:
    fingerprint = _fingerprint()
    assert fingerprint.to_dict() == {
        "generator": "Acme Creator",
        "producer": "Acme Producer",
        "pdf_version": "1.7",
        "xref_hash": "aaa",
        "structure_hash": "bbb",
        "metadata_hash": "ccc",
        "font_hash": None,
        "page_tree_hash": "ddd",
        "feature_hash": "eee",
        "algorithm": "sha256",
        "schema_version": "1.0.0",
    }


def test_matching_fields_identical_fingerprints() -> None:
    a = _fingerprint()
    b = _fingerprint()
    assert a.matching_fields(b) == {
        "xref_hash",
        "structure_hash",
        "metadata_hash",
        "page_tree_hash",
        "feature_hash",
    }


def test_matching_fields_excludes_font_hash_even_when_both_none() -> None:
    a = _fingerprint(font_hash=None)
    b = _fingerprint(font_hash=None)
    assert "font_hash" not in a.matching_fields(b)


def test_matching_fields_partial_overlap() -> None:
    a = _fingerprint(xref_hash="aaa", structure_hash="bbb")
    b = _fingerprint(xref_hash="aaa", structure_hash="different")
    assert a.matching_fields(b) == {"metadata_hash", "page_tree_hash", "feature_hash", "xref_hash"}


def test_matching_fields_no_overlap() -> None:
    a = _fingerprint(
        xref_hash="1", structure_hash="2", metadata_hash="3", page_tree_hash="4", feature_hash="5"
    )
    b = _fingerprint(
        xref_hash="6", structure_hash="7", metadata_hash="8", page_tree_hash="9", feature_hash="10"
    )
    assert a.matching_fields(b) == set()
