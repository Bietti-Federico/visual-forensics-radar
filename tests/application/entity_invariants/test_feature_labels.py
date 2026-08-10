from pdf_forensics.application.entity_invariants.feature_labels import (
    humanize_feature,
    humanize_value,
)


def test_known_boolean_feature_gets_a_label() -> None:
    assert (
        humanize_feature("catalog.has_acroform") == "tiene formulario o firma digital (/AcroForm)"
    )


def test_unknown_feature_falls_back_to_raw_name() -> None:
    assert humanize_feature("some.future_feature") == "some.future_feature"


def test_known_histogram_key_gets_a_translated_type_name() -> None:
    assert (
        humanize_feature("objects.type_histogram::array") == "cantidad de objetos de tipo arreglo"
    )


def test_filter_name_in_histogram_key_is_left_untranslated() -> None:
    assert (
        humanize_feature("streams.filter_histogram::DCTDecode")
        == "cantidad de streams con filtro DCTDecode"
    )


def test_unknown_histogram_base_falls_back_to_raw_name() -> None:
    assert humanize_feature("some.future_histogram::key") == "some.future_histogram::key"


def test_humanize_value_renders_booleans_in_spanish() -> None:
    assert humanize_value(True) == "Sí"
    assert humanize_value(False) == "No"


def test_humanize_value_renders_other_types_as_str() -> None:
    assert humanize_value(4) == "4"
    assert humanize_value("mPDF 5.7") == "mPDF 5.7"
