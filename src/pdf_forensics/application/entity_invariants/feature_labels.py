"""Human-readable Spanish labels for the bounded set of features that can
ever become a `LearnedInvariant` (`FitEntityInvariantsUseCase` only mines
`FeatureType.BOOLEAN` and `FeatureType.DICT` features — see its own
docstring), so a violation reads as a sentence instead of a raw feature
name and Python repr.

Bounded and generic, not per-entity: every feature name here comes from
`plugins/features/*.py` and applies to every entity the same way — adding
a new entity never needs an entry here. A feature this platform doesn't
yet have a label for still displays (as its raw technical name), it's
just less polished — this is presentation, not correctness.
"""

from __future__ import annotations

from typing import Any

from pdf_forensics.application.entity_invariants.fit_entity_invariants_use_case import (
    HISTOGRAM_KEY_SEPARATOR,
)

_BOOLEAN_FEATURE_LABELS = {
    "catalog.has_catalog": "tiene un diccionario /Catalog válido",
    "catalog.type_is_catalog": "el /Root declara /Type /Catalog",
    "catalog.has_acroform": "tiene formulario o firma digital (/AcroForm)",
    "catalog.has_outlines": "tiene índice de marcadores (/Outlines)",
    "incremental_updates.has_incremental_updates": (
        "fue guardado más de una vez (tiene actualizaciones incrementales)"
    ),
    "metadata.has_info_dict": "tiene diccionario de metadatos (/Info)",
    "metadata.has_title": "tiene título en los metadatos",
    "metadata.has_author": "tiene autor en los metadatos",
    "metadata.has_creator": "tiene creador en los metadatos",
    "metadata.has_producer": "tiene productor en los metadatos",
    "metadata.has_creation_date": "tiene fecha de creación en los metadatos",
    "metadata.has_mod_date": "tiene fecha de modificación en los metadatos",
    "security.is_encrypted": "está cifrado",
    "trailer.has_root": "el trailer tiene una entrada /Root",
    "trailer.has_info": "el trailer tiene una entrada /Info",
    "trailer.has_id": "el trailer tiene un /ID",
    "trailer.has_encrypt": "el trailer tiene una entrada /Encrypt",
    "xref.uses_xref_stream": "usa tabla de referencias cruzadas en formato stream",
    "xref.uses_hybrid_xref": "usa xref híbrido (tabla clásica + stream)",
}

#: Base name of a `FeatureType.DICT` feature -> phrase completed by the
#: histogram key, e.g. "cantidad de objetos de tipo" + "array".
_HISTOGRAM_FEATURE_LABELS = {
    "objects.type_histogram": "cantidad de objetos de tipo",
    "streams.filter_histogram": "cantidad de streams con filtro",
    "statistics.anomaly_code_histogram": "cantidad de anomalías de tipo",
}

#: COS type names (`domain/pdf/objects.py::COS_TYPE_NAMES`) that appear as
#: `objects.type_histogram` keys. Filter names (`streams.filter_histogram`
#: keys, e.g. "FlateDecode") are left as-is — they're PDF spec identifiers,
#: not general vocabulary, and translating them would only make them
#: harder to recognize for anyone cross-referencing the spec.
_COS_TYPE_LABELS = {
    "null": "nulo",
    "boolean": "booleano",
    "number": "número",
    "name": "nombre",
    "literal_string": "cadena de texto",
    "hex_string": "cadena hexadecimal",
    "reference": "referencia",
    "array": "arreglo",
    "dictionary": "diccionario",
    "stream": "stream",
}


def humanize_feature(feature_name: str) -> str:
    if HISTOGRAM_KEY_SEPARATOR in feature_name:
        base_name, key = feature_name.split(HISTOGRAM_KEY_SEPARATOR, 1)
        template = _HISTOGRAM_FEATURE_LABELS.get(base_name)
        if template is None:
            return feature_name
        if base_name == "objects.type_histogram":
            key = _COS_TYPE_LABELS.get(key, key)
        return f"{template} {key}"
    return _BOOLEAN_FEATURE_LABELS.get(feature_name, feature_name)


def humanize_value(value: Any) -> str:
    if isinstance(value, bool):
        return "Sí" if value else "No"
    return str(value)
