"""Shared stream-content decoding (FlateDecode + predictor) for structural streams.

Both cross-reference streams and object streams (`xref_stream_parser.py`,
`object_stream_expander.py`) need the exact same decode step — factored here so
that logic, and its anomaly reporting, exists in exactly one place.
"""

from __future__ import annotations

from pdf_forensics.domain.pdf.anomalies import AnomalyCode, AnomalyCollector, AnomalySeverity
from pdf_forensics.domain.pdf.objects import PdfArray, PdfDictionary, PdfStream
from pdf_forensics.infrastructure.parsing.filters.flate import FilterError, flate_decode
from pdf_forensics.infrastructure.parsing.filters.predictors import apply_predictor


def decode_stream(stream: PdfStream, anomalies: AnomalyCollector) -> bytes | None:
    """Return fully decoded stream bytes, or `None` if decoding isn't supported or failed."""
    filter_names = stream.filter_names
    if not filter_names:
        return stream.raw_data
    if filter_names != ("FlateDecode",):
        anomalies.record(
            AnomalyCode.UNSUPPORTED_FILTER,
            AnomalySeverity.INFO,
            f"Filter(s) {filter_names!r} are not decoded by this module; raw bytes retained.",
        )
        return None

    try:
        decoded = flate_decode(stream.raw_data)
    except FilterError as exc:
        anomalies.record(
            AnomalyCode.FILTER_DECODE_FAILED,
            AnomalySeverity.WARNING,
            f"FlateDecode failed: {exc}",
        )
        return None

    params = _decode_parms(stream.dictionary)
    predictor = params.get_int("Predictor") or 1
    if predictor <= 1:
        return decoded

    colors = params.get_int("Colors") or 1
    bits_per_component = params.get_int("BitsPerComponent") or 8
    columns = params.get_int("Columns") or 1
    try:
        return apply_predictor(
            decoded,
            predictor=predictor,
            colors=colors,
            bits_per_component=bits_per_component,
            columns=columns,
        )
    except ValueError as exc:
        anomalies.record(
            AnomalyCode.FILTER_DECODE_FAILED,
            AnomalySeverity.WARNING,
            f"Predictor decoding failed: {exc}",
        )
        return None


def _decode_parms(dictionary: PdfDictionary) -> PdfDictionary:
    value = dictionary.get("DecodeParms") or dictionary.get("DP")
    if isinstance(value, PdfDictionary):
        return value
    if isinstance(value, PdfArray):
        for item in value.items:
            if isinstance(item, PdfDictionary):
                return item
    return PdfDictionary({})
