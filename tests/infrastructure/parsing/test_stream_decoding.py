import zlib
from decimal import Decimal

from pdf_forensics.domain.pdf.anomalies import AnomalyCode, AnomalyCollector
from pdf_forensics.domain.pdf.objects import PdfArray, PdfDictionary, PdfName, PdfNumber, PdfStream
from pdf_forensics.infrastructure.parsing.stream_decoding import decode_stream


def test_no_filter_returns_raw_bytes() -> None:
    stream = PdfStream(PdfDictionary({}), b"raw content")
    anomalies = AnomalyCollector()
    assert decode_stream(stream, anomalies) == b"raw content"
    assert not anomalies.to_list()


def test_unsupported_filter_records_anomaly_and_returns_none() -> None:
    stream = PdfStream(PdfDictionary({"Filter": PdfName("DCTDecode")}), b"\xff\xd8")
    anomalies = AnomalyCollector()
    assert decode_stream(stream, anomalies) is None
    codes = [a.code for a in anomalies.to_list()]
    assert AnomalyCode.UNSUPPORTED_FILTER in codes


def test_flate_decode_failure_records_anomaly() -> None:
    stream = PdfStream(PdfDictionary({"Filter": PdfName("FlateDecode")}), b"not compressed")
    anomalies = AnomalyCollector()
    assert decode_stream(stream, anomalies) is None
    codes = [a.code for a in anomalies.to_list()]
    assert AnomalyCode.FILTER_DECODE_FAILED in codes


def test_flate_decode_with_png_predictor_via_decode_parms_dict() -> None:
    raw_row = bytes([0, 1, 2, 3])  # filter type 0 (None), one row of 3 columns
    compressed = zlib.compress(raw_row)
    decode_parms = PdfDictionary(
        {
            "Predictor": PdfNumber(Decimal(12), True),
            "Colors": PdfNumber(Decimal(1), True),
            "BitsPerComponent": PdfNumber(Decimal(8), True),
            "Columns": PdfNumber(Decimal(3), True),
        }
    )
    stream = PdfStream(
        PdfDictionary({"Filter": PdfName("FlateDecode"), "DecodeParms": decode_parms}),
        compressed,
    )
    anomalies = AnomalyCollector()
    assert decode_stream(stream, anomalies) == bytes([1, 2, 3])


def test_decode_parms_as_array_of_one_dict() -> None:
    raw_row = bytes([0, 5, 6])  # filter type 0 (None), 2 columns
    compressed = zlib.compress(raw_row)
    decode_parms = PdfArray(
        (
            PdfDictionary(
                {"Predictor": PdfNumber(Decimal(10), True), "Columns": PdfNumber(Decimal(2), True)}
            ),
        )
    )
    stream = PdfStream(
        PdfDictionary({"Filter": PdfName("FlateDecode"), "DecodeParms": decode_parms}),
        compressed,
    )
    anomalies = AnomalyCollector()
    assert decode_stream(stream, anomalies) == bytes([5, 6])


def test_predictor_failure_records_anomaly() -> None:
    compressed = zlib.compress(b"irrelevant")
    decode_parms = PdfDictionary(
        {"Predictor": PdfNumber(Decimal(2), True), "BitsPerComponent": PdfNumber(Decimal(4), True)}
    )
    stream = PdfStream(
        PdfDictionary({"Filter": PdfName("FlateDecode"), "DecodeParms": decode_parms}),
        compressed,
    )
    anomalies = AnomalyCollector()
    assert decode_stream(stream, anomalies) is None
    codes = [a.code for a in anomalies.to_list()]
    assert AnomalyCode.FILTER_DECODE_FAILED in codes


def test_predictor_one_is_a_no_op() -> None:
    compressed = zlib.compress(b"hello")
    stream = PdfStream(PdfDictionary({"Filter": PdfName("FlateDecode")}), compressed)
    anomalies = AnomalyCollector()
    assert decode_stream(stream, anomalies) == b"hello"
