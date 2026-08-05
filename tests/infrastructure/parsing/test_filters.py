import zlib

import pytest

from pdf_forensics.infrastructure.parsing.filters.flate import FilterError, flate_decode
from pdf_forensics.infrastructure.parsing.filters.predictors import apply_predictor


def test_flate_decode_round_trip() -> None:
    original = b"the quick brown fox jumps over the lazy dog" * 10
    assert flate_decode(zlib.compress(original)) == original


def test_flate_decode_raises_filter_error_on_garbage() -> None:
    with pytest.raises(FilterError):
        flate_decode(b"not compressed data")


def test_predictor_none_passthrough() -> None:
    data = b"abcdef"
    assert apply_predictor(data, predictor=1) == data


def test_png_predictor_sub_reconstructs_row() -> None:
    # 1 color, 8 bpc, 3 columns: row = [10, 5, 5] with filter type 1 (Sub).
    # Sub-encoded: byte[i] = raw[i] - raw[i-1] (raw[-1] == 0).
    encoded_row = bytes([1, 10, (5 - 10) & 0xFF, (5 - 5) & 0xFF])
    decoded = apply_predictor(encoded_row, predictor=10, colors=1, bits_per_component=8, columns=3)
    assert decoded == bytes([10, 5, 5])


def test_png_predictor_up_uses_previous_row() -> None:
    row0 = bytes([0, 1, 2, 3])  # filter type 0 (None): raw row [1, 2, 3]
    row1 = bytes([2, 1, 1, 1])  # filter type 2 (Up): raw[i] = enc[i] + prevRaw[i]
    decoded = apply_predictor(row0 + row1, predictor=10, colors=1, bits_per_component=8, columns=3)
    assert decoded == bytes([1, 2, 3, 2, 3, 4])


def test_png_predictor_paeth() -> None:
    # Single row [10, 20, 30], filter type 4 (Paeth), hand-derived per ISO 32000-1
    # Annex C: each byte's predictor is the raw value of its left neighbor here
    # (up/up-left are 0 on the first row), so encoded[i] = raw[i] - raw[i - 1].
    encoded_row = bytes([4, 10, 10, 10])
    decoded = apply_predictor(encoded_row, predictor=10, colors=1, bits_per_component=8, columns=3)
    assert decoded == bytes([10, 20, 30])


def test_png_predictor_average() -> None:
    # Row [10, 20, 30], filter type 3 (Average): encoded[i] = raw[i] - (left+up)//2.
    encoded_row = bytes([3, 10, 15, 20])
    decoded = apply_predictor(encoded_row, predictor=10, colors=1, bits_per_component=8, columns=3)
    assert decoded == bytes([10, 20, 30])


def test_png_predictor_unrecognized_filter_tag_passes_through() -> None:
    encoded_row = bytes([7, 55, 66, 77])
    decoded = apply_predictor(encoded_row, predictor=10, colors=1, bits_per_component=8, columns=3)
    assert decoded == bytes([55, 66, 77])


def test_png_predictor_paeth_returns_up_or_upper_left_when_closer() -> None:
    # Row 0 (filter None): raw [2, 1]. Row 1 (filter Paeth): hand-picked so the
    # second byte's predictor neighbors are (left=3, up=1, upper-left=2), a case
    # where Paeth picks upper-left (2) over both left and up.
    row0 = bytes([0, 2, 1])
    row1 = bytes([4, 1, 5])
    decoded = apply_predictor(row0 + row1, predictor=10, colors=1, bits_per_component=8, columns=2)
    assert decoded == bytes([2, 1, 3, 7])


def test_tiff_predictor_reconstructs_row() -> None:
    # 1 color, 8 bpc, 3 columns, horizontal differencing: encoded[i] = raw[i] - raw[i-1].
    encoded = bytes([10, (5 - 10) & 0xFF, (5 - 5) & 0xFF])
    decoded = apply_predictor(encoded, predictor=2, colors=1, bits_per_component=8, columns=3)
    assert decoded == bytes([10, 5, 5])


def test_tiff_predictor_rejects_non_8_bit() -> None:
    with pytest.raises(ValueError, match="8-bit"):
        apply_predictor(b"\x00", predictor=2, bits_per_component=4)


def test_unsupported_predictor_value_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        apply_predictor(b"\x00", predictor=3)
