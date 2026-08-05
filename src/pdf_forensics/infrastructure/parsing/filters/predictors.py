"""PNG and TIFF predictors (ISO 32000-1 §7.4.4.4), applied after FlateDecode.

Both xref streams and object streams are near-universally written with
`/Predictor 12` (PNG "Up" as the *common* choice, though the actual per-row
algorithm is read from the row itself, not the `/Predictor` value — see
`_undo_png_predictor`). TIFF predictor 2 support is limited to 8-bit components,
the only depth structural streams use in practice; anything else raises
`ValueError`, which callers catch and record as `AnomalyCode.UNSUPPORTED_FILTER`
rather than silently producing corrupted bytes.
"""

from __future__ import annotations


def apply_predictor(
    data: bytes,
    *,
    predictor: int,
    colors: int = 1,
    bits_per_component: int = 8,
    columns: int = 1,
) -> bytes:
    if predictor <= 1:
        return data
    if predictor == 2:
        return _undo_tiff_predictor(data, colors, bits_per_component, columns)
    if predictor >= 10:
        return _undo_png_predictor(data, colors, bits_per_component, columns)
    raise ValueError(f"Unsupported /Predictor value: {predictor}")


def _undo_tiff_predictor(data: bytes, colors: int, bits_per_component: int, columns: int) -> bytes:
    if bits_per_component != 8:
        raise ValueError("TIFF predictor is only supported for 8-bit components")
    bytes_per_pixel = max(1, colors)
    row_bytes = max(1, colors * columns)
    out = bytearray()
    for row_start in range(0, len(data) - len(data) % row_bytes, row_bytes):
        row = bytearray(data[row_start : row_start + row_bytes])
        for i in range(bytes_per_pixel, len(row)):
            row[i] = (row[i] + row[i - bytes_per_pixel]) & 0xFF
        out.extend(row)
    return bytes(out)


def _undo_png_predictor(data: bytes, colors: int, bits_per_component: int, columns: int) -> bytes:
    bytes_per_pixel = max(1, (colors * bits_per_component + 7) // 8)
    row_bytes = max(1, (colors * bits_per_component * columns + 7) // 8)
    stride = row_bytes + 1  # +1 for the per-row filter-type tag byte
    out = bytearray()
    prev_row = bytearray(row_bytes)
    pos = 0
    while pos + stride <= len(data):
        filter_type = data[pos]
        row = bytearray(data[pos + 1 : pos + stride])
        for i in range(len(row)):
            left = row[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
            up = prev_row[i]
            up_left = prev_row[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
            row[i] = (row[i] + _predict(filter_type, left, up, up_left)) & 0xFF
        out.extend(row)
        prev_row = row
        pos += stride
    return bytes(out)


def _predict(filter_type: int, left: int, up: int, up_left: int) -> int:
    if filter_type == 0:
        return 0
    if filter_type == 1:
        return left
    if filter_type == 2:
        return up
    if filter_type == 3:
        return (left + up) // 2
    if filter_type == 4:
        return _paeth(left, up, up_left)
    return 0  # unrecognized per-row filter tag: pass the byte through unmodified


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c
