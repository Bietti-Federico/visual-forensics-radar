"""FlateDecode (ISO 32000-1 §7.4.4) — the only compression filter this module implements.

Cross-reference streams and object streams (both introduced in PDF 1.5 to store
structural data compactly) are, in every generator observed, Flate-compressed —
decoding this one filter is sufficient to parse *structure* even though this
module does not decode filters used purely for image/font content (DCTDecode,
CCITTFaxDecode, LZWDecode, ...). Those are recorded via
`AnomalyCode.UNSUPPORTED_FILTER` by the caller and left as raw bytes for a later
module to handle.
"""

from __future__ import annotations

import zlib

# Caps decompression-bomb streams (a tiny compressed payload expanding to
# gigabytes) from exhausting memory. Chosen well above any legitimate PDF
# structural stream (xref/object streams are metadata, not media) while
# still bounding worst-case memory use per stream.
_MAX_DECOMPRESSED_BYTES = 100 * 1024 * 1024


class FilterError(Exception):
    """Raised when Flate-decoding fails; callers catch this and record an anomaly."""


def flate_decode(data: bytes) -> bytes:
    decompressor = zlib.decompressobj()
    try:
        result = decompressor.decompress(data, _MAX_DECOMPRESSED_BYTES)
        if decompressor.unconsumed_tail:
            raise FilterError(f"decompressed output exceeds the {_MAX_DECOMPRESSED_BYTES}-byte cap")
        result += decompressor.flush()
    except zlib.error as exc:
        raise FilterError(str(exc)) from exc
    return result
