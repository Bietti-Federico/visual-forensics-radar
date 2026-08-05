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


class FilterError(Exception):
    """Raised when Flate-decoding fails; callers catch this and record an anomaly."""


def flate_decode(data: bytes) -> bytes:
    try:
        return zlib.decompress(data)
    except zlib.error as exc:
        raise FilterError(str(exc)) from exc
