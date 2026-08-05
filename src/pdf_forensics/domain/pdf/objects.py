"""The COS (Carousel Object Structure) type system — pure data, no parsing logic.

Every PDF value, however it was produced (a hand-crafted incremental update, a
compliant generator, a corrupted transfer), decomposes into exactly these eight
primitive/compound types (ISO 32000-1 §7.3). Keeping this module free of parsing
logic means the same types serve raw hand-built fixtures in tests, the tokenizer's
output, and eventually a re-serializer, without coupling any of them to how bytes
were turned into values. Types are frozen: a `PdfObject` is a claim about what one
revision of the file contained at a point in time, and forensic reasoning depends
on that claim never mutating out from under a caller holding a reference to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class PdfNull:
    """The PDF `null` keyword."""


@dataclass(frozen=True, slots=True)
class PdfBoolean:
    """The PDF `true`/`false` keywords."""

    value: bool


@dataclass(frozen=True, slots=True)
class PdfNumber:
    """A PDF numeric object.

    `Decimal` (not `float`) preserves the exact digits a generator wrote — useful
    forensic signal when comparing how different tools format the same value
    (e.g. trailing zeros, exponent-free notation mandated by the spec).
    """

    value: Decimal
    is_integer: bool

    def as_int(self) -> int:
        return int(self.value)


@dataclass(frozen=True, slots=True)
class PdfName:
    """A PDF name object (`/Foo`), with `#xx` escapes already decoded to text."""

    value: str


@dataclass(frozen=True, slots=True)
class PdfLiteralString:
    """A PDF literal string `(...)`, with escapes already decoded to raw bytes."""

    value: bytes

    def decode_text(self) -> str:
        return _decode_pdf_text_bytes(self.value)


@dataclass(frozen=True, slots=True)
class PdfHexString:
    """A PDF hex string `<...>`, with whitespace stripped and hex-decoded to bytes."""

    value: bytes

    def decode_text(self) -> str:
        return _decode_pdf_text_bytes(self.value)


def _decode_pdf_text_bytes(raw: bytes) -> str:
    """Best-effort text decode per ISO 32000-1 §7.9.2.2: UTF-16BE with a BOM, else PDFDocEncoding.

    PDFDocEncoding is a single-byte encoding close enough to Latin-1 for the
    printable ASCII range that generators care about here; a full PDFDocEncoding
    table is out of scope for this module and not needed for structural forensics.
    """
    if raw.startswith(b"\xfe\xff"):
        try:
            return raw[2:].decode("utf-16-be")
        except UnicodeDecodeError:
            return raw.decode("latin-1")
    return raw.decode("latin-1")


@dataclass(frozen=True, slots=True)
class PdfReference:
    """An indirect reference `N G R` to another object, not the object itself."""

    obj_num: int
    generation: int


@dataclass(frozen=True, slots=True)
class PdfArray:
    """A PDF array `[...]`."""

    items: tuple[PdfValue, ...]


@dataclass(frozen=True, slots=True)
class PdfDictionary:
    """A PDF dictionary `<< ... >>`.

    Accessor helpers return `None` on a missing key or a type mismatch rather than
    raising — a dictionary with the wrong value type for a well-known key (e.g.
    `/Length` given as a name instead of a number) is exactly the kind of anomaly
    this platform exists to surface, not a reason to crash the caller.
    """

    entries: dict[str, PdfValue]

    def get(self, key: str) -> PdfValue | None:
        return self.entries.get(key)

    def get_name(self, key: str) -> str | None:
        value = self.entries.get(key)
        return value.value if isinstance(value, PdfName) else None

    def get_int(self, key: str) -> int | None:
        value = self.entries.get(key)
        return value.as_int() if isinstance(value, PdfNumber) else None

    def get_ref(self, key: str) -> PdfReference | None:
        value = self.entries.get(key)
        return value if isinstance(value, PdfReference) else None

    def get_array(self, key: str) -> PdfArray | None:
        value = self.entries.get(key)
        return value if isinstance(value, PdfArray) else None

    def get_dict(self, key: str) -> PdfDictionary | None:
        value = self.entries.get(key)
        return value if isinstance(value, PdfDictionary) else None


@dataclass(frozen=True, slots=True)
class PdfStream:
    """A PDF stream object: its dictionary plus the raw (still-encoded) bytes.

    Decoding filters is an infrastructure concern (see
    `pdf_forensics.infrastructure.parsing.filters`) — the domain type only records
    what bytes were present between `stream`/`endstream`, and separately, via
    `filter_names`, which filters the dictionary *claims* apply, even when this
    module doesn't implement decoding them.
    """

    dictionary: PdfDictionary
    raw_data: bytes

    @property
    def filter_names(self) -> tuple[str, ...]:
        filter_value = self.dictionary.get("Filter")
        if isinstance(filter_value, PdfName):
            return (filter_value.value,)
        if isinstance(filter_value, PdfArray):
            return tuple(item.value for item in filter_value.items if isinstance(item, PdfName))
        return ()


PdfValue = (
    PdfNull
    | PdfBoolean
    | PdfNumber
    | PdfName
    | PdfLiteralString
    | PdfHexString
    | PdfReference
    | PdfArray
    | PdfDictionary
    | PdfStream
)


@dataclass(frozen=True, slots=True)
class PdfObject:
    """One indirect object definition: `N G obj ... endobj`, as found in one revision."""

    obj_num: int
    generation: int
    value: PdfValue
    offset: int | None
