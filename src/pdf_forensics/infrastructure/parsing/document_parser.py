"""Top-level orchestration: raw PDF bytes in, a `PdfDocument` out.

Resolution order per ISO 32000-1 §7.5.5/§7.5.8, with a forensic-grade fallback the
spec doesn't require: locate the header, follow `startxref` through the `/Prev`
chain of revisions (classic tables and/or xref streams), merge every revision's
entries (most recent wins), resolve every object, expand object streams, then
detect encryption and cross-revision duplicate object ids. If the xref chain
yields nothing usable, fall back to a brute-force `N G obj` byte scan rather than
giving up — a forensics tool must still say what it can about a file whose xref
is itself the tampering.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from pdf_forensics.domain.pdf.anomalies import AnomalyCode, AnomalyCollector, AnomalySeverity
from pdf_forensics.domain.pdf.document import PdfDocument, Revision, XrefEntry, XrefEntryType
from pdf_forensics.domain.pdf.errors import NotAPdfError, UnrecoverableStructureError
from pdf_forensics.domain.pdf.objects import (
    PdfDictionary,
    PdfNumber,
    PdfObject,
    PdfReference,
    PdfValue,
)
from pdf_forensics.infrastructure.parsing.object_parser import LengthResolver, ObjectParser
from pdf_forensics.infrastructure.parsing.object_stream_expander import expand_object_stream
from pdf_forensics.infrastructure.parsing.tokenizer import PdfTokenizer, Token, TokenKind
from pdf_forensics.infrastructure.parsing.trailer_resolver import walk_revision_chain
from pdf_forensics.infrastructure.parsing.xref_stream_parser import parse_xref_stream
from pdf_forensics.infrastructure.parsing.xref_table_parser import parse_xref_table

_BRUTE_FORCE_OBJECT_PATTERN = re.compile(rb"(?<![0-9])(\d+)[ \t]+(\d+)[ \t]+obj\b")
_PDF_VERSION_PATTERN = re.compile(rb"%PDF-(\d\.\d)")


class PdfDocumentParser:
    """Parses raw PDF bytes into a `PdfDocument`. Stateless; safe to reuse across calls."""

    def parse(self, data: bytes) -> PdfDocument:
        header_index = data.find(b"%PDF-")
        if header_index == -1:
            raise NotAPdfError("No %PDF- header found in file.")
        pdf_version = self._detect_pdf_version(data, header_index)

        anomalies = AnomalyCollector()
        tokenizer = PdfTokenizer(data)

        revisions = self._walk_from_startxref(data, tokenizer, anomalies)
        effective_entries = self._merge_entries(revisions)

        if not effective_entries:
            revisions = self._brute_force_reconstruct(data, tokenizer, anomalies)
            effective_entries = self._merge_entries(revisions)

        if not effective_entries:
            raise UnrecoverableStructureError(
                "No object could be located via xref, xref stream, or a brute-force byte scan."
            )

        object_parser = ObjectParser(
            tokenizer,
            anomalies,
            length_resolver=self._make_length_resolver(tokenizer, effective_entries),
        )
        objects = self._resolve_direct_objects(data, object_parser, effective_entries, anomalies)
        self._expand_compressed_objects(objects, effective_entries, anomalies)

        document = PdfDocument(
            revisions=revisions,
            objects=objects,
            anomalies=anomalies.to_list(),
            raw_size=len(data),
            pdf_version=pdf_version,
        )
        self._detect_encryption(document)
        return document

    def _detect_pdf_version(self, data: bytes, header_index: int) -> str | None:
        match = _PDF_VERSION_PATTERN.match(data, header_index)
        return match.group(1).decode("ascii") if match else None

    # -- Revision discovery -------------------------------------------------

    def _walk_from_startxref(
        self, data: bytes, tokenizer: PdfTokenizer, anomalies: AnomalyCollector
    ) -> list[Revision]:
        startxref_offset = self._locate_startxref(data, tokenizer)
        if startxref_offset is None:
            return []

        def parse_revision_at(offset: int) -> Revision | None:
            return self._parse_one_revision(tokenizer, offset, anomalies)

        return walk_revision_chain(startxref_offset, parse_revision_at, anomalies)

    def _locate_startxref(self, data: bytes, tokenizer: PdfTokenizer) -> int | None:
        keyword_offset = data.rfind(b"startxref")
        if keyword_offset == -1:
            return None
        keyword_tok = tokenizer.next_token(keyword_offset)
        offset_tok = tokenizer.next_token(keyword_tok.end)
        return _read_int(offset_tok)

    def _parse_one_revision(
        self, tokenizer: PdfTokenizer, offset: int, anomalies: AnomalyCollector
    ) -> Revision | None:
        scratch_parser = ObjectParser(tokenizer, anomalies)

        table_revision = parse_xref_table(tokenizer, scratch_parser, offset, anomalies)
        if table_revision is not None:
            self._merge_hybrid_xref_stream(scratch_parser, table_revision, anomalies)
            return table_revision

        obj, _ = scratch_parser.parse_indirect_object(offset)
        if obj is None:
            return None
        return parse_xref_stream(obj, offset, anomalies)

    def _merge_hybrid_xref_stream(
        self,
        object_parser: ObjectParser,
        table_revision: Revision,
        anomalies: AnomalyCollector,
    ) -> None:
        """Fill in compressed-object entries from `/XRefStm` (hybrid-reference files, §7.5.8.4)."""
        hybrid_offset = table_revision.trailer.get_int("XRefStm")
        if hybrid_offset is None:
            return
        obj, _ = object_parser.parse_indirect_object(hybrid_offset)
        if obj is None:
            return
        hybrid_revision = parse_xref_stream(obj, hybrid_offset, anomalies)
        if hybrid_revision is None:
            return
        for obj_num, entry in hybrid_revision.xref_entries.items():
            table_revision.xref_entries.setdefault(obj_num, entry)

    # -- Brute-force fallback -------------------------------------------------

    def _brute_force_reconstruct(
        self, data: bytes, tokenizer: PdfTokenizer, anomalies: AnomalyCollector
    ) -> list[Revision]:
        matches = list(_BRUTE_FORCE_OBJECT_PATTERN.finditer(data))
        if not matches:
            return []

        anomalies.record(
            AnomalyCode.XREF_UNPARSEABLE_FALLBACK_USED,
            AnomalySeverity.CRITICAL,
            "No usable xref table or xref stream was found; the object table was "
            "reconstructed by scanning the file for 'N G obj' patterns.",
        )

        entries: dict[int, XrefEntry] = {}
        for match in matches:
            obj_num, generation = int(match.group(1)), int(match.group(2))
            if obj_num in entries:
                anomalies.record(
                    AnomalyCode.DUPLICATE_OBJECT_ID,
                    AnomalySeverity.WARNING,
                    f"Object {obj_num} is defined more than once; the later definition wins.",
                    byte_offset=match.start(),
                )
            entries[obj_num] = XrefEntry(
                obj_num=obj_num,
                generation=generation,
                entry_type=XrefEntryType.IN_USE,
                offset_or_stream_obj_num=match.start(),
            )

        trailer = self._recover_trailer(data, tokenizer, anomalies)
        return [
            Revision(trailer=trailer, xref_entries=entries, xref_offset=0, is_xref_stream=False)
        ]

    def _recover_trailer(
        self, data: bytes, tokenizer: PdfTokenizer, anomalies: AnomalyCollector
    ) -> PdfDictionary:
        trailer_offset = data.rfind(b"trailer")
        if trailer_offset == -1:
            anomalies.record(
                AnomalyCode.TRAILER_MISSING_ROOT,
                AnomalySeverity.WARNING,
                "No /Root key found in any revision's trailer.",
            )
            return PdfDictionary({})
        object_parser = ObjectParser(tokenizer, anomalies)
        keyword_tok = tokenizer.next_token(trailer_offset)
        value, _ = object_parser.parse_value(keyword_tok.end)
        trailer = value if isinstance(value, PdfDictionary) else PdfDictionary({})
        if trailer.get_ref("Root") is None:
            anomalies.record(
                AnomalyCode.TRAILER_MISSING_ROOT,
                AnomalySeverity.WARNING,
                "No /Root key found in any revision's trailer.",
            )
        return trailer

    # -- Object resolution -------------------------------------------------

    @staticmethod
    def _merge_entries(revisions: list[Revision]) -> dict[int, XrefEntry]:
        merged: dict[int, XrefEntry] = {}
        for revision in reversed(revisions):
            merged.update(revision.xref_entries)
        return merged

    def _make_length_resolver(
        self,
        tokenizer: PdfTokenizer,
        effective_entries: dict[int, XrefEntry],
    ) -> LengthResolver:
        def resolve(ref: PdfReference) -> int | None:
            entry = effective_entries.get(ref.obj_num)
            if entry is None or entry.entry_type != XrefEntryType.IN_USE:
                return None
            scratch = AnomalyCollector()
            parser = ObjectParser(tokenizer, scratch)
            obj, _ = parser.parse_indirect_object(entry.offset_or_stream_obj_num)
            if obj is not None and isinstance(obj.value, PdfNumber):
                return obj.value.as_int()
            return None

        return resolve

    def _resolve_direct_objects(
        self,
        data: bytes,
        object_parser: ObjectParser,
        effective_entries: dict[int, XrefEntry],
        anomalies: AnomalyCollector,
    ) -> dict[tuple[int, int], PdfObject]:
        objects: dict[tuple[int, int], PdfObject] = {}
        for entry in effective_entries.values():
            if entry.entry_type != XrefEntryType.IN_USE:
                continue
            obj = self._resolve_in_use_entry(data, object_parser, entry, anomalies)
            if obj is not None:
                objects[(obj.obj_num, obj.generation)] = obj
        return objects

    def _resolve_in_use_entry(
        self,
        data: bytes,
        object_parser: ObjectParser,
        entry: XrefEntry,
        anomalies: AnomalyCollector,
    ) -> PdfObject | None:
        obj, _ = object_parser.parse_indirect_object(entry.offset_or_stream_obj_num)
        if obj is not None and obj.obj_num == entry.obj_num and obj.generation == entry.generation:
            return obj

        anomalies.record(
            AnomalyCode.XREF_OFFSET_MISMATCH,
            AnomalySeverity.WARNING,
            f"xref offset {entry.offset_or_stream_obj_num} for object "
            f"{entry.obj_num} {entry.generation} did not point at a matching 'obj' "
            "definition; recovered by scanning the file.",
            object_ref=PdfReference(entry.obj_num, entry.generation),
            byte_offset=entry.offset_or_stream_obj_num,
        )
        pattern = re.compile(
            rf"(?<![0-9]){entry.obj_num}[ \t]+{entry.generation}[ \t]+obj\b".encode("ascii")
        )
        match = pattern.search(data)
        if match is None:
            return None
        recovered, _ = object_parser.parse_indirect_object(match.start())
        return recovered

    def _expand_compressed_objects(
        self,
        objects: dict[tuple[int, int], PdfObject],
        effective_entries: dict[int, XrefEntry],
        anomalies: AnomalyCollector,
    ) -> None:
        processed_containers: set[int] = set()
        for entry in effective_entries.values():
            if entry.entry_type != XrefEntryType.COMPRESSED:
                continue
            container_num = entry.offset_or_stream_obj_num
            if container_num in processed_containers:
                continue
            processed_containers.add(container_num)

            container_obj = objects.get((container_num, 0))
            if container_obj is None:
                anomalies.record(
                    AnomalyCode.OBJSTM_INDEX_OUT_OF_RANGE,
                    AnomalySeverity.WARNING,
                    f"Object stream container {container_num} could not be resolved.",
                )
                continue
            for expanded in expand_object_stream(container_obj, anomalies):
                objects[(expanded.obj_num, expanded.generation)] = expanded

    # -- Encryption flagging -------------------------------------------------

    def _detect_encryption(self, document: PdfDocument) -> None:
        encrypt_value: PdfValue | None = document.trailer.get("Encrypt")
        if isinstance(encrypt_value, PdfReference):
            document.is_encrypted = True
            resolved = document.resolve(encrypt_value)
            if resolved is not None and isinstance(resolved.value, PdfDictionary):
                document.encryption_dict = resolved.value
        elif isinstance(encrypt_value, PdfDictionary):
            document.is_encrypted = True
            document.encryption_dict = encrypt_value


def _read_int(token: Token) -> int | None:
    if token.kind != TokenKind.NUMBER or not isinstance(token.value, bytes):
        return None
    try:
        return int(Decimal(token.value.decode("ascii", errors="replace")))
    except (InvalidOperation, ValueError):
        return None
