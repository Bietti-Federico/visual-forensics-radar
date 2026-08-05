import datetime
import gc
import hashlib
import logging
import re

import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PDFMetadataExtractor")

PDF_VERSION_PATTERN = re.compile(rb"%PDF-(\d\.\d)")

# PDF date strings look like "D:20260722173545+00'00'" (ISO-ish but not ISO 8601) —
# kept as a named-group regex instead of strptime since the timezone suffix uses a
# quote-delimited "HH'mm'" form strptime doesn't parse directly, and the offset/
# apostrophes are themselves worth surfacing raw for a human comparing across banks.
PDF_DATE_PATTERN = re.compile(
    r"^D:(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})"
    r"(?P<hour>\d{2})?(?P<minute>\d{2})?(?P<second>\d{2})?"
    r"(?P<tz>[Z+\-].*)?$"
)


class PDFMetadataExtractor:
    """
    Investigative extractor for a branch where the ingestion flow now only accepts
    PDFs downloaded directly from a bank's own site (no photos, no screenshots of a
    phone screen). The question this answers isn't "what does the receipt say"
    (that's the OCR pipeline on the image-based branch) but "what does the PDF FILE
    ITSELF carry and reveal" — producer/creator strings, whether it has a real text
    layer or is a flattened scan, digital signature presence, embedded files, and
    incremental-update history (a PDF edited after the bank generated it leaves
    visible traces of that in its own byte structure). Nothing here is a fraud
    verdict — it's raw signal for a human to review, to decide what's controllable
    and what isn't, per document type and per bank.
    """

    def analyze(self, pdf_path: str) -> dict:
        try:
            with open(pdf_path, "rb") as f:
                raw_bytes = f.read()

            doc = fitz.open(pdf_path)
            try:
                result = {
                    "status": "success",
                    "file": self._file_info(raw_bytes),
                    "standard_metadata": self._standard_metadata(doc),
                    "structure": self._structure_info(doc, raw_bytes),
                    "text_layer": self._text_layer_info(doc),
                    "signatures": self._signature_info(doc),
                    "embedded_files": self._embedded_files_info(doc),
                    "xmp_metadata": self._xmp_metadata(doc),
                }
            finally:
                doc.close()
                del doc
                # Page/Widget wrapper objects hold native MuPDF pointers back into the
                # document and can form reference cycles CPython's refcounting alone
                # won't free — left to the cyclic collector, their finalizers can run
                # an arbitrary amount of time later (a background Streamlit rerun, an
                # unrelated later request), crashing the process nowhere near the
                # request that actually created them. Forcing collection here, right
                # after close() while nothing else is going on, makes that cleanup
                # happen NOW instead of as a delayed, hard-to-place native segfault.
                gc.collect()

            return result

        except Exception as e:
            logger.error(f"Error analyzing PDF: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _file_info(self, raw_bytes: bytes) -> dict:
        version_match = PDF_VERSION_PATTERN.search(raw_bytes[:2048])
        return {
            "size_bytes": len(raw_bytes),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "pdf_version": version_match.group(1).decode() if version_match else None,
        }

    def _parse_pdf_date(self, raw_value: str | None) -> dict | None:
        if not raw_value:
            return None
        match = PDF_DATE_PATTERN.match(raw_value)
        if not match:
            return {"raw": raw_value, "parsed": False}
        groups = match.groupdict()
        return {
            "raw": raw_value,
            "parsed": True,
            "year": int(groups["year"]),
            "month": int(groups["month"]),
            "day": int(groups["day"]),
            "hour": int(groups["hour"]) if groups["hour"] else None,
            "minute": int(groups["minute"]) if groups["minute"] else None,
            "second": int(groups["second"]) if groups["second"] else None,
            "timezone": groups["tz"],
        }

    def _parse_tz_offset(self, tz_raw: str | None) -> datetime.timedelta | None:
        if not tz_raw:
            return None
        if tz_raw == "Z":
            return datetime.timedelta(0)
        match = re.match(r"^([+\-])(\d{2})'?(\d{2})'?$", tz_raw)
        if not match:
            return None
        sign, hours, minutes = match.groups()
        delta = datetime.timedelta(hours=int(hours), minutes=int(minutes))
        return -delta if sign == "-" else delta

    def _pdf_date_to_utc(self, parsed: dict | None) -> datetime.datetime | None:
        if not parsed or not parsed.get("parsed") or parsed.get("hour") is None:
            return None
        offset = self._parse_tz_offset(parsed.get("timezone"))
        if offset is None:
            return None
        try:
            naive = datetime.datetime(
                parsed["year"], parsed["month"], parsed["day"],
                parsed.get("hour") or 0, parsed.get("minute") or 0, parsed.get("second") or 0,
            )
        except ValueError:
            return None
        return naive.replace(tzinfo=datetime.timezone(offset))

    def _standard_metadata(self, doc: "fitz.Document") -> dict:
        meta = doc.metadata or {}
        creation_date = self._parse_pdf_date(meta.get("creationDate"))
        mod_date = self._parse_pdf_date(meta.get("modDate"))

        # Creation and modification are commonly stamped in DIFFERENT timezone
        # notations (e.g. local -03'00' vs a signing step that stamps Z/UTC) — comparing
        # the raw hour:minute:second fields directly, as a human eyeballing the two raw
        # strings would, can make a same-instant save look like a multi-hour gap or
        # vice versa. This normalizes both to UTC first, so the gap reported is real.
        creation_dt = self._pdf_date_to_utc(creation_date)
        mod_dt = self._pdf_date_to_utc(mod_date)
        gap_seconds = (mod_dt - creation_dt).total_seconds() if creation_dt and mod_dt else None

        return {
            "title": meta.get("title") or None,
            "author": meta.get("author") or None,
            "subject": meta.get("subject") or None,
            "keywords": meta.get("keywords") or None,
            "creator": meta.get("creator") or None,
            "producer": meta.get("producer") or None,
            "creation_date": creation_date,
            "mod_date": mod_date,
            "creation_to_modification_gap_seconds": gap_seconds,
            "trapped": meta.get("trapped") or None,
        }

    def _structure_info(self, doc: "fitz.Document", raw_bytes: bytes) -> dict:
        # Every incremental save (edit-and-resave, e.g. adding a signature or annotating
        # in a PDF editor) appends its own xref table + trailer + "%%EOF" to the file
        # rather than rewriting it from scratch — the ORIGINAL content stays byte-for-
        # byte in place, followed by one more of these blocks per save. A freshly
        # generated PDF normally has exactly ONE "%%EOF" — but every reference sample
        # checked so far (signed ANSES receipt, two unsigned municipal payroll stubs,
        # different producers) came in with exactly TWO: one for the original render,
        # one appended by whatever step finalizes/signs it. That second save is the
        # NORMAL baseline for these sources, not a red flag — the threshold below is
        # set to flag a THIRD (or more) save, which none of the known-good samples have.
        eof_count = raw_bytes.count(b"%%EOF")
        startxref_count = raw_bytes.count(b"startxref")

        return {
            "page_count": doc.page_count,
            "is_encrypted": doc.is_encrypted,
            "needs_password": doc.needs_pass,
            "permissions": doc.permissions if doc.is_encrypted else None,
            "xref_entry_count": doc.xref_length(),
            "eof_marker_count": eof_count,
            "startxref_count": startxref_count,
            "likely_incrementally_updated": eof_count > 2,
        }

    def _text_layer_info(self, doc: "fitz.Document") -> dict:
        pages_detail = []
        total_chars = 0
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            text = page.get_text("text") or ""
            image_count = len(page.get_images(full=True))
            char_count = len(text.strip())
            total_chars += char_count
            pages_detail.append({
                "page": page_index + 1,
                "char_count": char_count,
                "image_count": image_count,
                # A page with images but (near) no extractable text is very likely a
                # flattened scan/photo embedded as an image rather than a native,
                # searchable PDF page — the opposite of what a bank's own PDF export
                # should look like.
                "looks_like_scanned_image": char_count < 10 and image_count > 0,
            })

        return {
            "has_text_layer": total_chars >= 10,
            "total_char_count": total_chars,
            "pages": pages_detail,
        }

    def _signature_info(self, doc: "fitz.Document") -> dict:
        # get_sigflags(): -1 = no signature fields at all in the AcroForm; otherwise a
        # bitfield (bit 1 = SignaturesExist, bit 2 = AppendOnly once signed).
        sigflags = doc.get_sigflags()
        fields = []
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            for widget in page.widgets() or []:
                if widget.field_type_string != "Signature":
                    continue
                fields.append({
                    "page": page_index + 1,
                    "field_name": widget.field_name,
                    # is_signed only reflects whether the field has been filled with a
                    # signature dictionary — this is NOT cryptographic verification of
                    # the certificate chain or the signed content's integrity.
                    "is_signed": bool(getattr(widget, "is_signed", False)),
                })

        return {
            "sigflags": sigflags,
            "has_signature_fields": sigflags != -1,
            "fields": fields,
            "note": "Presence/fill-state only — no certificate chain or content integrity verification performed.",
        }

    def _embedded_files_info(self, doc: "fitz.Document") -> list[dict]:
        embedded = []
        for name in doc.embfile_names():
            info = doc.embfile_info(name)
            embedded.append({
                "name": name,
                "filename": info.get("filename"),
                "description": info.get("desc") or None,
                "size_bytes": info.get("size"),
            })
        return embedded

    def _xmp_metadata(self, doc: "fitz.Document") -> dict:
        raw_xml = doc.get_xml_metadata() or ""
        return {
            "present": bool(raw_xml.strip()),
            # Left as raw XML rather than parsed RDF — this branch is for seeing what
            # each bank's export actually carries before deciding what's worth modeling
            # properly, and XMP schemas vary widely (some carry a full edit-history
            # xmpMM:History sequence with per-revision tool/agent/timestamp).
            "raw_xml": raw_xml[:8000] if raw_xml else None,
            "truncated": len(raw_xml) > 8000,
        }
