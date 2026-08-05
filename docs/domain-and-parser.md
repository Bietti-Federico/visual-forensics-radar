# Domain layer + low-level PDF parser

This document covers `src/pdf_forensics/domain/pdf/` and
`src/pdf_forensics/infrastructure/parsing/` — the foundation module of
pdf-forensics-platform: the data model for a parsed PDF, and a from-scratch
byte-level parser that produces it.

## Why not PyMuPDF / pikepdf

Every mainstream PDF library optimizes for *rendering* a document, which means
silently repairing or hiding exactly the deviations this platform exists to
find: a broken xref offset, an object redefined outside the normal incremental-
update mechanism, a stream whose declared `/Length` doesn't match its real
extent. A parser built for forensics has to see those deviations, not route
around them. That requires owning every byte-level decision — hence a
dependency-free tokenizer and object parser instead of a wrapper around an
existing library.

## Anomalies are data, not exceptions

The central design decision of this module: malformed PDF structure is
*expected* input, not a bug to crash on. `StructuralAnomaly`
(`domain/pdf/anomalies.py`) is a first-class value recorded by an
`AnomalyCollector` threaded through every parsing component. Two exceptions
remain (`domain/pdf/errors.py`), reserved for input with genuinely no evidence
to extract:

- `NotAPdfError` — no `%PDF-` header anywhere in the file.
- `UnrecoverableStructureError` — no object locatable by xref, xref stream, or
  a brute-force `N G obj` byte scan.

Everything else short of that degrades to an anomaly and a best-effort
reconstruction, because a forensics tool that aborts on the first malformed
byte is useless against exactly the files it needs to examine.

## COS model → spec sections

| Type | ISO 32000-1 |
|---|---|
| `PdfNull`, `PdfBoolean`, `PdfNumber`, `PdfName` | §7.3.2–7.3.5 |
| `PdfLiteralString`, `PdfHexString` | §7.3.4 |
| `PdfArray`, `PdfDictionary` | §7.3.6, §7.3.7 |
| `PdfStream` | §7.3.8 |
| `PdfReference` | §7.3.10 |
| `Revision` (xref + trailer) | §7.5.4–7.5.6 |
| Cross-reference streams | §7.5.8 |
| Object streams | §7.5.7 |

## Xref / object-stream resolution algorithm

`PdfDocumentParser.parse` (`infrastructure/parsing/document_parser.py`):

1. Confirm a `%PDF-` header exists anywhere in the file (else `NotAPdfError`).
2. Locate `startxref` from the end of the file and follow `/Prev` (via
   `trailer_resolver.walk_revision_chain`), parsing each revision as either a
   classic xref table (`xref_table_parser.py`) or an xref stream
   (`xref_stream_parser.py`) — whichever the offset actually contains. Hybrid
   files' `/XRefStm` supplementary stream is merged into the same revision
   without overriding the classic table's own entries.
3. Merge every revision's entries, most-recent revision winning per object
   number. If that yields nothing, fall back to scanning the whole file for
   `N G obj` patterns (`AnomalyCode.XREF_UNPARSEABLE_FALLBACK_USED`) — a
   forensics tool must still report what it can about a file whose xref is
   itself the tampering. If even that finds nothing, `UnrecoverableStructureError`.
4. Resolve every in-use entry to a `PdfObject`. A mismatch between the xref's
   claimed offset and what's actually there (`AnomalyCode.XREF_OFFSET_MISMATCH`)
   triggers a targeted regex recovery scan for that specific `N G obj`.
5. Expand every referenced object stream (`object_stream_expander.py`) once
   per container, merging its packed objects into the object table.
6. Detect `/Encrypt` (flag only — no decryption) and, throughout steps 2–4,
   duplicate object-id definitions within one revision or the brute-force scan
   (`AnomalyCode.DUPLICATE_OBJECT_ID`, latest occurrence wins).

## Deferred to later modules

Represented only as flags or untouched raw dictionaries so these types won't
need to change shape when the real logic arrives:

| Feature | Current representation |
|---|---|
| Encryption/decryption | `PdfDocument.is_encrypted`, `.encryption_dict` (raw, unparsed) |
| Digital signatures | Untouched inside whatever dictionary carries them |
| Fonts, images, annotations, forms, embedded files | Not parsed; visible as ordinary dictionaries/streams |
| XMP / Info dictionary semantics | Info dict kept as a raw `PdfDictionary` |
| Linearization | `/Linearized` dict left unparsed |
| Filters other than FlateDecode | `PdfStream.filter_names` records what's claimed; `AnomalyCode.UNSUPPORTED_FILTER` when decoding is skipped |

## Testing

`tests/fixtures/pdf_builder.py` builds byte-exact PDFs programmatically (single-
revision, incremental-update, xref-stream, object-stream, broken-offset, and
duplicate-object-id fixtures) rather than shipping opaque binary files, since
exact byte offsets are what the tests are actually verifying.
