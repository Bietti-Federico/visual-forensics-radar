"""A dependency-free, byte-exact PDF builder for tests.

Exact byte offsets matter for xref testing (that's the whole point of the module
under test), so fixtures are assembled here rather than shipped as opaque binary
files — every offset a test asserts on is computed the same way the parser has to
recompute it, making mismatches easy to diagnose.
"""

from __future__ import annotations

import zlib

_HEADER = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"


class PdfBuilder:
    def __init__(self) -> None:
        self._chunks: list[bytes] = [_HEADER]
        self._pos = len(_HEADER)

    @property
    def pos(self) -> int:
        """Current byte offset — the offset the next appended object would start at."""
        return self._pos

    def _append(self, data: bytes) -> None:
        self._chunks.append(data)
        self._pos += len(data)

    def add_raw(self, data: bytes) -> int:
        """Append arbitrary bytes verbatim; returns the offset they start at."""
        offset = self._pos
        self._append(data)
        return offset

    def add_object(self, obj_num: int, generation: int, body: str) -> int:
        """Append `N G obj\\nBODY\\nendobj\\n`; returns the byte offset it starts at."""
        offset = self._pos
        text = f"{obj_num} {generation} obj\n{body}\nendobj\n".encode("latin-1")
        self._append(text)
        return offset

    def add_stream_object(
        self, obj_num: int, generation: int, dict_body: str, stream_data: bytes
    ) -> int:
        """Append a stream object; `dict_body` should already include `/Length`."""
        offset = self._pos
        text = (
            f"{obj_num} {generation} obj\n{dict_body}\nstream\n".encode("latin-1")
            + stream_data
            + b"\nendstream\nendobj\n"
        )
        self._append(text)
        return offset

    def add_classic_xref_and_trailer(
        self,
        entries: list[tuple[int, int, int, str]],
        size: int,
        root_ref: str,
        prev: int | None = None,
        extra_trailer: str = "",
    ) -> int:
        """Append a classic `xref`/`trailer`/`startxref` block. Returns the `xref` offset.

        `entries` is a list of `(obj_num, byte_offset, generation, 'n' or 'f')`, in
        the exact order to emit them — consecutive entries whose object numbers
        increase by exactly 1 are grouped into one `start count` subsection, same
        as any other run gets its own subsection (including two runs claiming the
        *same* object number, deliberately, for malformed-file fixtures). Object 0
        (the free-list head) is NOT added automatically; include it explicitly.
        """
        xref_offset = self._pos
        lines = [b"xref\n"]
        i, n = 0, len(entries)
        while i < n:
            start_num = entries[i][0]
            j = i
            while j + 1 < n and entries[j + 1][0] == entries[j][0] + 1:
                j += 1
            lines.append(f"{start_num} {j - i + 1}\n".encode("ascii"))
            for k in range(i, j + 1):
                _, off, gen, kind = entries[k]
                lines.append(f"{off:010d} {gen:05d} {kind} \n".encode("ascii"))
            i = j + 1

        trailer_parts = [f"/Size {size}", f"/Root {root_ref}"]
        if prev is not None:
            trailer_parts.append(f"/Prev {prev}")
        if extra_trailer:
            trailer_parts.append(extra_trailer)
        trailer_text = "<< " + " ".join(trailer_parts) + " >>"

        tail = b"".join(
            lines
        ) + f"trailer\n{trailer_text}\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
        self._append(tail)
        return xref_offset

    def add_xref_stream(
        self,
        obj_num: int,
        entries: dict[int, tuple[int, int, int]],
        size: int,
        root_ref: str,
        prev: int | None = None,
        compress: bool = True,
    ) -> int:
        """Append a `/Type /XRef` stream object (PDF 1.5+) and its `startxref`/`%%EOF`.

        `entries` maps object number -> (type_code, field2, field3) for every
        object number *except* `obj_num` itself, whose own entry (type 1, its own
        offset) is added automatically. Returns the xref stream object's offset.
        """
        self_offset = self._pos
        full_entries = dict(entries)
        full_entries[obj_num] = (1, self_offset, 0)

        w1, w2, w3 = 1, 2, 1
        row = bytearray()
        for n in range(size):
            type_code, field2, field3 = full_entries.get(n, (0, 0, 0))
            row += type_code.to_bytes(w1, "big")
            row += field2.to_bytes(w2, "big")
            row += field3.to_bytes(w3, "big")
        raw = bytes(row)

        stream_data = zlib.compress(raw) if compress else raw
        dict_parts = [
            "/Type /XRef",
            f"/Size {size}",
            f"/W [{w1} {w2} {w3}]",
            f"/Root {root_ref}",
            f"/Length {len(stream_data)}",
        ]
        if prev is not None:
            dict_parts.append(f"/Prev {prev}")
        if compress:
            dict_parts.append("/Filter /FlateDecode")
        dict_text = "<< " + " ".join(dict_parts) + " >>"

        self.add_stream_object(obj_num, 0, dict_text, stream_data)
        tail = f"startxref\n{self_offset}\n%%EOF\n".encode("ascii")
        self._append(tail)
        return self_offset

    def add_object_stream(
        self,
        container_obj_num: int,
        packed_objects: list[tuple[int, str]],
        compress: bool = True,
    ) -> int:
        """Append a `/Type /ObjStm` container packing `packed_objects` as `(obj_num, body)`."""
        body = bytearray()
        offsets: list[tuple[int, int]] = []
        for obj_num, obj_body in packed_objects:
            offsets.append((obj_num, len(body)))
            body += obj_body.encode("latin-1")
            body += b" "

        header = (" ".join(f"{n} {off}" for n, off in offsets) + " ").encode("ascii")
        raw = header + bytes(body)
        stream_data = zlib.compress(raw) if compress else raw

        dict_parts = [
            "/Type /ObjStm",
            f"/N {len(packed_objects)}",
            f"/First {len(header)}",
            f"/Length {len(stream_data)}",
        ]
        if compress:
            dict_parts.append("/Filter /FlateDecode")
        dict_text = "<< " + " ".join(dict_parts) + " >>"

        return self.add_stream_object(container_obj_num, 0, dict_text, stream_data)

    def build(self) -> bytes:
        return b"".join(self._chunks)
