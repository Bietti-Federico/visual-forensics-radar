"""Document Catalog features (ISO 32000-1 §7.7.2) — resolved via the trailer's /Root.

Only the `/Pages/Count` key is read for a page count — walking the full page
tree is explicitly deferred (see `plugins/features/__init__.py`), but the
total leaf-page count is a direct dictionary lookup, not a tree walk, so it's
in scope.
"""

from __future__ import annotations

from pdf_forensics.domain.features.enums import FeatureCategory, FeatureType
from pdf_forensics.domain.features.feature import Feature
from pdf_forensics.domain.pdf.document import PdfDocument
from pdf_forensics.domain.pdf.objects import PdfDictionary
from pdf_forensics.plugins.features._shared import make_feature


class CatalogFeatureExtractor:
    category = FeatureCategory.CATALOG

    def extract(self, document: PdfDocument) -> list[Feature]:
        catalog = self._resolve_catalog(document)

        if catalog is None:
            return [
                make_feature(
                    self.category,
                    "catalog.has_catalog",
                    False,
                    FeatureType.BOOLEAN,
                    "Whether the trailer's /Root entry resolves to a dictionary.",
                    "PdfDocument.trailer./Root",
                ),
            ]

        pages = self._resolve_pages(document, catalog)
        page_count = pages.get_int("Count") if pages is not None else None

        return [
            make_feature(
                self.category,
                "catalog.has_catalog",
                True,
                FeatureType.BOOLEAN,
                "Whether the trailer's /Root entry resolves to a dictionary.",
                "PdfDocument.trailer./Root",
            ),
            make_feature(
                self.category,
                "catalog.type_is_catalog",
                catalog.get_name("Type") == "Catalog",
                FeatureType.BOOLEAN,
                "Whether /Root's /Type is exactly the name /Catalog.",
                "Root./Type",
            ),
            make_feature(
                self.category,
                "catalog.version",
                catalog.get_name("Version"),
                FeatureType.STRING,
                "/Root/Version, which overrides the header's %PDF-x.y version per spec; "
                "None if absent.",
                "Root./Version",
            ),
            make_feature(
                self.category,
                "catalog.page_count",
                page_count,
                FeatureType.INTEGER,
                "/Root/Pages/Count: the total number of leaf pages, or None if unresolvable.",
                "Root./Pages/Count",
            ),
            make_feature(
                self.category,
                "catalog.has_acroform",
                catalog.get("AcroForm") is not None,
                FeatureType.BOOLEAN,
                "Whether the catalog has an /AcroForm entry (an interactive form is present).",
                "Root./AcroForm",
            ),
            make_feature(
                self.category,
                "catalog.has_outlines",
                catalog.get("Outlines") is not None,
                FeatureType.BOOLEAN,
                "Whether the catalog has an /Outlines entry (a document outline/bookmarks).",
                "Root./Outlines",
            ),
        ]

    def _resolve_catalog(self, document: PdfDocument) -> PdfDictionary | None:
        root_ref = document.trailer.get_ref("Root")
        if root_ref is None:
            return None
        root_obj = document.resolve(root_ref)
        if root_obj is None or not isinstance(root_obj.value, PdfDictionary):
            return None
        return root_obj.value

    def _resolve_pages(self, document: PdfDocument, catalog: PdfDictionary) -> PdfDictionary | None:
        pages_ref = catalog.get_ref("Pages")
        if pages_ref is None:
            return None
        pages_obj = document.resolve(pages_ref)
        if pages_obj is None or not isinstance(pages_obj.value, PdfDictionary):
            return None
        return pages_obj.value
