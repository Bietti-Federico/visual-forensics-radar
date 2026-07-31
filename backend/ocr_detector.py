import logging
import re
from dataclasses import dataclass

import easyocr
import numpy as np
from PIL import Image, ImageOps, ImageFilter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OCRDetector")


@dataclass
class OcrWord:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]


class OCRDetector:
    """
    Lightweight OCR stage tuned for document triage.
    Extracts words, candidate numeric regions and a compact structural summary.
    Backed by EasyOCR (neural detector+recognizer) — handles real-world photo
    conditions like colored/highlighted backgrounds and non-standard fonts noticeably
    better than a classical pipeline (Tesseract was tried side-by-side and read
    significantly worse on these documents, so it was removed rather than kept as a
    second, weaker option).
    """

    def __init__(self):
        # Loaded ONCE and reused across requests, same pattern as every other model in
        # this project (see lifespan() in main.py). gpu=False since this deployment has
        # no GPU available.
        logger.info("Loading EasyOCR reader (es+en)...")
        self.reader = easyocr.Reader(["es", "en"], gpu=False)
        logger.info("EasyOCR reader ready.")

    CRITICAL_KEYWORDS = (
        "total",
        "neto",
        "cbu",
        "cuil",
        "cuit",
        "dni",
        "doc",
        "fecha",
        "haber",
        "descuento",
        "cobrar",
        "cuenta",
        "importe",
        "monto",
        "saldo",
        "benef",
        "periodo",
        "sueldo",
        "recibo",
        "comprobante",
        "aporte",
        "jubila",
        "pension",
        "pensión",
        "beneficio",
        "titular",
        "vencimiento",
        "resumen",
        "detalle",
        "retencion",
        "retención",
        "movimiento",
        "transferencia",
        "alias",
        "operacion",
        "operación",
        "destinatario",
        "origen",
        "cvu",
    )

    MONTH_NAMES = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
        "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
        "noviembre": 11, "diciembre": 12,
    }

    PERIOD_KEYWORD_PATTERN = re.compile(
        r"per[ií]odo(?:\s+(?:abonado|liquidado))?|per\.?\s*abonado|per\.?\s*liquidado|sueldo\s*mes|mes\s*liquidado",
        re.IGNORECASE,
    )

    RANGE_BOUNDARY_PATTERN = re.compile(r"\bdesde\b|\bhasta\b", re.IGNORECASE)

    # Legal/reference lines (acta, acuerdo, artículo...) routinely contain date-shaped
    # substrings ("A.A.16/06/19", "Ac.09/10") that are reference labels, not receipt
    # dates — grammatically valid dates that happen to sit in the wrong kind of line.
    REFERENCE_LINE_KEYWORDS = ("acta", "acuerdo", "art.", " art ", "resolucion", "resolución", "decreto", "ley ")

    # A bbox straddling two adjacent table cells (common in tightly-packed screenshots)
    # gets OCR'd as one garbled token concatenating both values, e.g. "342,902.340,25"
    # merging "342,90" and "2.340,25" — two or more decimal endings, or more than one
    # currency sign, in a single token reliably signals this rather than a real field.
    MERGED_TOKEN_PATTERN = re.compile(r",\d{2}")

    PERIOD_VALUE_PATTERN = re.compile(
        r"(?:(?<!\d)(?P<mm>0?[1-9]|1[0-2])[\-/](?P<yyyy>\d{4}))"
        # (?<!\d) on the compact "yymm" form prevents it from reading the trailing 4
        # digits of an unrelated longer number (e.g. a receipt/tramite number) that
        # simply happens to end in a valid month.
        r"|(?:(?<!\d)(?P<yy2>\d{2})(?P<mm2>0[1-9]|1[0-2])\b)"
        r"|(?:(?P<month_name>enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\s*(?P<year_after_name>\d{4})?)",
        re.IGNORECASE,
    )

    NUMERIC_PATTERN = re.compile(r"[0-9][0-9\.,:/\-]*")
    DATE_PATTERN = re.compile(r"\b(?:[0-3]?\d[\-/][01]?\d[\-/](?:\d{2}|\d{4})|[01]?\d[\-/](?:\d{2}|\d{4}))\b")
    AMOUNT_PATTERN = re.compile(r"\$?\s*\(?\s*[0-9]{1,3}(?:[\.,][0-9]{3})*(?:[\.,][0-9]{1,2})\s*\)?")
    DNI_TRAMITE_PATTERN = re.compile(r"(?:tramite|tr[aá]mite)[^0-9]{0,20}([0-9]{6,14})", re.IGNORECASE)
    DNI_NUMBER_PATTERN = re.compile(r"\b([0-9]{7,8})\b")
    CARD_NUMBER_PATTERN = re.compile(r"(?:\b\d[\d\s\-]{11,22}\d\b|\b\*{2,}\s*\d{4,6}\b)")
    # A real CUIL/CUIT is always exactly 11 digits (2-8-1), with or without the
    # separating dashes.
    CUIL_PATTERN = re.compile(r"\b(\d{2}-?\d{8}-?\d)\b")
    CUIL_DASHED_PATTERN = re.compile(r"\d{2}-\d{7,8}-\d")
    PERSON_LABEL_PATTERN = re.compile(r"personal|titular|beneficiario|apellido\s+y\s+nombre", re.IGNORECASE)

    # Structural/identity lines already covered by a dedicated extractor (DNI, CUIL,
    # período, persona, banco) — excluded from the generic line-item extraction below
    # so a legajo/DNI/account number never gets miscategorized as a receipt concept.
    # "concepto" is included because a "Concepto | Haberes | Deducciones" column-header
    # row is meant to be itemized via each concept's OWN line (already self-sufficient,
    # label + amount together) — pairing the header itself against the row below would
    # duplicate/garble the first concept instead of adding anything.
    LINE_ITEM_EXCLUDE_KEYWORDS = (
        "legajo", "dni", "cuil", "cuit", "periodo", "período", "nro.rec", "nro rec",
        "localidad", "destino", "personal", "titular", "beneficiario", "acreditado",
        "caja de ahorro", "sucursal", "cbu", "cvu", "alias", "concepto",
        "n° de beneficio", "nro de beneficio", "n de beneficio", "prestacion", "prestación",
    )

    # A CUIL/CUIT (XX-XXXXXXXX-X) is never itself a monetary amount, even though its
    # digit-and-dash shape can otherwise look like a plausible trailing "amount" (e.g.
    # the "-7" tail of "20-44389704-7" would match a negative single-digit amount) —
    # any line containing one is identity data end to end, handled by extract_cuil.
    IDENTITY_DASH_PATTERN = re.compile(r"\b\d{1,2}-\d{6,9}-\d\b")

    # An Argentine payroll/government receipt names its issuing organism somewhere in
    # the header — matched independently of (and in addition to) BANK_KEYWORDS, which
    # only covers homebanking/bank statements.
    ISSUER_KEYWORDS = {
        "anses": "ANSES",
        "consejo provincial de educacion": "Consejo Provincial de Educación",
        "consejo provincial de educación": "Consejo Provincial de Educación",
        "provincia de santa cruz": "Provincia de Santa Cruz",
        "provincia de buenos aires": "Provincia de Buenos Aires",
        "municipalidad": "Municipalidad",
        "ministerio": "Ministerio",
        "policia": "Policía",
        "policía": "Policía",
    }

    BANK_KEYWORDS = (
        "BBVA",
        "SANTANDER",
        "PROVINCIA",
        "NACION",
        "NACIÓN",
        "NATIVA",
        "GALICIA",
        "PATAGONIA",
        "MACRO",
        "ICBC",
        "SUPERVIELLE",
        "CREDICOOP",
        "CIUDAD",
        "HSBC",
        "MERCADOPAGO",
        "MERCADO PAGO",
        "FRANCES",
        "FRANCA",
        "BNA",
        "BANCO",
    )

    RECEIPT_VALUE_PATTERNS = {
        "total_haberes": [r"TOTAL\s+HABERES?", r"\bHABERES\b"],
        "total_descuentos": [r"TOTAL\s+DESCUENTOS?", r"\bDESCUENTOS\b"],
        "neto": [r"NETO\s+ANS(?:ES)?", r"\bNETO\b"],
        "neto_a_cobrar": [r"NETO\s+A\s+COBRAR", r"A\s+COBRAR", r"NETO\s+EFECTIVO", r"LIQ(?:\.|UIDO)?\s*(?:QUINCENA|MENSUAL)?"],
        "liquido": [r"LIQ\.?\s*(?:QUINCENA|MENSUAL)", r"TOTAL\s+LIQUIDO"],
    }

    FIELD_DISPLAY_NAMES = {
        "total_haberes": "Total Haberes",
        "total_descuentos": "Total Descuentos",
        "neto": "Neto",
        "neto_a_cobrar": "Neto a Cobrar",
        "liquido": "Líquido",
    }

    RECEIPT_LABEL_HINTS = (
        "haberes",
        "descuentos",
        "neto",
        "liquido",
        "a cobrar",
        "total",
        "periodo",
        "periodo liquidado",
        "periodo abonado",
        "fecha",
        "venc",
        "item",
        "cod",
        "concepto",
        "cuota",
        "retencion",
        "retención",
        "acreditado",
        "sueldo",
        "haber mensual",
        "aporte",
        "jubilatorio",
        "beneficio",
        "prestacion",
        "prestación",
        "titular",
        "vencimiento",
        "resumen",
        "detalle",
        "saldo",
        "importe",
        "monto",
        "movimiento",
        "transferencia",
        "alias",
        "operacion",
        "operación",
        "destinatario",
        "origen",
        "cvu",
        "cbu",
    )

    def _preprocess(self, image: Image.Image) -> tuple[Image.Image, float]:
        """
        Returns the downscaled image plus the inverse scale factor needed to map
        coordinates from the downscaled image back to the original image's
        coordinate space. Unlike Tesseract, EasyOCR expects something close to the
        natural image (its own pipeline already normalizes contrast/binarization),
        so this only downscales for speed on very large images — no grayscale,
        autocontrast or sharpening.
        """
        image = image.convert("RGB")
        max_width = 2000
        inverse_scale = 1.0
        if image.width > max_width:
            ratio = max_width / image.width
            inverse_scale = image.width / max_width
            image = image.resize((max_width, int(image.height * ratio)))
        return image, inverse_scale

    def _cluster_into_lines(self, tokens: list[dict]) -> list[tuple]:
        """
        EasyOCR gives no block/par/line numbers the way Tesseract does, so lines are
        reconstructed geometrically: sort tokens by vertical center, then group
        consecutive tokens whose vertical center falls within a tolerance (a
        fraction of token height) of the running line's center. Returns a line_id
        (an opaque tuple, same role as Tesseract's (block, par, line) tuple) per
        token, in the same order as the input list.
        """
        indexed = sorted(range(len(tokens)), key=lambda i: (tokens[i]["bbox"][1] + tokens[i]["bbox"][3]) / 2)

        line_ids: list[tuple] = [None] * len(tokens)
        current_line = 0
        current_center = None
        current_height = None

        for i in indexed:
            x1, y1, x2, y2 = tokens[i]["bbox"]
            center_y = (y1 + y2) / 2
            height = max(1, y2 - y1)

            if current_center is None:
                current_center = center_y
                current_height = height
            else:
                tolerance = 0.6 * max(current_height, height)
                if abs(center_y - current_center) > tolerance:
                    current_line += 1
                    current_center = center_y
                    current_height = height
                else:
                    # Running average keeps the line's reference center stable as
                    # more tokens are added to it.
                    current_center = (current_center + center_y) / 2
                    current_height = (current_height + height) / 2

            line_ids[i] = ("line", current_line)

        return line_ids

    def analyze(self, image_path: str) -> dict:
        try:
            image = Image.open(image_path)
            processed, inverse_scale = self._preprocess(image)

            raw_results = self.reader.readtext(np.array(processed), detail=1, paragraph=False)

            combined_tokens = []
            for polygon, raw_text, confidence in raw_results:
                raw_text = str(raw_text).strip()
                if not raw_text:
                    continue
                xs = [point[0] for point in polygon]
                ys = [point[1] for point in polygon]
                bbox = (
                    int(min(xs) * inverse_scale),
                    int(min(ys) * inverse_scale),
                    int(max(xs) * inverse_scale),
                    int(max(ys) * inverse_scale),
                )
                combined_tokens.append({
                    "text": raw_text,
                    "confidence": float(confidence) * 100.0,
                    "bbox": bbox,
                })

            line_ids = self._cluster_into_lines(combined_tokens)
            for token, line_id in zip(combined_tokens, line_ids):
                token["line_id"] = line_id

            # Unlike Tesseract's image_to_data (which returns tokens in guaranteed
            # block/par/line/word reading order), EasyOCR's readtext() returns
            # detections in whatever order its detector found them — not necessarily
            # top-to-bottom/left-to-right. Every proximity-based heuristic downstream
            # (HEADER_ROW_CONTEXT_PATTERN's "concepto" lookback, extract_amount_after_
            # keywords' forward window) assumes a label's neighboring text sits near it
            # in the flattened text_blob, so tokens must be restored to reading order —
            # by line (already clustered above), then left-to-right within each line —
            # before text_blob/line_text_map are built.
            combined_tokens.sort(key=lambda t: (t["line_id"][1], t["bbox"][0]))

            words: list[OcrWord] = []
            candidate_regions = []
            keyword_hits = []
            numeric_tokens = 0
            confidence_values = []
            token_entries = []
            line_text_map = {}
            line_bbox_map: dict[tuple, list[int]] = {}
            # Parallel to line_text_map (same order/index per line) — line_text_map only
            # keeps the joined text, which loses each token's own position. Needed to
            # align a column header against the matching value token on another line
            # (see _extract_table_column_value), which line_bbox_map's single aggregate
            # bbox per line can't provide.
            line_token_bboxes: dict[tuple, list[tuple[int, int, int, int]]] = {}

            for token in combined_tokens:
                raw_text = token["text"]
                confidence = token["confidence"]
                bbox = token["bbox"]
                line_id = token["line_id"]

                words.append(OcrWord(text=raw_text, confidence=confidence, bbox=bbox))
                confidence_values.append(confidence)

                line_text_map.setdefault(line_id, []).append(raw_text)
                line_token_bboxes.setdefault(line_id, []).append(bbox)

                x1, y1, x2, y2 = bbox
                if line_id not in line_bbox_map:
                    line_bbox_map[line_id] = [x1, y1, x2, y2]
                else:
                    box = line_bbox_map[line_id]
                    box[0] = min(box[0], x1)
                    box[1] = min(box[1], y1)
                    box[2] = max(box[2], x2)
                    box[3] = max(box[3], y2)

                normalized = raw_text.lower()
                if any(keyword in normalized for keyword in self.CRITICAL_KEYWORDS):
                    keyword_hits.append(raw_text)

                is_numeric = bool(self.NUMERIC_PATTERN.search(raw_text))
                is_date = bool(self.DATE_PATTERN.search(raw_text))
                is_amount = bool(self.AMOUNT_PATTERN.search(raw_text))

                if is_numeric:
                    numeric_tokens += 1

                token_entries.append({
                    "text": raw_text,
                    "normalized": normalized,
                    "bbox": bbox,
                    "confidence": confidence,
                    "line_id": line_id,
                    "is_numeric": is_numeric,
                    "is_date": is_date,
                    "is_amount": is_amount,
                })

            for entry in token_entries:
                line_tokens = line_text_map.get(entry["line_id"], [])
                line_text = " ".join(line_tokens).lower()
                has_key_hint = any(hint in line_text for hint in self.RECEIPT_LABEL_HINTS)

                if not entry["is_numeric"] and not has_key_hint:
                    continue

                # A bbox that merged two adjacent cells into one garbled token isn't a
                # real field at all — analyzing it with ELA/typography would just
                # measure a meaningless crop spanning a cell boundary.
                if len(self.MERGED_TOKEN_PATTERN.findall(entry["text"])) >= 2 or entry["text"].count("$") >= 2:
                    continue

                is_reference_line = any(keyword in line_text for keyword in self.REFERENCE_LINE_KEYWORDS)
                effective_is_date = entry["is_date"] and not is_reference_line

                priority = 0
                region_type = "context"
                is_key_field = False

                if entry["is_numeric"] and (entry["is_amount"] or effective_is_date) and has_key_hint:
                    priority = 4
                    region_type = "key_numeric"
                    is_key_field = True
                elif entry["is_numeric"] and has_key_hint:
                    # Numeric and sitting near a receipt keyword, but not recognizably an
                    # amount or date — typically a reference/ID number (BENEFICIO NRO,
                    # CUENTA NRO, COMPROBANTE, CODIGO CAJERO...). There's no "expected
                    # value" to judge a reference number against, so it's still analyzed
                    # (kept as a candidate) but NOT treated as a key field — it must not
                    # get the key-field ELA weight boost or trigger the strong-evidence
                    # escalations meant for genuine amount/date tampering.
                    priority = 3
                    region_type = "line_numeric"
                elif entry["is_numeric"] and (entry["is_amount"] or effective_is_date):
                    priority = 2
                    region_type = "generic_numeric"
                elif has_key_hint:
                    priority = 1
                    region_type = "key_context"

                if priority > 0:
                    candidate_regions.append({
                        "text": entry["text"],
                        "bbox": entry["bbox"],
                        "type": region_type,
                        "priority": priority,
                        "is_key_field": is_key_field,
                        "is_numeric": entry["is_numeric"],
                        "is_date": effective_is_date,
                        "is_amount": entry["is_amount"],
                        "line_text": " ".join(line_tokens)[:220],
                    })

            candidate_regions.sort(
                key=lambda item: (item.get("priority", 0), len(item.get("text", ""))),
                reverse=True,
            )

            mean_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
            text_blob = " ".join(word.text for word in words)
            text_blob_upper = text_blob.upper()
            document_like = len(words) >= 8 or len(keyword_hits) >= 2 or numeric_tokens >= 3

            dni_tramite = self.extract_dni_tramite(text_blob)
            dni_number = self.extract_dni_number(text_blob)
            bank_name = self.extract_bank_name(text_blob_upper)
            card_numbers = self.extract_card_numbers(text_blob)
            masked_card_numbers = self.extract_masked_card_numbers(text_blob)
            cuil = self.extract_cuil(text_blob, line_text_map, line_bbox_map, line_token_bboxes)
            persona = self.extract_person_name(line_text_map, line_bbox_map, line_token_bboxes)
            tipo_recibo = self.classify_receipt_issuer(text_blob_upper, bank_name)
            period_info = self.analyze_period_consistency(text_blob)
            periodo = period_info["periods_found"][0]["raw"] if period_info["periods_found"] else None
            line_items = self.extract_line_items(line_text_map, line_bbox_map, line_token_bboxes)
            template_detail = self.extract_template_detail(
                tipo_recibo, persona, line_text_map, line_bbox_map, line_token_bboxes
            )
            receipt_consistency = self.analyze_receipt_consistency(
                text_blob, mean_confidence, image=image, line_text_map=line_text_map, line_bbox_map=line_bbox_map,
                line_token_bboxes=line_token_bboxes,
            )

            score = 0
            signals = []

            if not words:
                score += 2
                signals.append("No se pudo extraer texto útil con OCR.")
            else:
                if document_like:
                    if len(keyword_hits) == 0:
                        score += 1
                        signals.append("OCR no encontró campos críticos habituales del documento.")
                    if mean_confidence < 45:
                        score += 1
                        signals.append(f"Confianza OCR baja para un documento ({mean_confidence:.1f}%).")
                else:
                    if mean_confidence < 35:
                        score += 1
                        signals.append(f"La lectura OCR es débil ({mean_confidence:.1f}%).")

            if document_like and numeric_tokens == 0:
                score += 1
                signals.append("No se detectaron tokens numéricos relevantes en un documento esperado.")

            richness_score = 0
            if words:
                richness_score += 1
            if numeric_tokens >= 3:
                richness_score += 1
            if len(keyword_hits) >= 2:
                richness_score += 1
            if mean_confidence >= 70:
                richness_score += 1

            result = {
                "status": "success",
                "word_count": len(words),
                "numeric_token_count": numeric_tokens,
                "keyword_hits": keyword_hits[:10],
                "mean_confidence": round(mean_confidence, 2),
                "document_like": document_like,
                "richness_score": richness_score,
                "text_excerpt": text_blob[:800],
                "text_blob": text_blob,
                # No practical cap on real documents (a receipt rarely has more than a
                # few dozen candidate fields) — this is only a sanity ceiling against a
                # pathologically noisy OCR read, so a genuine key field is never silently
                # excluded just because it ranked below some arbitrary cutoff.
                "candidate_regions": candidate_regions[:200],
                "suspicious_signals": signals,
                "ocr_score": min(score, 4),
                "ocr_quality_score": min(richness_score, 4),
                "receipt_consistency": receipt_consistency,
                "line_items": line_items,
                "template_detail": template_detail,
                "extracted_fields": {
                    "dni_tramite": dni_tramite,
                    "dni_number": dni_number,
                    "bank_name": bank_name,
                    "card_numbers": card_numbers,
                    "masked_card_numbers": masked_card_numbers,
                    "cuil": cuil,
                    "persona": persona,
                    "tipo_recibo": tipo_recibo,
                    "periodo": periodo,
                },
            }

            logger.info("OCR analysis complete.")
            return result

        except Exception as e:
            logger.error(f"Error during OCR analysis: {str(e)}")
            return {"status": "error", "message": str(e)}

    def extract_dni_tramite(self, text_blob: str) -> str | None:
        match = self.DNI_TRAMITE_PATTERN.search(text_blob)
        if match:
            return match.group(1)
        return None

    def extract_dni_number(self, text_blob: str) -> str | None:
        candidates = self.DNI_NUMBER_PATTERN.findall(text_blob)
        for candidate in candidates:
            if len(candidate) >= 7:
                return candidate
        return None

    def extract_bank_name(self, text_blob_upper: str) -> str | None:
        for keyword in self.BANK_KEYWORDS:
            if keyword in text_blob_upper:
                return keyword.title()
        return None

    def extract_card_numbers(self, text_blob: str) -> list[str]:
        matches = self.CARD_NUMBER_PATTERN.findall(text_blob)
        numbers = []
        for match in matches:
            normalized = re.sub(r"[^0-9]", "", match)
            if 13 <= len(normalized) <= 19:
                numbers.append(normalized)
        return list(dict.fromkeys(numbers))

    def extract_masked_card_numbers(self, text_blob: str) -> list[str]:
        masked = []
        for match in re.findall(r"\*{2,}\s*\d{3,6}", text_blob):
            masked.append(match.replace(" ", ""))
        return list(dict.fromkeys(masked))

    # A value token that legitimately belongs to a column starts (not just centers)
    # at or after that column's own left edge — checking the CENTER instead let an
    # oversized neighboring value (e.g. a long name overflowing past its own narrow
    # header's width) bleed into the LAST column's zone whenever that column has no
    # right neighbor to cap it (nothing stops an unbounded-right span). A small
    # tolerance absorbs ordinary kerning/alignment jitter between a header and its
    # value without reopening that hole.
    LABELED_COLUMN_BOUNDARY_TOLERANCE_PX = 12

    def _find_labeled_column_value(
        self,
        label_pattern: re.Pattern,
        line_text_map: dict,
        line_bbox_map: dict,
        line_token_bboxes: dict,
        is_valid_value,
        find_all: bool = False,
    ) -> str | list[str] | None:
        """
        Looks up a label that may appear either inline on the same line as its value
        ("CUIL: 20-44389704-7") or as a column header in a 2-line table — e.g. a
        "Titular" header with the name sitting on the line below, aligned by X
        position (the same layout extract_line_items' Case B handles for amounts).

        For the column-header case, the label's own column boundaries are taken from
        its IMMEDIATE neighboring header tokens on the same line (whatever sits to the
        left of the next column's header start, and to the right of the previous
        column's header end) — every value-line token whose bbox falls inside that
        span (start/end, not just center — see LABELED_COLUMN_BOUNDARY_TOLERANCE_PX) is
        collected, in left-to-right order, so a multi-word value (a full name under a
        single "Titular" header) isn't truncated to just its nearest token, while an
        oversized NEIGHBORING value that merely overlaps the column's open edge (e.g.
        the last column has no right neighbor to bound it) doesn't bleed in either.

        By default returns the FIRST valid match found (document order). Some
        templates repeat the same label more than once (e.g. a receipt showing a
        "CUIL" column for both the titular and an unset "persona apoderada") — pass
        `find_all=True` to collect every valid match instead, in document order.
        """
        ordered_lines = sorted(line_bbox_map.items(), key=lambda item: item[1][1])
        results = []

        for position, (line_id, _bbox) in enumerate(ordered_lines):
            tokens = line_text_map.get(line_id, [])
            bboxes = line_token_bboxes.get(line_id, [])
            if not tokens or len(tokens) != len(bboxes):
                continue

            line_text = " ".join(tokens)
            match = label_pattern.search(line_text)
            if not match:
                continue

            # Inline: label and value share the same line.
            remainder = line_text[match.end():].strip(" :-.")
            if remainder and is_valid_value(remainder):
                if not find_all:
                    return remainder
                results.append(remainder)
                continue

            # Column header: gather value-line tokens whose column falls between this
            # label's immediate left/right neighbors on the header line.
            matched_boxes = []
            cursor = 0
            for token_text, token_bbox in zip(tokens, bboxes):
                token_start, token_end = cursor, cursor + len(token_text)
                if token_start < match.end() and token_end > match.start():
                    matched_boxes.append(token_bbox)
                cursor = token_end + 1
            if not matched_boxes or position + 1 >= len(ordered_lines):
                continue

            label_x1 = min(box[0] for box in matched_boxes)
            label_x2 = max(box[2] for box in matched_boxes)
            left_boundary = max(
                (box[2] for box in bboxes if box[2] <= label_x1), default=None
            )
            right_boundary = min(
                (box[0] for box in bboxes if box[0] >= label_x2), default=None
            )

            value_line_id, _ = ordered_lines[position + 1]
            value_tokens = line_text_map.get(value_line_id, [])
            value_bboxes = line_token_bboxes.get(value_line_id, [])
            if not value_tokens or len(value_tokens) != len(value_bboxes):
                continue

            tolerance = self.LABELED_COLUMN_BOUNDARY_TOLERANCE_PX
            collected = []
            for value_text, value_bbox in zip(value_tokens, value_bboxes):
                if left_boundary is not None and value_bbox[0] < left_boundary - tolerance:
                    continue
                if right_boundary is not None and value_bbox[2] > right_boundary + tolerance:
                    continue
                collected.append((value_bbox[0], value_text))

            if not collected:
                continue
            collected.sort(key=lambda item: item[0])
            column_value = " ".join(text for _, text in collected).strip(" :-.")
            if column_value and is_valid_value(column_value):
                if not find_all:
                    return column_value
                results.append(column_value)

        return results if find_all else None

    def extract_cuil(
        self,
        text_blob: str,
        line_text_map: dict | None = None,
        line_bbox_map: dict | None = None,
        line_token_bboxes: dict | None = None,
    ) -> str | None:
        """
        Prefers the column-aligned value under/after a "CUIL" label (handles both
        inline "CUIL: X" labels and column-header table layouts) over a bare text-
        proximity search, which can pick up an unrelated 11-digit-shaped number (a
        beneficio/legajo number, say) that merely happens to sit near the word "CUIL"
        in reading order without actually being paired with it.

        Some receipts show a "CUIL" column/label more than once (e.g. one for the
        titular, one for an unset "persona apoderada" that's blank or "-") — a
        genuine CUIL is reliably printed with its separating dashes ("20-44389704-7"),
        while an unrelated same-length reference number usually isn't, so a dashed
        match is tried across ALL "cuil" occurrences before falling back to accepting
        a dashless 11-digit match anywhere.
        """
        if line_text_map and line_bbox_map and line_token_bboxes:
            # A CUIL should never contain whitespace, but the column-collector joins
            # multiple OCR tokens with a space by default (needed for multi-word
            # values like a name) — if EasyOCR split the number itself mid-token (e.g.
            # "20-44389704-7" read as two tokens, "20-44389704" and "-7"), that leaves
            # a stray space that would otherwise make a perfectly good match fail
            # fullmatch validation.
            for validator in (
                lambda v: bool(self.CUIL_DASHED_PATTERN.fullmatch(v.replace(" ", "").strip())),
                lambda v: bool(self.CUIL_PATTERN.fullmatch(v.replace(" ", "").strip())),
            ):
                column_value = self._find_labeled_column_value(
                    re.compile(r"\bcuil\b", re.IGNORECASE),
                    line_text_map, line_bbox_map, line_token_bboxes,
                    is_valid_value=validator,
                )
                if column_value:
                    match = self.CUIL_PATTERN.fullmatch(column_value.replace(" ", "").strip())
                    if match:
                        return match.group(1)

        near_cuil_any = None
        fallback = None
        for match in self.CUIL_PATTERN.finditer(text_blob):
            context_before = text_blob[max(0, match.start() - 15):match.start()]
            near_label = "cuil" in context_before.lower()
            is_dashed = bool(self.CUIL_DASHED_PATTERN.fullmatch(match.group(1)))

            if near_label and is_dashed:
                return match.group(1)
            if near_label and near_cuil_any is None:
                near_cuil_any = match.group(1)
            if fallback is None:
                fallback = match.group(1)
        return near_cuil_any or fallback

    def extract_person_name(
        self,
        line_text_map: dict,
        line_bbox_map: dict | None = None,
        line_token_bboxes: dict | None = None,
    ) -> str | None:
        """
        Looks for a "Titular" column/label first (the most specific, reliable anchor
        for the account/beneficiary holder's name), via column-aware lookup so a
        multi-word name isn't confused with the next column's header ("CUIL"). Falls
        back to the older same-line heuristic for templates that inline the name
        instead of using a column header (e.g. "Personal DNI 25985232 COLMAN ADRIANA
        ALEJANDRA CUIL 27-25985232-9").
        """
        def looks_like_name(value: str) -> bool:
            if not re.search(r"[A-Za-zÀ-ÿ]{3,}", value) or value.strip().isdigit():
                return False
            lowered = value.lower()
            return not any(kw in lowered for kw in ("cuil", "dni", "titular", "personal", "beneficiario"))

        if line_bbox_map and line_token_bboxes:
            column_value = self._find_labeled_column_value(
                re.compile(r"\btitular\b", re.IGNORECASE),
                line_text_map, line_bbox_map, line_token_bboxes,
                is_valid_value=looks_like_name,
            )
            if column_value:
                return column_value

        for tokens in line_text_map.values():
            line_text = " ".join(tokens)
            label_match = self.PERSON_LABEL_PATTERN.search(line_text)
            if not label_match:
                continue

            dni_match = re.search(r"DNI\s*[:\-]?\s*\d{6,9}", line_text, re.IGNORECASE)
            cuil_match = re.search(r"CUIL", line_text, re.IGNORECASE)
            if dni_match and cuil_match and cuil_match.start() > dni_match.end():
                candidate = line_text[dni_match.end():cuil_match.start()].strip(" :-.")
                if candidate:
                    return candidate

            remainder = line_text[label_match.end():]
            name_words = [word for word in remainder.split() if re.fullmatch(r"[A-Za-zÀ-ÿ.]+", word)]
            if name_words:
                return " ".join(name_words)

        return None

    # The issuing organism's name is reliably near the document's own header/logo —
    # checking only a leading window avoids matching an unrelated mention deeper in
    # the body (e.g. a bank statement whose transaction description happens to
    # mention "ANSES" as a transfer counterparty, which isn't this document's issuer).
    ISSUER_HEADER_WINDOW_CHARS = 250

    # A receipt that simply mentions a bank (for deposit/payment info) isn't itself a
    # homebanking screenshot — these payroll-specific signals mean it's a salary/
    # pension receipt regardless of which bank it happens to reference.
    PAYROLL_SIGNAL_KEYWORDS = (
        "haberes", "descuentos", "legajo", "jubilat", "recibo de sueldo",
        "recibo de haberes", "aportes personal", "cct ",
    )

    def classify_receipt_issuer(self, text_blob_upper: str, bank_name: str | None) -> str:
        header_window = text_blob_upper.lower()[:self.ISSUER_HEADER_WINDOW_CHARS]
        for keyword, label in self.ISSUER_KEYWORDS.items():
            if keyword in header_window:
                return label

        lowered = text_blob_upper.lower()
        if bank_name and not any(keyword in lowered for keyword in self.PAYROLL_SIGNAL_KEYWORDS):
            return f"Homebanking / {bank_name}"
        return "Desconocido"

    # --- ANSES-specific structured extraction -------------------------------------
    # Each receipt template gets its own dedicated extraction flow once we've studied
    # it closely enough to model its exact field layout — a generic schema (persona/
    # cuil/periodo) is a reasonable lowest-common-denominator fallback, but templates
    # like this one carry more structure (a titular AND a separate "persona apoderada"
    # with their own CUIL each, a Concepto/Haberes/Deducciones table) worth capturing
    # precisely instead of flattening into the generic fields.
    PRESTACION_LABEL_PATTERN = re.compile(r"prestaci[oó]n", re.IGNORECASE)
    BENEFICIO_LABEL_PATTERN = re.compile(r"n[°'\"]?\s*de\s*beneficio", re.IGNORECASE)
    APODERADA_LABEL_PATTERN = re.compile(r"persona\s+apoderada", re.IGNORECASE)
    CUIL_LABEL_PATTERN = re.compile(r"\bcuil\b", re.IGNORECASE)

    CONCEPTO_HEADER_PATTERN = re.compile(r"\bconcepto\b", re.IGNORECASE)
    HABERES_HEADER_PATTERN = re.compile(r"\bhaberes\b", re.IGNORECASE)
    DEDUCCIONES_HEADER_PATTERN = re.compile(r"\bdeducciones\b", re.IGNORECASE)
    SUBTOTAL_LABEL_PATTERN = re.compile(r"\bsubtotal\b", re.IGNORECASE)
    TOTAL_A_COBRAR_LABEL_PATTERN = re.compile(r"total\s+a\s+cobrar", re.IGNORECASE)

    def _looks_like_name(self, value: str) -> bool:
        if not re.search(r"[A-Za-zÀ-ÿ]{3,}", value) or value.strip().isdigit():
            return False
        lowered = value.lower()
        return not any(kw in lowered for kw in ("cuil", "dni", "titular", "personal", "beneficiario"))

    def extract_template_detail(
        self,
        tipo_recibo: str,
        persona: str | None,
        line_text_map: dict,
        line_bbox_map: dict,
        line_token_bboxes: dict,
    ) -> dict | None:
        """
        Dispatches to a template-specific extraction flow once `tipo_recibo` names a
        receipt whose exact layout we've studied closely enough to model — each one
        gets analyzed and added one at a time as real cases come in, rather than
        forcing every template into one generic schema. Returns None for a template
        that doesn't have a dedicated flow yet (the generic extracted_fields/
        line_items above still apply regardless).
        """
        if tipo_recibo == "ANSES":
            detail = self.extract_anses_detail(line_text_map, line_bbox_map, line_token_bboxes)
            detail["titular"] = persona
            return detail
        return None

    def extract_anses_detail(
        self,
        line_text_map: dict,
        line_bbox_map: dict,
        line_token_bboxes: dict,
    ) -> dict:
        """
        Structured identity fields specific to the ANSES "Recibo de haberes" template:
        a titular row (Prestación/Titular/CUIL) and a second, often-empty apoderada
        row (N° de beneficio/Persona apoderada/CUIL) — both 3-column header+value
        tables handled by _find_labeled_column_value, plus the CUIL label repeating
        once per row (find_all=True keeps them in document order: titular's first,
        apoderada's second).
        """
        prestacion = self._find_labeled_column_value(
            self.PRESTACION_LABEL_PATTERN, line_text_map, line_bbox_map, line_token_bboxes,
            is_valid_value=self._looks_like_name,
        )
        n_beneficio = self._find_labeled_column_value(
            self.BENEFICIO_LABEL_PATTERN, line_text_map, line_bbox_map, line_token_bboxes,
            is_valid_value=lambda v: bool(re.fullmatch(r"[0-9]{5,15}", v.replace(" ", "").strip())),
        )
        persona_apoderada = self._find_labeled_column_value(
            self.APODERADA_LABEL_PATTERN, line_text_map, line_bbox_map, line_token_bboxes,
            is_valid_value=self._looks_like_name,
        )

        cuils = []
        for validator in (
            lambda v: bool(self.CUIL_DASHED_PATTERN.fullmatch(v.replace(" ", "").strip())),
            lambda v: bool(self.CUIL_PATTERN.fullmatch(v.replace(" ", "").strip())),
        ):
            found = self._find_labeled_column_value(
                self.CUIL_LABEL_PATTERN, line_text_map, line_bbox_map, line_token_bboxes,
                is_valid_value=validator, find_all=True,
            )
            for raw in found:
                normalized = self.CUIL_PATTERN.fullmatch(raw.replace(" ", "").strip())
                if normalized and normalized.group(1) not in cuils:
                    cuils.append(normalized.group(1))

        table = self.extract_concepto_haberes_deducciones_table(line_text_map, line_bbox_map, line_token_bboxes)

        return {
            "prestacion": prestacion,
            "titular": None,  # filled in by the caller, which already has extract_person_name's result
            "cuil_titular": cuils[0] if len(cuils) > 0 else None,
            "n_beneficio": n_beneficio,
            "persona_apoderada": persona_apoderada,
            "cuil_apoderada": cuils[1] if len(cuils) > 1 else None,
            "tabla_haberes_deducciones": table,
        }

    def extract_concepto_haberes_deducciones_table(
        self,
        line_text_map: dict,
        line_bbox_map: dict,
        line_token_bboxes: dict,
    ) -> dict:
        """
        The ANSES payroll breakdown table: a "Concepto | Haberes | Deducciones" header
        row, one row per concept (amount in whichever of the two columns applies), a
        "Subtotal" row with ONE amount per column, and a closing "Total a cobrar" row.
        Unlike extract_line_items (generic, single amount per line), several of these
        rows carry TWO amounts side by side (the Subtotal row) — column position
        against the header's own 3 zones tags each token correctly instead of the
        generic extractor's "grab the trailing amount" rule mangling the first one.
        """
        empty_result = {
            "detected": False, "items": [],
            "subtotal_haberes": None, "subtotal_deducciones": None, "total_a_cobrar": None,
        }

        ordered_lines = sorted(line_bbox_map.items(), key=lambda item: item[1][1])

        def match_span_bbox(tokens, bboxes, match):
            boxes = []
            cursor = 0
            for token_text, token_bbox in zip(tokens, bboxes):
                token_start, token_end = cursor, cursor + len(token_text)
                if token_start < match.end() and token_end > match.start():
                    boxes.append(token_bbox)
                cursor = token_end + 1
            if not boxes:
                return None
            return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))

        header_position = None
        concepto_box = haberes_box = deducciones_box = None
        for position, (line_id, _bbox) in enumerate(ordered_lines):
            tokens = line_text_map.get(line_id, [])
            bboxes = line_token_bboxes.get(line_id, [])
            if not tokens or len(tokens) != len(bboxes):
                continue
            line_text = " ".join(tokens)

            concepto_m = self.CONCEPTO_HEADER_PATTERN.search(line_text)
            haberes_m = self.HABERES_HEADER_PATTERN.search(line_text)
            deducciones_m = self.DEDUCCIONES_HEADER_PATTERN.search(line_text)
            if not (concepto_m and haberes_m and deducciones_m):
                continue

            concepto_box = match_span_bbox(tokens, bboxes, concepto_m)
            haberes_box = match_span_bbox(tokens, bboxes, haberes_m)
            deducciones_box = match_span_bbox(tokens, bboxes, deducciones_m)
            header_position = position
            break

        if header_position is None or not (concepto_box and haberes_box and deducciones_box):
            return empty_result

        concepto_haberes_mid = (concepto_box[2] + haberes_box[0]) / 2
        haberes_deducciones_mid = (haberes_box[2] + deducciones_box[0]) / 2

        def zone_for(x_center: float) -> str:
            if x_center < concepto_haberes_mid:
                return "concepto"
            if x_center < haberes_deducciones_mid:
                return "haberes"
            return "deducciones"

        items = []
        subtotal_haberes = subtotal_deducciones = total_a_cobrar = None

        for position in range(header_position + 1, len(ordered_lines)):
            line_id, _bbox = ordered_lines[position]
            tokens = line_text_map.get(line_id, [])
            bboxes = line_token_bboxes.get(line_id, [])
            if not tokens or len(tokens) != len(bboxes):
                continue
            line_text = " ".join(tokens)

            total_match = self.TOTAL_A_COBRAR_LABEL_PATTERN.search(line_text)
            if total_match:
                remainder = line_text[total_match.end():].strip(" :-.")
                parsed = self.parse_amount(remainder) if remainder else None
                if parsed is None and position + 1 < len(ordered_lines):
                    next_line_id, _ = ordered_lines[position + 1]
                    next_tokens = line_text_map.get(next_line_id, [])
                    if next_tokens:
                        parsed = self.parse_amount(" ".join(next_tokens))
                total_a_cobrar = parsed
                break

            zone_tokens = {"concepto": [], "haberes": [], "deducciones": []}
            for token_text, token_bbox in zip(tokens, bboxes):
                center = (token_bbox[0] + token_bbox[2]) / 2
                zone_tokens[zone_for(center)].append(token_text)

            concepto_text = " ".join(zone_tokens["concepto"]).strip(" .:-")
            haberes_text = " ".join(zone_tokens["haberes"]).strip()
            deducciones_text = " ".join(zone_tokens["deducciones"]).strip()

            haberes_value = self.parse_amount(haberes_text) if haberes_text and haberes_text != "-" else None
            deducciones_value = self.parse_amount(deducciones_text) if deducciones_text and deducciones_text != "-" else None

            if not concepto_text:
                continue

            if self.SUBTOTAL_LABEL_PATTERN.search(concepto_text):
                subtotal_haberes = haberes_value
                subtotal_deducciones = deducciones_value
                continue

            if haberes_value is not None or deducciones_value is not None:
                items.append({
                    "concepto": concepto_text,
                    "haberes": haberes_value,
                    "deducciones": deducciones_value,
                })

        return {
            "detected": True,
            "items": items,
            "subtotal_haberes": subtotal_haberes,
            "subtotal_deducciones": subtotal_deducciones,
            "total_a_cobrar": total_a_cobrar,
        }

    def parse_amount(self, raw_value: str) -> float | None:
        if not raw_value:
            return None

        value = raw_value.strip()
        negative = False
        if value.startswith("(") and value.endswith(")"):
            negative = True
            value = value[1:-1]

        value = value.replace("$", "").replace(" ", "")
        value = re.sub(r"[^0-9,\.\-]", "", value)
        if not value:
            return None

        last_comma = value.rfind(",")
        last_dot = value.rfind(".")

        if last_comma == -1 and last_dot == -1:
            try:
                parsed = float(int(value))
                return -parsed if negative else parsed
            except Exception:
                return None

        if last_comma > last_dot:
            normalized = value.replace(".", "").replace(",", ".")
        else:
            normalized = value.replace(",", "")

        try:
            parsed = float(normalized)
            return -parsed if negative else parsed
        except Exception:
            return None

    # Some ANSES-style receipts are laid out as a table with column headers
    # ("Concepto | Haberes | Deducciones") instead of "Etiqueta: $Valor" lines — a bare
    # "HABERES" match right after "CONCEPTO" is that column header, not a labeled total,
    # and there's genuinely no adjacent amount to read (not a font/OCR problem at all).
    # Likewise, "RECIBO DE HABERES" is the canonical title of an Argentine payroll
    # receipt — a bare "HABERES" match inside that title is the document's own name,
    # not a labeled total either, and has no amount anywhere near it.
    HEADER_ROW_CONTEXT_PATTERN = re.compile(r"concepto|recibo\s+de", re.IGNORECASE)

    def extract_amount_after_keywords(self, text_blob: str, keyword_patterns: list[str]) -> tuple[float | None, str | None, bool]:
        """
        Returns (value, raw_text, keyword_found). `keyword_found` is True whenever the
        label itself (e.g. "NETO") was located in the text as a genuine value label
        (not a table column header — see HEADER_ROW_CONTEXT_PATTERN), even if no valid
        amount could be read nearby — distinguishing "this document doesn't have this
        field" from "the label is there but its value is unreadable", which callers
        should treat as a red flag in its own right (e.g. an unusual/edited font that
        OCR can't parse), not silence.
        """
        keyword_found = False
        for keyword_pattern in keyword_patterns:
            for match in re.finditer(keyword_pattern, text_blob, re.IGNORECASE):
                context_before = text_blob[max(0, match.start() - 20):match.start()]
                if self.HEADER_ROW_CONTEXT_PATTERN.search(context_before):
                    continue

                keyword_found = True

                start = match.end()
                window = text_blob[start:start + 120]
                # Every real amount on these receipts is formatted with a mandatory
                # 2-digit decimal (",XX") — requiring it here rejects a bare stray digit
                # ("3") or a comma-stripped OCR misread ("53558199" instead of
                # "535581,99") that would otherwise silently corrupt the arithmetic
                # consistency check with a bogus number. Thousands "." separators are
                # optional since OCR often drops them while keeping the decimal comma.
                amount_match = re.search(r"\(?\$?\s*([0-9]{1,3}(?:\.?[0-9]{3})*,[0-9]{2})\)?", window)
                if amount_match:
                    raw_amount = amount_match.group(0)
                    parsed = self.parse_amount(raw_amount)
                    if parsed is not None:
                        return parsed, raw_amount.strip(), True

        return None, None, keyword_found

    # A column header's value token on the row below rarely lines up pixel-perfect —
    # print/scan skew, kerning and column padding all shift it a bit — but it should
    # still land close to the header's own horizontal span. This bounds how far a
    # "nearest token" match is allowed to be before it's rejected as unrelated.
    TABLE_COLUMN_MAX_GAP_RATIO = 1.5
    TABLE_COLUMN_MIN_GAP_PX = 40

    def _extract_table_column_value(
        self,
        keyword_patterns: list[str],
        line_text_map: dict,
        line_bbox_map: dict,
        line_token_bboxes: dict,
    ) -> tuple[float | None, str | None]:
        """
        Some receipts summarize totals as a 2-line table: a header row naming each
        column ("R. con Aportes  R. sin Aportes  T.Sal.Fam  Retenciones  Total Liquido")
        followed immediately below by one line of values in the same left-to-right
        column order. `extract_amount_after_keywords` only looks a fixed window AHEAD
        on the SAME line, so a label whose value sits on the NEXT line (rather than to
        its right) never finds it — a distinct layout from the single-row
        "Concepto|Haberes|Deducciones" case HEADER_ROW_CONTEXT_PATTERN already handles.

        Unlike that pattern, a header-row match is exactly the signal this method looks
        for (not something to exclude) — the label being a column header is the whole
        point.
        """
        ordered_lines = sorted(line_bbox_map.items(), key=lambda item: item[1][1])

        for position, (line_id, _bbox) in enumerate(ordered_lines):
            tokens = line_text_map.get(line_id, [])
            token_bboxes = line_token_bboxes.get(line_id, [])
            if not tokens or len(tokens) != len(token_bboxes):
                continue

            line_text = " ".join(tokens)
            match = None
            for pattern in keyword_patterns:
                match = re.search(pattern, line_text, re.IGNORECASE)
                if match:
                    break
            if not match:
                continue

            # Map the match's character span back to the token(s) it covers, by
            # walking the same join() logic used to build line_text, then union their
            # bboxes to get the label's own horizontal footprint.
            matched_boxes = []
            cursor = 0
            for token_text, token_bbox in zip(tokens, token_bboxes):
                token_start = cursor
                token_end = cursor + len(token_text)
                if token_start < match.end() and token_end > match.start():
                    matched_boxes.append(token_bbox)
                cursor = token_end + 1  # +1 for the joining space

            if not matched_boxes:
                continue

            label_x1 = min(box[0] for box in matched_boxes)
            label_x2 = max(box[2] for box in matched_boxes)
            label_center_x = (label_x1 + label_x2) / 2
            label_width = label_x2 - label_x1

            if position + 1 >= len(ordered_lines):
                continue
            value_line_id, _ = ordered_lines[position + 1]
            value_tokens = line_text_map.get(value_line_id, [])
            value_bboxes = line_token_bboxes.get(value_line_id, [])
            if not value_tokens or len(value_tokens) != len(value_bboxes):
                continue

            # The line below must actually look like a row of values, not an unrelated
            # paragraph that merely happens to sit right underneath the header.
            numeric_count = sum(1 for text in value_tokens if self.NUMERIC_PATTERN.search(text))
            if numeric_count < max(1, len(value_tokens) / 2):
                continue

            max_gap = max(label_width, self.TABLE_COLUMN_MIN_GAP_PX) * self.TABLE_COLUMN_MAX_GAP_RATIO
            best_candidate = None
            best_distance = None
            for value_text, value_bbox in zip(value_tokens, value_bboxes):
                value_center_x = (value_bbox[0] + value_bbox[2]) / 2
                distance = abs(value_center_x - label_center_x)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_candidate = value_text

            if best_candidate is None or best_distance > max_gap:
                continue

            parsed = self.parse_amount(best_candidate)
            if parsed is not None:
                return parsed, best_candidate.strip()

        return None, None

    def _split_into_column_groups(
        self, bboxes: list[tuple[int, int, int, int]], num_groups: int
    ) -> list[list[int]] | None:
        """
        Segments a header line's tokens (by index) into `num_groups` left-to-right
        column groups by cutting at the largest horizontal gaps between consecutive
        tokens — a multi-word column header ("R. con Aportes") has small gaps between
        its own words and a comparatively large gap before the next column's header
        starts. Returns None if there aren't enough tokens to form that many groups.
        """
        if num_groups <= 0 or len(bboxes) < num_groups:
            return None
        if num_groups == 1:
            return [list(range(len(bboxes)))]

        order = sorted(range(len(bboxes)), key=lambda i: bboxes[i][0])
        gaps = [
            (bboxes[order[k]][0] - bboxes[order[k - 1]][2], k)
            for k in range(1, len(order))
        ]
        gaps.sort(key=lambda item: item[0], reverse=True)
        cut_positions = sorted(k for _, k in gaps[: num_groups - 1])

        groups = []
        start = 0
        for cut in cut_positions:
            groups.append([order[i] for i in range(start, cut)])
            start = cut
        groups.append([order[i] for i in range(start, len(order))])
        return groups

    # A trailing numeric token anchored to the END of a line's joined text — tolerant of
    # EasyOCR occasionally splitting a decimal separator into its own token ("506029 .
    # 27" reads as three tokens joined with spaces, which this still matches as one
    # token since it operates on the joined line text, not per-token). The leading
    # `(?<![0-9])` is load-bearing: without it, this would happily match just the last
    # 1-3 digits of a much longer, separator-less number (a CUIL/legajo/beneficio
    # number) and mistake that suffix for a standalone trailing amount. A bare space is
    # only allowed immediately around a "." or "," separator — NOT between two whole
    # digit runs with no separator at all — so two unrelated numbers that happen to
    # land on the same (mis-clustered, densely packed) line never get fused into one
    # inflated amount (e.g. "19032 34,783.45" must NOT read as "1903234783.45").
    TRAILING_NUMBER_TOKEN_PATTERN = re.compile(
        r"(?<![0-9])-?\(?\$?\s*[0-9]+(?:\s?[.,]\s?[0-9]+)*\)?\s*$"
    )
    LINE_ITEM_MIN_LABEL_CHARS = 3

    def _is_plausible_amount_token(self, raw: str) -> bool:
        """
        The lookbehind above stops TRAILING_NUMBER_TOKEN_PATTERN from slicing a suffix
        off a longer ID, but a short bare number with no decimal (a stray reference
        digit, or the whole of a short code) is still not reliably a monetary amount.
        A genuine decimal fraction (",XX"/".XX") is trusted regardless of length; a
        bare integer with no decimal shown is only trusted if it's long enough that a
        real amount is more likely than a stray code (these payroll concepts don't
        realistically list amounts under 1000 pesos).
        """
        stripped = raw.strip()
        digits_only = re.sub(r"[^0-9]", "", stripped)
        if not digits_only:
            return False
        if re.search(r"[\.,]\s?[0-9]{1,2}\)?\s*$", stripped):
            return True
        return len(digits_only) >= 4

    def extract_line_items(
        self,
        line_text_map: dict,
        line_bbox_map: dict,
        line_token_bboxes: dict,
    ) -> list[dict]:
        """
        Generic concept:amount extraction, independent of the fixed
        RECEIPT_VALUE_PATTERNS keyword list — captures every line-item concept a
        receipt lists (e.g. "301 ASIGNACION POR CARGO: 506029.27"), not just the
        handful of known summary totals extracted by analyze_receipt_consistency.

        Two layouts are handled, mirroring the same two cases as the fixed-field
        extraction above:
        - Case A: the concept and its amount sit on the SAME line, amount trailing.
        - Case B: a header line of 2+ concept labels, immediately followed by a line
          of 2+ values in the same left-to-right order (see _extract_table_column_value
          for the single-keyword version of this same 2-line table layout).
        """
        ordered_lines = sorted(line_bbox_map.items(), key=lambda item: item[1][1])
        consumed_value_lines = set()
        items = []

        for position, (line_id, _bbox) in enumerate(ordered_lines):
            if line_id in consumed_value_lines:
                continue

            tokens = line_text_map.get(line_id, [])
            bboxes = line_token_bboxes.get(line_id, [])
            if not tokens or len(tokens) != len(bboxes):
                continue

            line_text = " ".join(tokens)
            lowered = line_text.lower()

            if self.IDENTITY_DASH_PATTERN.search(line_text):
                continue

            if any(keyword in lowered for keyword in self.LINE_ITEM_EXCLUDE_KEYWORDS):
                # This header/label is identity or reference data (DNI, CUIL, período,
                # legajo, beneficio number...), already covered by a dedicated
                # extractor. If the line right below has no concept text of its own
                # (a bare value entirely dependent on THIS label for meaning, e.g. a
                # lone "06/2026" under a "Período" header), consume it too so it's
                # never independently re-visited as an orphan Case A candidate with no
                # real label — but leave it alone if it has its own alphabetic text,
                # since that means it's a self-sufficient concept+amount row in its
                # own right (e.g. "HABER MENSUAL $552.322,59" below a "Concepto"
                # header row).
                if position + 1 < len(ordered_lines):
                    next_line_id, _ = ordered_lines[position + 1]
                    next_text = " ".join(line_text_map.get(next_line_id, []))
                    if next_text and not any(char.isalpha() for char in next_text):
                        consumed_value_lines.add(next_line_id)
                continue

            # Case A: same-line "concepto ... monto".
            amount_match = self.TRAILING_NUMBER_TOKEN_PATTERN.search(line_text)
            # A trailing number directly after a "/" is the year of a DD/MM/YYYY date
            # ("FECHA PROX. COBRO DESDE: 13/08/2026"), not a monetary amount.
            is_date_tail = bool(amount_match) and line_text[:amount_match.start()].rstrip().endswith("/")
            if (
                amount_match and not is_date_tail
                and amount_match.group(0).strip()
                and self._is_plausible_amount_token(amount_match.group(0))
            ):
                label_text = line_text[:amount_match.start()].strip(" .:-")
                parsed = self.parse_amount(amount_match.group(0))
                if parsed is not None and len(label_text) >= self.LINE_ITEM_MIN_LABEL_CHARS:
                    # Map the match span back to the token(s) it covers, for a bbox.
                    matched_boxes = []
                    cursor = 0
                    for token_text, token_bbox in zip(tokens, bboxes):
                        token_start = cursor
                        token_end = cursor + len(token_text)
                        if token_start < amount_match.end() and token_end > amount_match.start():
                            matched_boxes.append(token_bbox)
                        cursor = token_end + 1
                    item_bbox = (
                        min(b[0] for b in matched_boxes), min(b[1] for b in matched_boxes),
                        max(b[2] for b in matched_boxes), max(b[3] for b in matched_boxes),
                    ) if matched_boxes else None
                    items.append({
                        "concepto": label_text,
                        "monto": parsed,
                        "raw": amount_match.group(0).strip(),
                        "bbox": item_bbox,
                    })
                    continue

            # Case B: 2-line table row — this line has no amount of its own; check if
            # it's a header for a values row immediately below.
            if not any(char.isalpha() for char in line_text):
                continue
            if position + 1 >= len(ordered_lines):
                continue

            value_line_id, _ = ordered_lines[position + 1]
            value_tokens = line_text_map.get(value_line_id, [])
            value_bboxes = line_token_bboxes.get(value_line_id, [])
            if not value_tokens or len(value_tokens) != len(value_bboxes):
                continue

            numeric_count = sum(1 for text in value_tokens if self.NUMERIC_PATTERN.search(text))
            if numeric_count < max(1, len(value_tokens) / 2) or len(value_tokens) < 2:
                continue

            label_groups = self._split_into_column_groups(bboxes, len(value_tokens))
            if label_groups is None:
                continue

            matched_any = False
            for group in label_groups:
                group_tokens = [tokens[i] for i in group]
                group_bboxes = [bboxes[i] for i in group]
                label_text = " ".join(group_tokens).strip(" .:-")
                if len(label_text) < self.LINE_ITEM_MIN_LABEL_CHARS:
                    continue

                group_x1 = min(b[0] for b in group_bboxes)
                group_x2 = max(b[2] for b in group_bboxes)
                group_center_x = (group_x1 + group_x2) / 2
                max_gap = max(group_x2 - group_x1, self.TABLE_COLUMN_MIN_GAP_PX) * self.TABLE_COLUMN_MAX_GAP_RATIO

                best_value, best_bbox, best_distance = None, None, None
                for value_text, value_bbox in zip(value_tokens, value_bboxes):
                    value_center_x = (value_bbox[0] + value_bbox[2]) / 2
                    distance = abs(value_center_x - group_center_x)
                    if best_distance is None or distance < best_distance:
                        best_distance, best_value, best_bbox = distance, value_text, value_bbox

                if best_value is None or best_distance > max_gap:
                    continue
                parsed = self.parse_amount(best_value)
                if parsed is not None:
                    items.append({
                        "concepto": label_text,
                        "monto": parsed,
                        "raw": best_value.strip(),
                        "bbox": best_bbox,
                    })
                    matched_any = True

            if matched_any:
                consumed_value_lines.add(value_line_id)

        return items

    RETRY_TRAILING_WIDTH_RATIO = 0.15

    # Each variant tried in order: (upscale_factor, use_binarization). Different
    # scale/binarization assumptions recover different failure modes — a value that
    # fails ALL of them is a stronger "genuinely unreadable" tell than one that only
    # got a single, fixed attempt. (PSM variation doesn't apply to EasyOCR.)
    RETRY_VARIANTS = (
        (3, False),  # moderate zoom, contrast-only (original attempt)
        (5, True),  # more aggressive zoom + hard black/white binarization
    )

    def _binarize(self, crop: Image.Image) -> Image.Image:
        threshold = float(np.asarray(crop, dtype=np.float64).mean())
        return crop.point(lambda pixel: 255 if pixel > threshold else 0)

    def _retry_read_line_amount(
        self, image: Image.Image, keyword_patterns: list[str], line_text_map: dict, line_bbox_map: dict,
    ) -> tuple[float | None, str | None, int]:
        """
        Second attempt(s) at reading a value whose label was found but whose amount
        wasn't: crops just that label's own line at full original resolution (not the
        possibly-downscaled copy the main OCR pass ran on) and re-reads it in
        isolation, tried across a few different zoom/binarization/segmentation
        variants — a field that failed once amid ~100+ other words on the full page
        has a real shot at being read correctly in isolation, at higher effective
        resolution, and different variants recover different failure modes.
        Returns (value, raw_text, attempts_tried) — `attempts_tried` lets the caller
        report how many variants were exhausted when all of them still failed.
        """
        for line_id, tokens in line_text_map.items():
            line_text = " ".join(tokens)
            if not any(re.search(pattern, line_text, re.IGNORECASE) for pattern in keyword_patterns):
                continue

            bbox = line_bbox_map.get(line_id)
            if not bbox:
                continue

            x1, y1, x2, y2 = bbox
            # Extend rightward past the last detected token — the value itself may have
            # produced no token at all, so the line's own bbox wouldn't include it.
            x2 = min(image.width, x2 + int(image.width * self.RETRY_TRAILING_WIDTH_RATIO))
            x1, y1 = max(0, x1 - 4), max(0, y1 - 4)
            y2 = min(image.height, y2 + 4)
            if x2 <= x1 or y2 <= y1:
                continue

            base_crop = image.crop((x1, y1, x2, y2)).convert("L")
            attempts_tried = 0

            for scale, use_binarization in self.RETRY_VARIANTS:
                attempts_tried += 1
                crop = base_crop.resize((base_crop.width * scale, base_crop.height * scale), Image.LANCZOS)
                crop = ImageOps.autocontrast(crop)
                if use_binarization:
                    crop = self._binarize(crop)
                else:
                    crop = crop.filter(ImageFilter.SHARPEN)

                try:
                    retry_lines = self.reader.readtext(np.array(crop.convert("RGB")), detail=0, paragraph=True)
                    retry_text = " ".join(retry_lines)
                except Exception as e:
                    logger.error(f"Error during retry OCR read (variant {attempts_tried}): {str(e)}")
                    continue

                amount_match = re.search(r"\(?\$?\s*([0-9]{1,3}(?:\.?[0-9]{3})*,[0-9]{2})\)?", retry_text)
                if amount_match:
                    raw_amount = amount_match.group(0)
                    parsed = self.parse_amount(raw_amount)
                    if parsed is not None:
                        return parsed, raw_amount.strip(), attempts_tried

            return None, None, attempts_tried

        return None, None, 0

    # Above this document-wide mean OCR confidence, a single field still failing to
    # extract is surprising enough to treat as "something specific about this field",
    # not "this photo is generally hard to read".
    LOCALIZED_FAILURE_MIN_CONFIDENCE = 65.0

    def analyze_receipt_consistency(
        self,
        text_blob: str,
        mean_confidence: float = 0.0,
        image: Image.Image | None = None,
        line_text_map: dict | None = None,
        line_bbox_map: dict | None = None,
        line_token_bboxes: dict | None = None,
    ) -> dict:
        text_lower = text_blob.lower()
        if not any(hint in text_lower for hint in self.RECEIPT_LABEL_HINTS):
            return {
                "status": "success",
                "detected": False,
                "consistency_score": 0,
                "signals": [],
                "fields": {},
                "localized_unreadable_fields": [],
            }

        fields = {}
        signals = []
        score = 0
        localized_unreadable_fields = []

        for field_name, patterns in self.RECEIPT_VALUE_PATTERNS.items():
            value, raw_value, keyword_found = self.extract_amount_after_keywords(text_blob, patterns)

            # Purely arithmetic (no OCR call) — tried before the image-based retry
            # below so a value resolvable from a 2-line summary table skips the cost
            # of re-running EasyOCR on a crop entirely.
            if value is None and keyword_found and line_text_map and line_bbox_map and line_token_bboxes:
                table_value, table_raw = self._extract_table_column_value(
                    patterns, line_text_map, line_bbox_map, line_token_bboxes
                )
                if table_value is not None:
                    value, raw_value = table_value, f"{table_raw} (recuperado de tabla de 2 líneas)"

            retry_attempts_tried = 0
            if value is None and keyword_found and image is not None and line_text_map and line_bbox_map:
                retry_value, retry_raw, retry_attempts_tried = self._retry_read_line_amount(
                    image, patterns, line_text_map, line_bbox_map
                )
                if retry_value is not None:
                    value, raw_value = retry_value, f"{retry_raw} (recuperado en 2do intento de lectura)"

            if value is not None:
                fields[field_name] = {"value": value, "raw": raw_value}
            elif keyword_found:
                display_name = self.FIELD_DISPLAY_NAMES.get(field_name, field_name)
                # Surviving every re-read variant (different zoom, binarization, and
                # segmentation mode) unreadable is a stronger tell than a single failed
                # attempt would be — worth saying explicitly for the human reviewer.
                retry_note = (
                    f" Se probaron {retry_attempts_tried} variantes de relectura (zoom, binarización, "
                    "segmentación) y ninguna lo recuperó."
                    if retry_attempts_tried
                    else ""
                )
                if mean_confidence >= self.LOCALIZED_FAILURE_MIN_CONFIDENCE:
                    # The rest of the document reads fine, so THIS field's failure isn't
                    # explained by general photo/OCR quality — a strong tell that it uses
                    # a font Tesseract's model doesn't recognize, which is exactly what a
                    # deliberately substituted value would look like. Weighted higher
                    # than a generic "OCR struggled" miss, and tracked separately so the
                    # decision engine can treat it as standalone strong evidence.
                    score += 2
                    localized_unreadable_fields.append(field_name)
                    signals.append(
                        f"Se encontró la etiqueta '{display_name}' pero no se pudo leer su monto, pese a que el "
                        f"resto del documento se lee con buena confianza ({mean_confidence:.1f}%) — posible fuente "
                        f"sustituida deliberadamente, revisar manualmente.{retry_note}"
                    )
                else:
                    # The label itself was found, but no valid amount could be read near
                    # it, and the whole document reads poorly — most likely generic photo
                    # quality (blur, low light), not a targeted edit. Silence would still
                    # hide a field that needs a human to read manually, so it's surfaced,
                    # just with less weight than the localized case above.
                    score += 1
                    signals.append(
                        f"Se encontró la etiqueta '{display_name}' pero no se pudo leer su monto — "
                        f"revisar manualmente.{retry_note}"
                    )

        haberes = fields.get("total_haberes", {}).get("value")
        descuentos = fields.get("total_descuentos", {}).get("value")
        neto = fields.get("neto", {}).get("value")
        neto_a_cobrar = fields.get("neto_a_cobrar", {}).get("value")
        liquido = fields.get("liquido", {}).get("value")

        computed_expected = None
        if haberes is not None and descuentos is not None:
            computed_expected = haberes - descuentos
            fields["expected_neto"] = {"value": computed_expected, "raw": f"{haberes} - {descuentos}"}

        def evaluate_difference(label: str, expected: float | None, observed: float | None, tolerance_ratio: float = 0.015, min_tolerance: float = 1.0):
            nonlocal score
            if expected is None or observed is None:
                return

            # An order-of-magnitude gap almost always means one side lost a decimal
            # separator during OCR extraction, not a genuine arithmetic inconsistency —
            # asserting "inconsistente" here would report an OCR bug as if it were
            # evidence of fraud. Defense-in-depth alongside the stricter amount regex.
            larger, smaller = max(abs(expected), abs(observed)), min(abs(expected), abs(observed))
            if smaller > 0 and larger / smaller > 10:
                signals.append(f"{label}: no se pudo comparar de forma confiable (posible error de lectura de OCR).")
                return

            tolerance = max(min_tolerance, abs(expected) * tolerance_ratio)
            delta = abs(expected - observed)
            fields.setdefault("differences", {})[label] = {
                "expected": expected,
                "observed": observed,
                "delta": delta,
                "tolerance": tolerance,
            }
            if delta <= tolerance:
                signals.append(f"{label} consistente (delta {delta:.2f}).")
            elif delta <= tolerance * 3:
                score += 1
                signals.append(f"{label} con diferencia leve (delta {delta:.2f}).")
            else:
                score += 2
                signals.append(f"{label} inconsistente (delta {delta:.2f}).")

        evaluate_difference("neto_vs_haberes_menos_descuentos", computed_expected, neto)
        evaluate_difference("neto_vs_neto_a_cobrar", neto, neto_a_cobrar)
        evaluate_difference("neto_vs_liquido", neto, liquido)

        if len(fields) >= 3:
            score += 1
            signals.append("Se detectaron suficientes montos clave para validar el recibo.")

        period_info = self.analyze_period_consistency(text_blob)
        if period_info["periods_found"]:
            fields["periods_found"] = period_info["periods_found"]
        signals.extend(period_info["signals"])
        score += period_info["consistency_score"]

        if score == 0 and fields:
            signals.append("Montos clave coherentes en la muestra OCR.")

        return {
            "status": "success",
            "detected": bool(fields),
            "consistency_score": min(score, 4),
            "signals": signals,
            "fields": fields,
            "localized_unreadable_fields": localized_unreadable_fields,
        }

    def _normalize_period(self, match: re.Match) -> tuple[int, int] | None:
        if match.group("mm") and match.group("yyyy"):
            return (int(match.group("mm")), int(match.group("yyyy")))

        if match.group("yy2") and match.group("mm2"):
            yy = int(match.group("yy2"))
            year = 2000 + yy if yy < 70 else 1900 + yy
            return (int(match.group("mm2")), year)

        if match.group("month_name"):
            month = self.MONTH_NAMES.get(match.group("month_name").lower())
            year_raw = match.group("year_after_name")
            if month and year_raw:
                return (month, int(year_raw))

        return None

    def analyze_period_consistency(self, text_blob: str) -> dict:
        """
        Cross-checks every period/month field mentioned in the document (e.g. "Periodo
        Liquidado" vs "Periodo Abonado" vs a "Sueldo Mes X" header). A legitimate receipt
        should reference a single period throughout; disagreement is a strong sign that
        an old receipt was reused and only the period field was edited.
        """
        periods = []
        for keyword_match in self.PERIOD_KEYWORD_PATTERN.finditer(text_blob):
            # "Desde"/"Hasta" mark a validity WINDOW (e.g. "Fecha Próximo Cobro Desde/Hasta"),
            # not a restated period — its "hasta" end legitimately falls in the next month
            # by design, so it must never be compared for equality against "Periodo
            # Liquidado"/"Periodo Abonado" style fields.
            context = text_blob[max(0, keyword_match.start() - 25):keyword_match.end() + 10]
            if self.RANGE_BOUNDARY_PATTERN.search(context):
                continue

            window = text_blob[keyword_match.end():keyword_match.end() + 40]
            value_match = self.PERIOD_VALUE_PATTERN.search(window)
            if not value_match:
                continue

            normalized = self._normalize_period(value_match)
            if normalized is None:
                continue

            periods.append({
                "label": keyword_match.group(0).strip(),
                "raw": value_match.group(0).strip(),
                "normalized": normalized,
            })

        signals = []
        score = 0
        distinct_periods = {period["normalized"] for period in periods}

        if len(distinct_periods) >= 2:
            score = 2
            first, second = periods[0], next(p for p in periods if p["normalized"] != first["normalized"])
            signals.append(
                f"Inconsistencia de período: '{first['label']} {first['raw']}' vs "
                f"'{second['label']} {second['raw']}'."
            )
        elif len(distinct_periods) == 1 and len(periods) >= 2:
            signals.append("Períodos consistentes entre los campos detectados.")

        return {
            "status": "success",
            "detected": bool(periods),
            "periods_found": periods,
            "consistency_score": score,
            "signals": signals,
        }