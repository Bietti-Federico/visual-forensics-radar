import logging
import os
import re
from dataclasses import dataclass

import pytesseract
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
    """

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
        Returns the preprocessed image plus the inverse scale factor needed to map
        coordinates from the preprocessed (possibly downscaled) image back to the
        original image's coordinate space.
        """
        image = image.convert("L")
        max_width = 1600
        inverse_scale = 1.0
        if image.width > max_width:
            ratio = max_width / image.width
            inverse_scale = image.width / max_width
            image = image.resize((max_width, int(image.height * ratio)))
        image = ImageOps.autocontrast(image)
        image = image.filter(ImageFilter.SHARPEN)
        return image, inverse_scale

    def _extract_tokens_from_data(self, data: dict, inverse_scale: float, pass_name: str) -> list[dict]:
        tokens = []
        total_items = len(data.get("text", []))
        for idx in range(total_items):
            raw_text = str(data["text"][idx]).strip()
            conf_raw = str(data.get("conf", ["-1"])[idx]).strip()
            try:
                confidence = float(conf_raw)
            except Exception:
                confidence = -1.0

            if not raw_text or confidence < 0:
                continue

            x = int(data["left"][idx])
            y = int(data["top"][idx])
            w = int(data["width"][idx])
            h = int(data["height"][idx])
            # bbox is rescaled back to the original image's coordinate space,
            # since Tesseract ran on a possibly downscaled copy.
            bbox = (
                int(x * inverse_scale),
                int(y * inverse_scale),
                int((x + w) * inverse_scale),
                int((y + h) * inverse_scale),
            )

            block_num = int(data.get("block_num", [0])[idx])
            par_num = int(data.get("par_num", [0])[idx])
            line_num = int(data.get("line_num", [0])[idx])

            tokens.append({
                "text": raw_text,
                "confidence": confidence,
                "bbox": bbox,
                "line_id": (pass_name, block_num, par_num, line_num),
            })

        return tokens

    def _bbox_overlap_ratio(self, bbox_a: tuple, bbox_b: tuple) -> float:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        return inter_area / area_a

    def _merge_ocr_passes(self, primary_tokens: list[dict], secondary_tokens: list[dict]) -> list[dict]:
        """
        PSM 6 (the primary pass) assumes a single uniform block of text, which can make
        Tesseract drop text sitting inside a bordered/boxed cell entirely — seen in
        practice as a boxed total silently missing while everything around it read
        fine. PSM 11 ("sparse text, no particular order") has no such layout
        assumption, so it runs as a second, purely ADDITIVE pass: anything it finds
        that the primary pass already covers (by bbox overlap) is discarded, so this
        can only add tokens the primary pass missed, never duplicate or replace one.
        """
        merged = list(primary_tokens)
        for token in secondary_tokens:
            overlaps_existing = any(
                self._bbox_overlap_ratio(token["bbox"], existing["bbox"]) > 0.3
                for existing in primary_tokens
            )
            if not overlaps_existing:
                merged.append(token)
        return merged

    def analyze(self, image_path: str) -> dict:
        try:
            image = Image.open(image_path)
            processed, inverse_scale = self._preprocess(image)

            primary_data = pytesseract.image_to_data(
                processed,
                output_type=pytesseract.Output.DICT,
                config="--oem 1 --psm 6",
                lang="spa+eng",
            )
            secondary_data = pytesseract.image_to_data(
                processed,
                output_type=pytesseract.Output.DICT,
                config="--oem 1 --psm 11",
                lang="spa+eng",
            )

            primary_tokens = self._extract_tokens_from_data(primary_data, inverse_scale, "primary")
            secondary_tokens = self._extract_tokens_from_data(secondary_data, inverse_scale, "secondary")
            combined_tokens = self._merge_ocr_passes(primary_tokens, secondary_tokens)

            words: list[OcrWord] = []
            candidate_regions = []
            keyword_hits = []
            numeric_tokens = 0
            confidence_values = []
            token_entries = []
            line_text_map = {}

            for token in combined_tokens:
                raw_text = token["text"]
                confidence = token["confidence"]
                bbox = token["bbox"]
                line_id = token["line_id"]

                words.append(OcrWord(text=raw_text, confidence=confidence, bbox=bbox))
                confidence_values.append(confidence)

                line_text_map.setdefault(line_id, []).append(raw_text)

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
            receipt_consistency = self.analyze_receipt_consistency(text_blob, mean_confidence)

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
                    if numeric_tokens >= 5:
                        score += 1
                        signals.append("OCR detectó suficiente estructura numérica para validar el documento.")
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
                "extracted_fields": {
                    "dni_tramite": dni_tramite,
                    "dni_number": dni_number,
                    "bank_name": bank_name,
                    "card_numbers": card_numbers,
                    "masked_card_numbers": masked_card_numbers,
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

    def extract_amount_after_keywords(self, text_blob: str, keyword_patterns: list[str]) -> tuple[float | None, str | None, bool]:
        """
        Returns (value, raw_text, keyword_found). `keyword_found` is True whenever the
        label itself (e.g. "NETO") was located in the text, even if no valid amount
        could be read nearby — distinguishing "this document doesn't have this field"
        from "the label is there but its value is unreadable", which callers should
        treat as a red flag in its own right (e.g. an unusual/edited font that OCR
        can't parse), not silence.
        """
        keyword_found = False
        for keyword_pattern in keyword_patterns:
            match = re.search(keyword_pattern, text_blob, re.IGNORECASE)
            if not match:
                continue
            keyword_found = True

            start = match.end()
            window = text_blob[start:start + 120]
            # Every real amount on these receipts is formatted with a mandatory 2-digit
            # decimal (",XX") — requiring it here rejects a bare stray digit ("3") or a
            # comma-stripped OCR misread ("53558199" instead of "535581,99") that would
            # otherwise silently corrupt the arithmetic consistency check with a bogus
            # number. Thousands "." separators are optional since OCR often drops them
            # while keeping the decimal comma intact.
            amount_match = re.search(r"\(?\$?\s*([0-9]{1,3}(?:\.?[0-9]{3})*,[0-9]{2})\)?", window)
            if amount_match:
                raw_amount = amount_match.group(0)
                parsed = self.parse_amount(raw_amount)
                if parsed is not None:
                    return parsed, raw_amount.strip(), True

        return None, None, keyword_found

    # Above this document-wide mean OCR confidence, a single field still failing to
    # extract is surprising enough to treat as "something specific about this field",
    # not "this photo is generally hard to read".
    LOCALIZED_FAILURE_MIN_CONFIDENCE = 65.0

    def analyze_receipt_consistency(self, text_blob: str, mean_confidence: float = 0.0) -> dict:
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
            if value is not None:
                fields[field_name] = {"value": value, "raw": raw_value}
            elif keyword_found:
                display_name = self.FIELD_DISPLAY_NAMES.get(field_name, field_name)
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
                        "sustituida deliberadamente, revisar manualmente."
                    )
                else:
                    # The label itself was found, but no valid amount could be read near
                    # it, and the whole document reads poorly — most likely generic photo
                    # quality (blur, low light), not a targeted edit. Silence would still
                    # hide a field that needs a human to read manually, so it's surfaced,
                    # just with less weight than the localized case above.
                    score += 1
                    signals.append(
                        f"Se encontró la etiqueta '{display_name}' pero no se pudo leer su monto — revisar manualmente."
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