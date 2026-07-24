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
        "acreditado",
    )

    def _preprocess(self, image: Image.Image) -> Image.Image:
        image = image.convert("L")
        max_width = 1600
        if image.width > max_width:
            ratio = max_width / image.width
            image = image.resize((max_width, int(image.height * ratio)))
        image = ImageOps.autocontrast(image)
        image = image.filter(ImageFilter.SHARPEN)
        return image

    def analyze(self, image_path: str) -> dict:
        try:
            image = Image.open(image_path)
            processed = self._preprocess(image)

            data = pytesseract.image_to_data(
                processed,
                output_type=pytesseract.Output.DICT,
                config="--oem 1 --psm 6",
                lang="spa+eng",
            )

            words: list[OcrWord] = []
            candidate_regions = []
            keyword_hits = []
            numeric_tokens = 0
            confidence_values = []
            token_entries = []
            line_text_map = {}

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
                bbox = (x, y, x + w, y + h)

                words.append(OcrWord(text=raw_text, confidence=confidence, bbox=bbox))
                confidence_values.append(confidence)

                block_num = int(data.get("block_num", [0])[idx])
                par_num = int(data.get("par_num", [0])[idx])
                line_num = int(data.get("line_num", [0])[idx])
                line_id = (block_num, par_num, line_num)

                line_text_map.setdefault(line_id, []).append(raw_text)

                normalized = raw_text.lower()
                if any(keyword in normalized for keyword in self.CRITICAL_KEYWORDS):
                    keyword_hits.append(raw_text)

                is_numeric = bool(self.NUMERIC_PATTERN.search(raw_text))
                is_date = bool(self.DATE_PATTERN.search(raw_text))
                is_amount = bool(self.AMOUNT_PATTERN.search(raw_text))

                if is_numeric:
                    numeric_tokens += 1
                elif any(keyword in normalized for keyword in self.CRITICAL_KEYWORDS):
                    pass

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

                priority = 0
                region_type = "context"
                is_key_field = False

                if entry["is_numeric"] and (entry["is_amount"] or entry["is_date"]) and has_key_hint:
                    priority = 4
                    region_type = "key_numeric"
                    is_key_field = True
                elif entry["is_numeric"] and has_key_hint:
                    priority = 3
                    region_type = "line_numeric"
                    is_key_field = True
                elif entry["is_numeric"] and (entry["is_amount"] or entry["is_date"]):
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
            receipt_consistency = self.analyze_receipt_consistency(text_blob)

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
                "candidate_regions": candidate_regions[:24],
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

    def extract_amount_after_keywords(self, text_blob: str, keyword_patterns: list[str]) -> tuple[float | None, str | None]:
        for keyword_pattern in keyword_patterns:
            match = re.search(keyword_pattern, text_blob, re.IGNORECASE)
            if not match:
                continue

            start = match.end()
            window = text_blob[start:start + 120]
            amount_match = re.search(r"\(?\$?\s*([0-9][0-9\.,\s]{0,30}[0-9](?:[\.,][0-9]{1,2})?|[0-9]+(?:[\.,][0-9]{1,2})?)\)?", window)
            if amount_match:
                raw_amount = amount_match.group(0)
                parsed = self.parse_amount(raw_amount)
                if parsed is not None:
                    return parsed, raw_amount.strip()

        return None, None

    def analyze_receipt_consistency(self, text_blob: str) -> dict:
        text_lower = text_blob.lower()
        if not any(hint in text_lower for hint in self.RECEIPT_LABEL_HINTS):
            return {
                "status": "success",
                "detected": False,
                "consistency_score": 0,
                "signals": [],
                "fields": {},
            }

        fields = {}
        signals = []
        score = 0

        for field_name, patterns in self.RECEIPT_VALUE_PATTERNS.items():
            value, raw_value = self.extract_amount_after_keywords(text_blob, patterns)
            if value is not None:
                fields[field_name] = {"value": value, "raw": raw_value}

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

        if score == 0 and fields:
            signals.append("Montos clave coherentes en la muestra OCR.")

        return {
            "status": "success",
            "detected": bool(fields),
            "consistency_score": min(score, 4),
            "signals": signals,
            "fields": fields,
        }