import os
import tempfile
import logging
import statistics
from typing import Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from PIL import Image

from backend.ela_detector import ElaDetector
from backend.clip_detector import ClipAuthenticator
from backend.metadata_detector import MetadataDetector
from backend.ocr_detector import OCRDetector
from backend.typography_detector import TypographyDetector

logging.basicConfig(level=logging.INFO, format ='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DeepFakeAPI")

RECEIPT_ROUTES = {"receipt", "unknown"}
IDENTITY_ROUTES = {"dni_front", "dni_back"}
CARD_ROUTES = {"card"}
CATEGORY_ONLY_ROUTES = IDENTITY_ROUTES | CARD_ROUTES | {"homebanking"}

# Global model dictionary (Ensures a single instance in VRAM)
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    SYSTEM ARCHITECTURE: Lifespan Management
    Loading models on every API request would instantly cause a VRAM overflow (OOM).
    Therefore, models are loaded into RAM/VRAM ONLY ONCE when the API starts.
    """

    logger.info("Loading AI Engines into VRAM... Please stand by.")
    try:
        models["ela"] = ElaDetector()
        models["clip"] = ClipAuthenticator()
        models["metadata"] = MetadataDetector()
        models["ocr"] = OCRDetector()
        models["typography"] = TypographyDetector()
        logger.info("Validation engines successfully initialized. API is ready to serve!")

    except Exception as e:
        logger.exception("Critical error while loading models")
        raise RuntimeError(f"Model initialization failed: {e}") from e

    yield
    
    logger.info("API is shutting down, clearing VRAM...")
    models.clear()

app = FastAPI(title="Deepfake Detection core API", version="1.0",lifespan=lifespan)


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _normalize_band(score: float) -> tuple[str, str]:
    if score >= 60:
        return "altamente_sospechoso", "Altamente sospechoso"
    if score >= 25:
        return "sospechoso", "Sospechoso"
    return "poco_sospechoso", "Poco sospechoso"


def _score_metadata(metadata_score: int) -> float:
    return _clamp((metadata_score / 4.0) * 100.0)


def _score_ocr(ocr_score: int) -> float:
    return _clamp((ocr_score / 4.0) * 100.0)


def _score_receipt_consistency(consistency_score: int) -> float:
    return _clamp((consistency_score / 4.0) * 100.0)


def _score_typography(typography_result: dict) -> float:
    anomalous_fields = typography_result.get("anomalous_fields", [])
    if not anomalous_fields:
        return 0.0

    key_field_hits = sum(1 for field in anomalous_fields if field.get("is_key_field"))
    other_hits = len(anomalous_fields) - key_field_hits
    return _clamp(key_field_hits * 35.0 + other_hits * 15.0)


def _score_global_ela(max_diff: int, anomaly_detected: bool, threshold_used: int = 35) -> float:
    if anomaly_detected:
        return _clamp(55.0 + max(0, max_diff - threshold_used) * 1.5)
    if max_diff >= max(18, int(threshold_used * 0.7)):
        return _clamp(25.0 + (max_diff / max(1, threshold_used)) * 20.0)
    return 0.0


def _score_region_ela(region: dict) -> float:
    """
    Real photos of printed documents routinely show ELA max_difference in the low-to-mid
    teens purely from JPEG recompression noise around high-contrast text — that is NOT
    evidence of tampering. Only `anomaly_detected` (max_diff strictly above the threshold)
    is treated as meaningful signal; values approaching the threshold get a small, fast-decaying
    credit so a handful of ordinary-noise regions can't stack up into a high aggregate score.
    """
    max_diff = region.get("max_difference", 0)
    key_multiplier = 1.45 if region.get("is_key_field") else 1.0
    threshold_used = region.get("threshold_used", 20)

    if region.get("anomaly_detected"):
        base = 78.0 + max(0, max_diff - threshold_used) * 1.5
        return _clamp(base * key_multiplier)

    if max_diff > threshold_used:
        # Cleared the absolute threshold but was downgraded by the relative, document-wide
        # calibration (not an outlier vs. its own document's peers) — still worth a modest,
        # CAPPED amount of attention. Without this branch the margin-based formula below
        # would keep growing the further max_diff sits above threshold_used, which defeats
        # the point of downgrading it in the first place.
        return _clamp(30.0 * key_multiplier)

    margin_below_threshold = threshold_used - max_diff
    if margin_below_threshold <= 4:
        base = 10.0 + max(0, 4 - margin_below_threshold) * 2.0
        return _clamp(base * key_multiplier)

    return 0.0


MIN_LOCAL_ELA_SAMPLES = 6
RELATIVE_ELA_Z_THRESHOLD = 2.5


def _leave_one_out_z_score(values: list[float], index: int, mad_floor: float = 1.0) -> float:
    """
    Modified (Iglewicz-Hoaglin) z-score of values[index] against the median/MAD of every
    OTHER value in the list, so a field is never compared against a population that
    includes itself. Mirrors the same robust-stats approach used in typography_detector.py.

    ELA's max_difference is a small integer (typically 10-20 for ordinary noise), so it's
    common for more than half the fields in a document to land on the exact same value —
    that alone makes MAD collapse to 0, which would otherwise make every z-score 0.0
    (even for a field that's genuinely a bit off from the rest). `mad_floor` sets the
    smallest deviation still treated as meaningful, so a real difference is measured
    even when the bulk of the document ties on one number.
    """
    others = values[:index] + values[index + 1:]
    if len(others) < 2:
        return 0.0

    median = statistics.median(others)
    mad = max(statistics.median([abs(v - median) for v in others]), mad_floor)

    return abs(0.6745 * (values[index] - median) / mad)


def _apply_relative_ela_calibration(local_ela_results: list[dict]) -> None:
    """
    A physical fold/crease, uneven lighting, or paper texture raises ELA's max_difference
    fairly uniformly across a WHOLE photographed document — a fixed absolute threshold
    alone can't tell that apart from a genuinely edited field. When there are enough
    comparable fields in the same document, a field is only kept as `anomaly_detected`
    if it's ALSO a statistical outlier relative to the rest of THIS document's own
    fields; a document-wide artifact that pushes every field past the absolute threshold
    together no longer trips a false alarm, since none of them stand out from their peers.
    With too few fields to trust a relative baseline, the absolute-threshold judgment
    from ElaDetector is left untouched.
    """
    if len(local_ela_results) < MIN_LOCAL_ELA_SAMPLES:
        for region in local_ela_results:
            region["relative_z_score"] = None
        return

    diffs = [region.get("max_difference", 0) for region in local_ela_results]
    for idx, region in enumerate(local_ela_results):
        z = _leave_one_out_z_score(diffs, idx)
        region["relative_z_score"] = round(z, 2)
        if region.get("anomaly_detected") and z <= RELATIVE_ELA_Z_THRESHOLD:
            region["anomaly_detected"] = False


def _ranked_local_ela_regions(local_ela_results: list[dict[str, Any]]) -> list[tuple[float, dict]]:
    """
    Scores every region the same way `_score_local_ela` does internally, but keeps the
    region alongside its score so callers (e.g. the primary-reason explainer) can point
    at the exact region driving the score instead of re-deriving the logic and getting
    out of sync with it — a region can score well above zero without `anomaly_detected`
    being True (see the `max_diff >= 18` branch), so filtering on that flag alone misses it.
    """
    scored = [(_score_region_ela(region), region) for region in local_ela_results]
    scored = [(score, region) for score, region in scored if score > 0]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored


def _top_local_ela_region(local_ela_results: list[dict[str, Any]]) -> dict | None:
    ranked = _ranked_local_ela_regions(local_ela_results)
    return ranked[0][1] if ranked else None


def _score_local_ela(local_ela_results: list[dict[str, Any]]) -> float:
    if not local_ela_results:
        return 0.0

    weighted_region_scores = [score for score, _ in _ranked_local_ela_regions(local_ela_results)]
    if not weighted_region_scores:
        return 0.0

    top = weighted_region_scores[:4]
    aggregate = top[0]
    if len(top) > 1:
        aggregate += top[1] * 0.7
    if len(top) > 2:
        aggregate += top[2] * 0.5
    if len(top) > 3:
        aggregate += top[3] * 0.3

    return _clamp(aggregate)

def _pad_bbox(bbox: tuple[int, int, int, int], image_width: int, image_height: int, padding: int = 12) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(image_width, x2 + padding),
        min(image_height, y2 + padding),
    )


TYPOGRAPHY_FEATURE_LABELS = {
    "height": "altura de letra",
    "ink_ratio": "densidad de tinta",
    "slant_angle": "inclinación/caligrafía",
    "aspect_ratio": "proporción de letra",
}


def _describe_typography_field(field: dict) -> str:
    feature_label = TYPOGRAPHY_FEATURE_LABELS.get(field.get("dominant_feature"), "tipografía")
    return f"{feature_label} distinta en '{field.get('text')}' (z-score {field.get('max_abs_z')})"


BBOX_MATCH_TOLERANCE_PX = 6.0


def _bbox_center(bbox: tuple[float, float, float, float] | None) -> tuple[float, float] | None:
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _same_field_location(bbox_a: tuple | None, bbox_b: tuple | None, tolerance: float = BBOX_MATCH_TOLERANCE_PX) -> bool:
    center_a, center_b = _bbox_center(bbox_a), _bbox_center(bbox_b)
    if center_a is None or center_b is None:
        return False
    return abs(center_a[0] - center_b[0]) <= tolerance and abs(center_a[1] - center_b[1]) <= tolerance


def _find_corroborated_field(local_ela_results: list[dict], typography_result: dict) -> dict | None:
    """
    Cross-references ELA-local anomalies with typography anomalies by field POSITION
    (bbox center), not text. Matching by text alone would let two different physical
    occurrences of the same value (e.g. a duplicated two-column ticket) cross-match
    each other and falsely count as "corroborated", even though neither field was
    independently confirmed by both methods. A key field flagged by BOTH methods AT
    THE SAME LOCATION is a much stronger signal of an edited value than either one
    alone, and is the basis for the score floor and headline alert below.
    """
    typography_key_anomalies = [
        field for field in typography_result.get("anomalous_fields", [])
        if field.get("is_key_field")
    ]
    if not typography_key_anomalies:
        return None

    for region in local_ela_results:
        if not (region.get("is_key_field") and region.get("anomaly_detected")):
            continue
        for field in typography_key_anomalies:
            if _same_field_location(region.get("bbox"), field.get("bbox")):
                return {"text": region.get("text"), "ela_region": region, "typography_field": field}

    return None


STRONG_TYPOGRAPHY_Z_THRESHOLD = 8.0


def _strongest_typography_key_field(typography_result: dict) -> dict | None:
    """
    A fraud pattern based purely on font/handwriting substitution (a receipt reprinted
    or hand-completed with a different tool) leaves NO compression trace for ELA to
    corroborate — requiring both signals to agree would systematically miss exactly
    that pattern. A typography anomaly on a key field with a z-score far beyond the
    3.5 flagging threshold (this checks >=8, more than double it) is treated as strong
    standalone evidence in its own right, without needing ELA agreement.
    """
    key_hits = [
        field for field in typography_result.get("anomalous_fields", [])
        if field.get("is_key_field") and field.get("max_abs_z", 0) >= STRONG_TYPOGRAPHY_Z_THRESHOLD
    ]
    if not key_hits:
        return None
    return max(key_hits, key=lambda field: field.get("max_abs_z", 0))


def decision_engine(ela_result: dict, metadata_result: dict, ocr_result: dict, local_ela_results: list[dict], typography_result: dict) -> dict:
    """
    Fast triage engine for receipt validation.
    Combines light-weight visual and semantic signals into three human-review levels.
    """

    ela_anomaly = ela_result.get("anomaly_detected",False)
    max_diff = ela_result.get("max_difference", 0)
    threshold_used = ela_result.get("threshold_used", 35)
    metadata_score = metadata_result.get("metadata_score", 0)
    metadata_signals = metadata_result.get("suspicious_signals", [])
    ocr_score = ocr_result.get("ocr_score", 0)
    ocr_signals = ocr_result.get("suspicious_signals", [])
    ocr_mean_conf = ocr_result.get("mean_confidence", 0)
    receipt_consistency = ocr_result.get("receipt_consistency", {})
    receipt_consistency_score = receipt_consistency.get("consistency_score", 0)
    receipt_consistency_signals = receipt_consistency.get("signals", [])

    global_ela_score = _score_global_ela(max_diff, ela_anomaly, threshold_used=threshold_used)
    metadata_norm = _score_metadata(metadata_score)
    ocr_norm = _score_ocr(ocr_score)
    receipt_consistency_norm = _score_receipt_consistency(receipt_consistency_score)
    local_ela_norm = _score_local_ela(local_ela_results)
    typography_norm = _score_typography(typography_result)

    weighted_components = {
        "local_ela": (0.57, local_ela_norm),
        "typography": (0.10, typography_norm),
        "receipt_consistency": (0.20, receipt_consistency_norm),
        "ocr": (0.08, ocr_norm),
        "global_ela": (0.03, global_ela_score),
        "metadata": (0.02, metadata_norm),
    }

    final_score = _clamp(sum(weight * value for weight, value in weighted_components.values()))

    corroborated_field = _find_corroborated_field(local_ela_results, typography_result)
    if corroborated_field is not None:
        final_score = max(final_score, 75.0)

    strong_typography_field = _strongest_typography_key_field(typography_result)
    if strong_typography_field is not None:
        final_score = max(final_score, 60.0)

    band_key, band_label = _normalize_band(final_score)

    decision = {
        "score": round(final_score, 2),
        "risk_band": band_key,
        "risk_label": band_label,
        "component_scores": {
            "local_ela": round(local_ela_norm, 2),
            "typography": round(typography_norm, 2),
            "ocr": round(ocr_norm, 2),
            "receipt_consistency": round(receipt_consistency_norm, 2),
            "global_ela": round(global_ela_score, 2),
            "metadata": round(metadata_norm, 2),
        },
        "evidence": [],
        "primary_reason": None,
        "metadata": metadata_result,
        "ocr": ocr_result,
        "local_ela_regions": local_ela_results,
        "typography": typography_result,
    }

    corroboration_message = None
    if corroborated_field is not None:
        corroboration_message = (
            f"Posible edición confirmada en campo clave: '{corroborated_field['text']}' "
            "(ELA local + tipografía coinciden)."
        )
        decision["evidence"].append(corroboration_message)

    strong_typography_message = None
    if strong_typography_field is not None:
        strong_typography_message = (
            f"Tipografía marcadamente distinta en campo clave: {_describe_typography_field(strong_typography_field)}. "
            "Patrón típico de sustitución de fuente/caligrafía sin rastro de compresión JPEG que ELA pueda corroborar."
        )
        decision["evidence"].append(strong_typography_message)

    if final_score >= 60:
        decision["evidence"].append("Revisión prioritaria recomendada.")
    elif final_score >= 25:
        decision["evidence"].append("Revisión humana sugerida.")
    else:
        decision["evidence"].append("Sin señales fuertes de fraude.")

    if local_ela_norm >= 70:
        decision["evidence"].append("Anomalía localizada fuerte en texto o números críticos.")
    elif local_ela_norm >= 40:
        decision["evidence"].append("Variación localizada compatible con edición.")

    key_region_alerts = 0
    for region in local_ela_results:
        if region.get("is_key_field") and region.get("anomaly_detected"):
            key_region_alerts += 1

    if key_region_alerts >= 2:
        decision["evidence"].append("Múltiples anomalías ELA en campos clave (fechas/montos).")
    elif key_region_alerts == 1:
        decision["evidence"].append("Anomalía ELA detectada en un campo clave (fecha o monto).")

    typography_key_hits = [
        field for field in typography_result.get("anomalous_fields", []) if field.get("is_key_field")
    ]
    if typography_key_hits:
        top_field = typography_key_hits[0]
        decision["evidence"].append(f"{_describe_typography_field(top_field).capitalize()} en campo clave.")

    if ocr_mean_conf and ocr_mean_conf < 45:
        decision["evidence"].append(f"OCR con confianza media baja ({ocr_mean_conf:.1f}%).")

    if receipt_consistency_score >= 3:
        decision["evidence"].append("Inconsistencia aritmética en montos del recibo.")
    elif receipt_consistency_score >= 1:
        decision["evidence"].append("Posible inconsistencia leve entre montos del recibo.")

    for signal in metadata_signals[:2]:
        if signal not in decision["evidence"]:
            decision["evidence"].append(signal)

    for signal in ocr_signals[:2]:
        if signal not in decision["evidence"]:
            decision["evidence"].append(signal)

    for signal in receipt_consistency_signals[:2]:
        if signal not in decision["evidence"]:
            decision["evidence"].append(signal)

    if corroboration_message is not None:
        decision["primary_reason"] = corroboration_message
    elif strong_typography_message is not None:
        decision["primary_reason"] = strong_typography_message
    else:
        contributions = {name: weight * value for name, (weight, value) in weighted_components.items()}
        winning_component = max(contributions, key=contributions.get)

        if contributions[winning_component] <= 0.5:
            decision["primary_reason"] = "Sin señales de peso relevante detectadas."
        elif winning_component == "local_ela":
            top_region = _top_local_ela_region(local_ela_results)
            field_text = (top_region or {}).get("text") or "sin identificar"
            decision["primary_reason"] = f"Motivo principal: anomalía ELA en el campo '{field_text}' (mayor peso en el score)."
        elif winning_component == "typography":
            top_field = typography_key_hits[0] if typography_key_hits else (typography_result.get("anomalous_fields") or [{}])[0]
            decision["primary_reason"] = f"Motivo principal: {_describe_typography_field(top_field)} (mayor peso en el score)."
        elif winning_component == "receipt_consistency":
            top_signal = receipt_consistency_signals[0] if receipt_consistency_signals else "inconsistencia en montos del recibo"
            decision["primary_reason"] = f"Motivo principal: {top_signal} (mayor peso en el score)."
        elif winning_component == "ocr":
            top_signal = ocr_signals[0] if ocr_signals else "baja calidad de lectura OCR"
            decision["primary_reason"] = f"Motivo principal: {top_signal} (mayor peso en el score)."
        elif winning_component == "metadata":
            top_signal = metadata_signals[0] if metadata_signals else "señales de metadatos"
            decision["primary_reason"] = f"Motivo principal: {top_signal} (mayor peso en el score)."
        else:
            decision["primary_reason"] = "Motivo principal: anomalía de compresión JPEG generalizada en la imagen (mayor peso en el score)."

    return decision


def build_document_type_response(file_name: str, type_result: dict, route: str, ocr_result: dict, extracted_fields: dict) -> dict:
    document_type = type_result.get("document_type", "unknown")
    confidence = type_result.get("document_type_confidence", 0)

    return {
        "status": "success",
        "final_decision": {
            "score": None,
            "risk_band": "not_applicable",
            "risk_label": "Sin control de fraude",
            "component_scores": {},
            "evidence": ["Este tipo de documento se analiza solo por extracción de información."],
        },
        "analysis_route": route,
        "document_type": document_type,
        "document_type_confidence": confidence,
        "diagnostics": {
            "file_name": file_name,
            "document_type_classification": type_result,
            "extracted_fields": extracted_fields,
            "validation_mode": "classification-plus-extraction",
        },
    }


def build_category_only_response(file_name: str, type_result: dict, ocr_result: dict) -> dict:
    return {
        "status": "success",
        "analysis_route": "category_only",
        "document_type": type_result.get("document_type", "unknown"),
        "document_type_confidence": type_result.get("document_type_confidence", 0),
        "final_decision": {
            "score": None,
            "risk_band": "not_applicable",
            "risk_label": "Solo categorización",
            "component_scores": {},
            "evidence": ["Este tipo de documento no recibe control de fraude."],
        },
        "diagnostics": {
            "file_name": file_name,
            "document_type_classification": type_result,
            "validation_mode": "category-only",
        },
    }


def build_receipt_response(file_name: str, type_result: dict, ela_res: dict, metadata_res: dict, ocr_res: dict, local_ela_regions: list[dict], typography_res: dict) -> dict:
    final_decision = decision_engine(ela_res, metadata_res, ocr_res, local_ela_regions, typography_res)
    return {
        "status": "success",
        "analysis_route": "receipt_control",
        "document_type": type_result.get("document_type", "unknown"),
        "document_type_confidence": type_result.get("document_type_confidence", 0),
        "final_decision": final_decision,
        "diagnostics": {
            "file_name": file_name,
            "document_type_classification": type_result,
            "ela_layer": ela_res,
            "metadata_layer": metadata_res,
            "ocr_layer": ocr_res,
            "ocr_local_ela_regions": local_ela_regions,
            "typography_layer": typography_res,
            "validation_mode": "receipt-control",
        },
    }


def analyze_file_path(temp_file_path: str, file_name: str) -> dict:
    logger.info(f"new analysis request received. Processing file: {file_name}")

    logger.info("Step 1: Classifying document type with CLIP...")
    type_result = models["clip"].classify_document_type(temp_file_path)
    if type_result.get("status") != "success":
        raise RuntimeError(type_result.get("message", "Document type classification failed"))

    document_type = type_result.get("document_type", "unknown")
    confidence = type_result.get("document_type_confidence", 0)

    logger.info(f"Document type detected: {document_type} ({confidence}%)")

    if document_type in IDENTITY_ROUTES and confidence >= 45:
        return build_category_only_response(file_name, type_result, {})

    if document_type in CARD_ROUTES and confidence >= 45:
        return build_category_only_response(file_name, type_result, {})

    if document_type in {"homebanking"} and confidence >= 45:
        return build_category_only_response(file_name, type_result, {})

    logger.info("Route selected: receipt control")

    # Decoded once and reused for the global ELA pass, every per-field ELA crop, and
    # typography analysis, instead of each one re-reading and re-decoding the
    # full-resolution photo from disk on its own.
    shared_image = Image.open(temp_file_path).convert("RGB")

    logger.info("Step 2: Running ELA (Pixel) Analysis...")
    ela_res = models["ela"].analyze(shared_image)

    logger.info("Step 3: Running Metadata Analysis...")
    metadata_res = models["metadata"].analyze(temp_file_path)

    logger.info("Step 4: Running OCR Analysis...")
    ocr_res = models["ocr"].analyze(temp_file_path)

    local_ela_regions = []
    candidate_regions = ocr_res.get("candidate_regions", [])
    image_width = metadata_res.get("width", 0)
    image_height = metadata_res.get("height", 0)

    for candidate in candidate_regions[:12]:
        bbox = candidate.get("bbox")
        if not bbox:
            continue
        padded_bbox = _pad_bbox(tuple(bbox), image_width, image_height, padding=14)
        # Photographed paper receipts (camera, not scanner) carry enough natural JPEG/print
        # noise that a stricter threshold on key fields was producing false positives;
        # 20 applies uniformly regardless of field type.
        local_threshold = 20
        local_ela = models["ela"].analyze_crop(shared_image, padded_bbox, anomaly_threshold=local_threshold)
        local_ela["text"] = candidate.get("text", "")
        local_ela["region_type"] = candidate.get("type", "unknown")
        local_ela["is_key_field"] = bool(candidate.get("is_key_field", False))
        local_ela["priority"] = candidate.get("priority", 0)
        local_ela["line_text"] = candidate.get("line_text", "")
        # Original (unpadded) candidate bbox, kept alongside the padded "crop_box" ELA
        # itself already returns — used to match this field against typography's
        # anomalies by position rather than by text.
        local_ela["bbox"] = bbox
        local_ela_regions.append(local_ela)

    _apply_relative_ela_calibration(local_ela_regions)

    if candidate_regions:
        logger.info("Step 5: Running Typography Consistency Analysis...")
        typography_res = models["typography"].analyze(shared_image, candidate_regions)
    else:
        typography_res = {"status": "success", "buckets": {}, "anomalous_fields": []}

    return build_receipt_response(file_name, type_result, ela_res, metadata_res, ocr_res, local_ela_regions, typography_res)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    The main endpoint that receives the image from the frontend and orchestrates the 3 engines.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400,detail="Please upload a valid image file.")

    required_models = ("ela", "clip", "metadata", "ocr", "typography")
    missing_models = [name for name in required_models if name not in models]
    if missing_models:
        raise HTTPException(
            status_code=503,
            detail=f"Models not initialized: {', '.join(missing_models)}"
        )
    
    temp_dir = tempfile.gettempdir()
    file_suffix = os.path.splitext(file.filename)[1] or ".img"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix, dir=temp_dir)
    temp_file_path = temp_file.name
    temp_file.close()

    try:
        with open(temp_file_path,"wb") as buffer:
            buffer.write(await file.read())
        return JSONResponse(content=analyze_file_path(temp_file_path, file.filename))
    
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"System Error: {str(e)}")
    
    # Garbage collection
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Temporary file deleted securely: {file.filename}")