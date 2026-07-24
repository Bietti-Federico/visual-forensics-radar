import os
import tempfile
import logging
from typing import Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from backend.ela_detector import ElaDetector
from backend.clip_detector import ClipAuthenticator
from backend.metadata_detector import MetadataDetector
from backend.ocr_detector import OCRDetector

logging.basicConfig(level=logging.INFO, format ='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DeepFakeAPI")

MAX_BATCH_SIZE = 4
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


def _score_clip(ai_prob: float) -> float:
    # CLIP is kept as a very soft supporting signal for receipts only.
    # We cap its influence to avoid document-type false positives dominating the score.
    return _clamp(max(0.0, ai_prob - 80.0) * 0.9)


def _score_metadata(metadata_score: int) -> float:
    return _clamp((metadata_score / 4.0) * 100.0)


def _score_ocr(ocr_score: int) -> float:
    return _clamp((ocr_score / 4.0) * 100.0)


def _score_receipt_consistency(consistency_score: int) -> float:
    return _clamp((consistency_score / 4.0) * 100.0)


def _score_global_ela(max_diff: int, anomaly_detected: bool, threshold_used: int = 35) -> float:
    if anomaly_detected:
        return _clamp(55.0 + max(0, max_diff - threshold_used) * 1.5)
    if max_diff >= max(18, int(threshold_used * 0.7)):
        return _clamp(25.0 + (max_diff / max(1, threshold_used)) * 20.0)
    return 0.0


def _score_local_ela(local_ela_results: list[dict[str, Any]]) -> float:
    if not local_ela_results:
        return 0.0

    weighted_region_scores = []
    for region in local_ela_results:
        region_score = 0.0
        max_diff = region.get("max_difference", 0)
        key_multiplier = 1.45 if region.get("is_key_field") else 1.0
        threshold_used = region.get("threshold_used", 20)

        if region.get("anomaly_detected"):
            base = 78.0 + max(0, max_diff - threshold_used) * 1.5
            region_score = _clamp(base * key_multiplier)
        elif max_diff >= 18:
            base = 42.0 + (max_diff - 18) * 1.7
            region_score = _clamp(base * key_multiplier)
        elif region.get("is_key_field") and max_diff >= 12:
            base = 20.0 + (max_diff - 12) * 2.0
            region_score = _clamp(base * key_multiplier)
        elif max_diff >= 10:
            base = 12.0 + (max_diff - 10) * 1.5
            region_score = _clamp(base * key_multiplier)

        if region_score > 0:
            weighted_region_scores.append(region_score)

    if not weighted_region_scores:
        return 0.0

    weighted_region_scores.sort(reverse=True)
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


def decision_engine(ela_result: dict, clip_result: dict, metadata_result: dict, ocr_result: dict, local_ela_results: list[dict]) -> dict:
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
    clip_probs = clip_result.get("detailed_probabilities",{})
    ai_prob = clip_probs.get("ai_probability",0)

    clip_score = _score_clip(ai_prob)
    global_ela_score = _score_global_ela(max_diff, ela_anomaly, threshold_used=threshold_used)
    metadata_norm = _score_metadata(metadata_score)
    ocr_norm = _score_ocr(ocr_score)
    receipt_consistency_norm = _score_receipt_consistency(receipt_consistency_score)
    local_ela_norm = _score_local_ela(local_ela_results)

    final_score = _clamp(
        (0.60 * local_ela_norm)
        + (0.20 * receipt_consistency_norm)
        + (0.08 * ocr_norm)
        + (0.06 * global_ela_score)
        + (0.04 * metadata_norm)
        + (0.02 * clip_score)
    )

    band_key, band_label = _normalize_band(final_score)

    clip_probs = clip_result.get("detailed_probabilities",{})
    decision = {
        "score": round(final_score, 2),
        "risk_band": band_key,
        "risk_label": band_label,
        "component_scores": {
            "local_ela": round(local_ela_norm, 2),
            "ocr": round(ocr_norm, 2),
            "receipt_consistency": round(receipt_consistency_norm, 2),
            "global_ela": round(global_ela_score, 2),
            "metadata": round(metadata_norm, 2),
            "clip": round(clip_score, 2),
        },
        "evidence": [],
        "metadata": metadata_result,
        "ocr": ocr_result,
        "local_ela_regions": local_ela_results,
    }

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


def build_receipt_response(file_name: str, type_result: dict, ela_res: dict, clip_res: dict, metadata_res: dict, ocr_res: dict, local_ela_regions: list[dict]) -> dict:
    final_decision = decision_engine(ela_res, clip_res, metadata_res, ocr_res, local_ela_regions)
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
            "clip_layer": clip_res,
            "metadata_layer": metadata_res,
            "ocr_layer": ocr_res,
            "ocr_local_ela_regions": local_ela_regions,
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

    logger.info("Step 2: Running ELA (Pixel) Analysis...")
    ela_res = models["ela"].analyze(temp_file_path)

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
        local_threshold = 16 if candidate.get("is_key_field") else 20
        local_ela = models["ela"].analyze_crop(temp_file_path, padded_bbox, anomaly_threshold=local_threshold)
        local_ela["text"] = candidate.get("text", "")
        local_ela["region_type"] = candidate.get("type", "unknown")
        local_ela["is_key_field"] = bool(candidate.get("is_key_field", False))
        local_ela["priority"] = candidate.get("priority", 0)
        local_ela["line_text"] = candidate.get("line_text", "")
        local_ela_regions.append(local_ela)

    clip_res = models["clip"].analyze(temp_file_path)
    return build_receipt_response(file_name, type_result, ela_res, clip_res, metadata_res, ocr_res, local_ela_regions)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    The main endpoint that receives the image from the frontend and orchestrates the 3 engines.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400,detail="Please upload a valid image file.")

    required_models = ("ela", "clip", "metadata", "ocr")
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


@app.post("/analyze-batch")
async def analyze_batch(files: list[UploadFile] = File(...)):
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Please upload at least one image file.")
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch too large. Maximum {MAX_BATCH_SIZE} images per request.",
        )

    required_models = ("ela", "clip", "metadata", "ocr")
    missing_models = [name for name in required_models if name not in models]
    if missing_models:
        raise HTTPException(
            status_code=503,
            detail=f"Models not initialized: {', '.join(missing_models)}"
        )

    results = []
    temp_paths = []

    try:
        for file in files:
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail=f"Invalid file type for {file.filename}")

            temp_dir = tempfile.gettempdir()
            file_suffix = os.path.splitext(file.filename)[1] or ".img"
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix, dir=temp_dir)
            temp_file_path = temp_file.name
            temp_file.close()
            temp_paths.append((temp_file_path, file.filename))

            with open(temp_file_path, "wb") as buffer:
                buffer.write(await file.read())

        for temp_file_path, file_name in temp_paths:
            results.append(analyze_file_path(temp_file_path, file_name))

        return JSONResponse(content={
            "status": "success",
            "batch_size": len(results),
            "results": results,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during batch analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"System Error: {str(e)}")
    finally:
        for temp_file_path, file_name in temp_paths:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.info(f"Temporary file deleted securely: {file_name}")