"""FastAPI app: document verification + minimal training-corpus admin.

Kept as a separate top-level package from `pdf_forensics` itself — the core
library stays free of web-framework concerns, the same reasoning `scripts/`
is a thin caller of the library rather than part of it.

The trained model bundle is loaded once at process startup (`lifespan`,
below) and held in `app.state`; `/retrain` reloads it in place afterward so
`/verify` reflects the new model immediately, with no process restart.
`/verify` never writes anything to disk — the uploaded bytes are processed
entirely in memory and discarded once the response is built.

Basic file logging (`logging_config.py`) is configured once at startup —
every endpoint logs its outcome, and any unhandled exception is caught by
`handle_unexpected_error` below and logged with its full traceback before
returning a generic 500, so a production error is always in the log even
if nothing else surfaces it.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pdf_forensics.application.document_scoring.score_document_use_case import (
    ScoreDocumentUseCase,
)
from pdf_forensics.application.model_persistence.load_trained_models_use_case import (
    LoadTrainedModelsUseCase,
)
from pdf_forensics.application.model_training.retrain_models_use_case import (
    RetrainModelsUseCase,
)
from pdf_forensics.application.pdf_analysis.parse_pdf_use_case import ParsePdfUseCase
from pdf_forensics.domain.pdf.errors import NotAPdfError, UnrecoverableStructureError
from pdf_forensics.infrastructure.training_corpus.filesystem_training_corpus_reader import (
    read_training_corpus_files,
)
from pdf_forensics_api.config import get_model_store_path, get_training_corpus_dir
from pdf_forensics_api.logging_config import configure_logging
from pdf_forensics_api.serialization import serialize_scoring_result

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"
_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
#: Defensive upper bound on an uploaded file's size — this is a document
#: verification API, not a general file store.
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def _sanitize_path_component(value: str, *, default: str = "unnamed") -> str:
    """Strips any directory components and anything but a conservative
    filename charset — both `entity` and the uploaded filename end up as
    path segments under `training_corpus/`, so neither may be trusted as-is
    (path traversal via `entity="../../etc"` or a crafted filename)."""
    candidate = Path(value).name
    candidate = _UNSAFE_PATH_CHARS.sub("_", candidate).strip("._")
    return candidate or default


def _reload_model_state(app: FastAPI) -> None:
    model_store_path = get_model_store_path()
    if model_store_path.is_file():
        app.state.entity_classifiers, app.state.per_entity = LoadTrainedModelsUseCase().execute(
            model_store_path
        )
        logger.info(
            "Modelo cargado desde %s (%d entidades)",
            model_store_path,
            len(app.state.per_entity),
        )
    else:
        app.state.entity_classifiers, app.state.per_entity = (), {}
        logger.info(
            "No hay modelo entrenado todavía en %s; arrancando sin modelo.", model_store_path
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Configured here, not at module import time, so tests can point
    # PDF_FORENSICS_LOG_PATH at a tmp_path before this runs — matching how
    # config.py's paths are read fresh rather than cached at import time.
    configure_logging()
    _reload_model_state(app)
    yield


app = FastAPI(title="PDF Forensics Verification API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    # `exc_info=exc` (no `logger.exception`) porque no estamos necesariamente
    # dentro de un bloque `except` desde el punto de vista de este handler —
    # pasar la excepción explícita es lo que garantiza el traceback en el log.
    logger.error("Error no controlado en %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > _MAX_UPLOAD_BYTES:
        logger.warning("Archivo %r rechazado por tamaño: %d bytes", file.filename, len(data))
        raise HTTPException(status_code=413, detail="File too large.")
    return data


def _parse_or_422(pdf_bytes: bytes, *, filename: str | None) -> None:
    """Fails fast on non-PDF uploads for the training endpoints, so the
    corpus never accumulates files `RetrainModelsUseCase` would just skip
    later anyway — surfacing the problem at upload time is more useful."""
    try:
        ParsePdfUseCase().execute(pdf_bytes)
    except (NotAPdfError, UnrecoverableStructureError) as exc:
        logger.warning("Archivo %r no es un PDF legible: %s", filename, exc)
        raise HTTPException(status_code=422, detail=f"Not a readable PDF: {exc}") from exc


@app.post("/verify")
async def verify(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
    entity: str | None = Form(None),
) -> dict[str, Any]:
    """`entity`, if given, overrides the classifier's own guess entirely —
    the entity identification model sometimes gets it wrong, and there's
    no reason to force a retrain just to re-score one document against
    the entity a human can already tell it actually is."""
    pdf_bytes = await _read_upload(file)
    use_case = ScoreDocumentUseCase(
        request.app.state.entity_classifiers, request.app.state.per_entity
    )
    try:
        result = use_case.execute(pdf_bytes, entity_override=entity)
    except (NotAPdfError, UnrecoverableStructureError) as exc:
        logger.warning("Verificación de %r falló: no es un PDF legible: %s", file.filename, exc)
        raise HTTPException(status_code=422, detail=f"Not a readable PDF: {exc}") from exc
    logger.info(
        "Verificado %r: riesgo=%d entidad=%s%s",
        file.filename,
        result.risk_report.risk_score,
        (
            result.entity_report.predictions[0].predicted_entity
            if result.entity_report.predictions
            else "desconocida"
        ),
        " (manual)" if entity else "",
    )
    return serialize_scoring_result(result)


#: Above this, a training upload is suggested as confirmed-fraud rather
#: than genuine — matches the frontend's own "risk-high" cutoff, so the
#: suggestion agrees with the color the user would see verifying the same
#: file. A suggestion only, never binding — see `suggest_entity` below.
_SUGGEST_FRAUD_RISK_THRESHOLD = 50


@app.post("/training-data/suggest-entity")
async def suggest_entity(
    request: Request, file: UploadFile = File(...)  # noqa: B008
) -> dict[str, Any]:
    """Neither the entity nor genuine/confirmed-fraud is meant to require
    someone to already know the right answer before uploading. Runs the
    full scoring pipeline (same as `/verify`) against an uploaded file and
    returns its best guess at both, for the frontend to pre-fill the
    training-upload form — still editable either way, since a genuinely
    new entity has no classifier to guess it from, and a risk-score-based
    genuine/fraud guess is exactly that: a guess, not a verdict."""
    pdf_bytes = await _read_upload(file)
    use_case = ScoreDocumentUseCase(
        request.app.state.entity_classifiers, request.app.state.per_entity
    )
    try:
        result = use_case.execute(pdf_bytes)
    except (NotAPdfError, UnrecoverableStructureError) as exc:
        raise HTTPException(status_code=422, detail=f"Not a readable PDF: {exc}") from exc

    predictions = result.entity_report.predictions
    top = predictions[0] if predictions else None
    risk_score = result.risk_report.risk_score
    return {
        "suggested_entity": top.predicted_entity if top else None,
        "confidence": top.confidence if top else None,
        "risk_score": risk_score,
        "suggested_genuine": risk_score < _SUGGEST_FRAUD_RISK_THRESHOLD,
    }


def _training_file_dir(entity: str, *, is_genuine: bool) -> Path:
    safe_entity = _sanitize_path_component(entity, default="UNKNOWN_ENTITY")
    subdir = "genuine" if is_genuine else "confirmed_fraud"
    return get_training_corpus_dir() / subdir / safe_entity


def _save_training_file(pdf_bytes: bytes, filename: str, entity: str, *, is_genuine: bool) -> Path:
    safe_filename = _sanitize_path_component(filename, default="upload.pdf")
    if not safe_filename.lower().endswith(".pdf"):
        safe_filename += ".pdf"

    target_dir = _training_file_dir(entity, is_genuine=is_genuine)
    target_dir.mkdir(parents=True, exist_ok=True)

    target_path = target_dir / safe_filename
    # Exclusive create, not an exists()-then-write_bytes() check: two
    # concurrent uploads sanitizing to the same filename could otherwise
    # both pass the check and one silently overwrite the other.
    while True:
        try:
            with target_path.open("xb") as target_file:
                target_file.write(pdf_bytes)
            return target_path
        except FileExistsError:
            target_path = (
                target_dir / f"{target_path.stem}-{uuid.uuid4().hex[:8]}{target_path.suffix}"
            )


@app.post("/training-data/genuine")
async def upload_genuine(
    entity: str = Form(...), file: UploadFile = File(...)  # noqa: B008
) -> dict[str, Any]:
    pdf_bytes = await _read_upload(file)
    _parse_or_422(pdf_bytes, filename=file.filename)
    path = _save_training_file(pdf_bytes, file.filename or "upload.pdf", entity, is_genuine=True)
    logger.info("Documento genuino agregado: entidad=%s archivo=%s", entity, path)
    return {"saved_to": str(path)}


@app.post("/training-data/confirmed-fraud")
async def upload_confirmed_fraud(
    entity: str = Form(...), file: UploadFile = File(...)  # noqa: B008
) -> dict[str, Any]:
    pdf_bytes = await _read_upload(file)
    _parse_or_422(pdf_bytes, filename=file.filename)
    path = _save_training_file(pdf_bytes, file.filename or "upload.pdf", entity, is_genuine=False)
    logger.info("Fraude confirmado agregado: entidad=%s archivo=%s", entity, path)
    return {"saved_to": str(path)}


@app.get("/training-data/summary")
def training_data_summary() -> dict[str, Any]:
    refs = read_training_corpus_files(get_training_corpus_dir())
    entities: dict[str, dict[str, int]] = {}
    for ref in refs:
        counts = entities.setdefault(ref.entity, {"genuine": 0, "confirmed_fraud": 0})
        counts["genuine" if ref.is_genuine else "confirmed_fraud"] += 1
    return {"entities": entities}


@app.get("/training-data/files")
def training_data_files() -> dict[str, Any]:
    """Per-file listing (not just aggregate counts) so the frontend can
    show, and let someone remove, individual training documents."""
    refs = read_training_corpus_files(get_training_corpus_dir())
    return {
        "files": [
            {"entity": ref.entity, "is_genuine": ref.is_genuine, "filename": ref.path.name}
            for ref in refs
        ]
    }


def _delete_training_file(entity: str, filename: str, *, is_genuine: bool) -> None:
    safe_filename = _sanitize_path_component(filename, default="upload.pdf")
    target_path = _training_file_dir(entity, is_genuine=is_genuine) / safe_filename
    if not target_path.is_file():
        logger.warning("Intento de borrar archivo inexistente: %s", target_path)
        raise HTTPException(status_code=404, detail="File not found.")
    target_path.unlink()
    logger.info("Archivo de entrenamiento eliminado: %s", target_path)


@app.delete("/training-data/genuine/{entity}/{filename}")
def delete_genuine(entity: str, filename: str) -> dict[str, Any]:
    _delete_training_file(entity, filename, is_genuine=True)
    return {"deleted": True}


@app.delete("/training-data/confirmed-fraud/{entity}/{filename}")
def delete_confirmed_fraud(entity: str, filename: str) -> dict[str, Any]:
    _delete_training_file(entity, filename, is_genuine=False)
    return {"deleted": True}


@app.post("/training-data/recategorize")
def recategorize_training_file(
    entity: str = Form(...),
    filename: str = Form(...),
    is_genuine: bool = Form(...),
    new_entity: str = Form(...),
    new_is_genuine: bool = Form(...),
) -> dict[str, Any]:
    """Moves an already-uploaded file to a different entity and/or
    genuine/confirmed-fraud bucket — for correcting a suggestion (entity
    or nature) after the fact, without deleting and re-uploading the file
    by hand. Doesn't retrain — same as upload/delete, a stale model still
    reflects the old categorization until the next `/retrain`."""
    safe_filename = _sanitize_path_component(filename, default="upload.pdf")
    old_path = _training_file_dir(entity, is_genuine=is_genuine) / safe_filename
    if not old_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    prospective_path = _training_file_dir(new_entity, is_genuine=new_is_genuine) / safe_filename
    if prospective_path == old_path:
        # No actual change (e.g. saving the edit form without touching
        # anything) — leave the file exactly as it is. Otherwise
        # save-then-delete would see its own about-to-be-deleted file as
        # a name collision and rename the file with a random suffix.
        return {"saved_to": str(old_path)}

    pdf_bytes = old_path.read_bytes()
    old_path.unlink()
    new_path = _save_training_file(pdf_bytes, filename, new_entity, is_genuine=new_is_genuine)
    logger.info("Recategorizado: %s -> %s", old_path, new_path)
    return {"saved_to": str(new_path)}


@app.post("/retrain")
def retrain(request: Request) -> dict[str, Any]:
    logger.info("Reentrenamiento iniciado.")
    summary = RetrainModelsUseCase(get_model_store_path()).execute(get_training_corpus_dir())
    _reload_model_state(request.app)
    logger.info(
        "Reentrenamiento terminado: %d documentos, %d omitidos, %d entidades.",
        summary.total_entries,
        summary.skipped_count,
        len(summary.per_entity),
    )
    return {
        "total_entries": summary.total_entries,
        "skipped_count": summary.skipped_count,
        "per_entity": [
            {
                "entity": entity_summary.entity,
                "genuine_count": entity_summary.genuine_count,
                "confirmed_fraud_count": entity_summary.confirmed_fraud_count,
                "anomaly_detection_fitted": entity_summary.anomaly_detection_fitted,
                "invariant_count": entity_summary.invariant_count,
                "ml_ensemble_ready": entity_summary.ml_ensemble_ready,
            }
            for entity_summary in summary.per_entity
        ],
    }
