"""FastAPI app: document verification + minimal training-corpus admin.

Kept as a separate top-level package from `pdf_forensics` itself — the core
library stays free of web-framework concerns, the same reasoning `scripts/`
is a thin caller of the library rather than part of it.

The trained model bundle is loaded once at process startup (`lifespan`,
below) and held in `app.state`; `/retrain` reloads it in place afterward so
`/verify` reflects the new model immediately, with no process restart.
`/verify` never writes anything to disk — the uploaded bytes are processed
entirely in memory and discarded once the response is built.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
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
from pdf_forensics_api.serialization import serialize_scoring_result

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
    else:
        app.state.entity_classifiers, app.state.per_entity = (), {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _reload_model_state(app)
    yield


app = FastAPI(title="PDF Forensics Verification API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")
    return data


def _parse_or_422(pdf_bytes: bytes) -> None:
    """Fails fast on non-PDF uploads for the training endpoints, so the
    corpus never accumulates files `RetrainModelsUseCase` would just skip
    later anyway — surfacing the problem at upload time is more useful."""
    try:
        ParsePdfUseCase().execute(pdf_bytes)
    except (NotAPdfError, UnrecoverableStructureError) as exc:
        raise HTTPException(status_code=422, detail=f"Not a readable PDF: {exc}") from exc


@app.post("/verify")
async def verify(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
    pdf_bytes = await _read_upload(file)
    use_case = ScoreDocumentUseCase(
        request.app.state.entity_classifiers, request.app.state.per_entity
    )
    try:
        result = use_case.execute(pdf_bytes)
    except (NotAPdfError, UnrecoverableStructureError) as exc:
        raise HTTPException(status_code=422, detail=f"Not a readable PDF: {exc}") from exc
    return serialize_scoring_result(result)


def _save_training_file(pdf_bytes: bytes, filename: str, entity: str, *, is_genuine: bool) -> Path:
    safe_entity = _sanitize_path_component(entity, default="UNKNOWN_ENTITY")
    safe_filename = _sanitize_path_component(filename, default="upload.pdf")
    if not safe_filename.lower().endswith(".pdf"):
        safe_filename += ".pdf"

    subdir = "genuine" if is_genuine else "confirmed_fraud"
    target_dir = get_training_corpus_dir() / subdir / safe_entity
    target_dir.mkdir(parents=True, exist_ok=True)

    target_path = target_dir / safe_filename
    if target_path.exists():
        target_path = target_dir / f"{target_path.stem}-{uuid.uuid4().hex[:8]}{target_path.suffix}"
    target_path.write_bytes(pdf_bytes)
    return target_path


@app.post("/training-data/genuine")
async def upload_genuine(
    entity: str = Form(...), file: UploadFile = File(...)  # noqa: B008
) -> dict[str, Any]:
    pdf_bytes = await _read_upload(file)
    _parse_or_422(pdf_bytes)
    path = _save_training_file(pdf_bytes, file.filename or "upload.pdf", entity, is_genuine=True)
    return {"saved_to": str(path)}


@app.post("/training-data/confirmed-fraud")
async def upload_confirmed_fraud(
    entity: str = Form(...), file: UploadFile = File(...)  # noqa: B008
) -> dict[str, Any]:
    pdf_bytes = await _read_upload(file)
    _parse_or_422(pdf_bytes)
    path = _save_training_file(pdf_bytes, file.filename or "upload.pdf", entity, is_genuine=False)
    return {"saved_to": str(path)}


@app.get("/training-data/summary")
def training_data_summary() -> dict[str, Any]:
    refs = read_training_corpus_files(get_training_corpus_dir())
    entities: dict[str, dict[str, int]] = {}
    for ref in refs:
        counts = entities.setdefault(ref.entity, {"genuine": 0, "confirmed_fraud": 0})
        counts["genuine" if ref.is_genuine else "confirmed_fraud"] += 1
    return {"entities": entities}


@app.post("/retrain")
def retrain(request: Request) -> dict[str, Any]:
    summary = RetrainModelsUseCase(get_model_store_path()).execute(get_training_corpus_dir())
    _reload_model_state(request.app)
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
