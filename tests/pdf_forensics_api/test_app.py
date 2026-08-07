from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.pdf_builder import PdfBuilder


def _pdf_bytes(variant: int = 0) -> bytes:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    off2 = builder.add_object(2, 0, f"<< /Type /Pages /Kids [] /Count {variant} >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (2, off2, 0, "n")], size=3, root_ref="1 0 R"
    )
    return builder.build()


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("PDF_FORENSICS_TRAINING_CORPUS_DIR", str(tmp_path / "corpus"))
    monkeypatch.setenv("PDF_FORENSICS_MODEL_STORE_PATH", str(tmp_path / "model.joblib"))
    # Imported after the env vars are set: config.py reads them fresh on
    # every call, but the app module itself must not have been imported
    # (and its lifespan run) before the paths are in place.
    from pdf_forensics_api.app import app

    with TestClient(app) as test_client:
        yield test_client


def test_index_serves_frontend(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert b"PDF Forensics" in response.content


def test_summary_starts_empty(client: TestClient) -> None:
    response = client.get("/training-data/summary")
    assert response.status_code == 200
    assert response.json() == {"entities": {}}


def test_verify_with_no_trained_model_still_scores(client: TestClient) -> None:
    response = client.post("/verify", files={"file": ("doc.pdf", _pdf_bytes(), "application/pdf")})
    assert response.status_code == 200
    body = response.json()
    assert body["entity_predictions"] == []
    assert isinstance(body["risk_score"], int)


def test_verify_rejects_non_pdf(client: TestClient) -> None:
    response = client.post("/verify", files={"file": ("doc.pdf", b"not a pdf", "application/pdf")})
    assert response.status_code == 422


def test_upload_genuine_saves_file_and_updates_summary(client: TestClient) -> None:
    response = client.post(
        "/training-data/genuine",
        data={"entity": "ANSES"},
        files={"file": ("doc.pdf", _pdf_bytes(), "application/pdf")},
    )
    assert response.status_code == 200
    saved_to = Path(response.json()["saved_to"])
    assert saved_to.is_file()
    assert saved_to.parent.name == "ANSES"

    summary = client.get("/training-data/summary").json()
    assert summary["entities"]["ANSES"]["genuine"] == 1
    assert summary["entities"]["ANSES"]["confirmed_fraud"] == 0


def test_upload_confirmed_fraud_saves_under_correct_subdir(client: TestClient) -> None:
    response = client.post(
        "/training-data/confirmed-fraud",
        data={"entity": "ANSES"},
        files={"file": ("fraud.pdf", _pdf_bytes(), "application/pdf")},
    )
    assert response.status_code == 200
    assert "confirmed_fraud" in response.json()["saved_to"]

    summary = client.get("/training-data/summary").json()
    assert summary["entities"]["ANSES"]["confirmed_fraud"] == 1


def test_upload_rejects_non_pdf(client: TestClient) -> None:
    response = client.post(
        "/training-data/genuine",
        data={"entity": "ANSES"},
        files={"file": ("doc.pdf", b"not a pdf", "application/pdf")},
    )
    assert response.status_code == 422


def test_upload_sanitizes_path_traversal_in_entity_name(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        "/training-data/genuine",
        data={"entity": "../../etc"},
        files={"file": ("doc.pdf", _pdf_bytes(), "application/pdf")},
    )
    assert response.status_code == 200
    saved_to = Path(response.json()["saved_to"])
    # Resolved path must stay inside the configured training corpus dir.
    corpus_dir = (tmp_path / "corpus").resolve()
    assert corpus_dir in saved_to.resolve().parents


def test_upload_sanitizes_path_traversal_in_filename(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        "/training-data/genuine",
        data={"entity": "ANSES"},
        files={"file": ("../../evil.pdf", _pdf_bytes(), "application/pdf")},
    )
    assert response.status_code == 200
    saved_to = Path(response.json()["saved_to"])
    corpus_dir = (tmp_path / "corpus").resolve()
    assert corpus_dir in saved_to.resolve().parents


def test_retrain_with_empty_corpus_succeeds(client: TestClient) -> None:
    response = client.post("/retrain")
    assert response.status_code == 200
    body = response.json()
    assert body["total_entries"] == 0
    assert body["per_entity"] == []


def test_retrain_reflects_uploaded_documents(client: TestClient) -> None:
    for i in range(3):
        client.post(
            "/training-data/genuine",
            data={"entity": "ANSES"},
            files={"file": (f"doc{i}.pdf", _pdf_bytes(i), "application/pdf")},
        )

    response = client.post("/retrain")
    assert response.status_code == 200
    body = response.json()
    assert body["total_entries"] == 3
    entity_summary = next(e for e in body["per_entity"] if e["entity"] == "ANSES")
    assert entity_summary["genuine_count"] == 3


def test_verify_uses_retrained_model(client: TestClient) -> None:
    for i in range(3):
        client.post(
            "/training-data/genuine",
            data={"entity": "ANSES"},
            files={"file": (f"doc{i}.pdf", _pdf_bytes(i), "application/pdf")},
        )
    client.post("/retrain")

    response = client.post("/verify", files={"file": ("doc.pdf", _pdf_bytes(0), "application/pdf")})
    assert response.status_code == 200
    body = response.json()
    assert len(body["entity_predictions"]) == 1
    assert body["entity_predictions"][0]["predicted_entity"] == "ANSES"
