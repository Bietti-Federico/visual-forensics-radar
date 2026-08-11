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
    monkeypatch.setenv("PDF_FORENSICS_LOG_PATH", str(tmp_path / "test.log"))
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
    assert body["entidades_predichas"] == []
    assert isinstance(body["puntaje_riesgo"], int)


def test_verify_response_uses_spanish_keys(client: TestClient) -> None:
    response = client.post("/verify", files={"file": ("doc.pdf", _pdf_bytes(), "application/pdf")})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "huella_digital",
        "entidades_predichas",
        "firmas",
        "puntaje_riesgo",
        "componentes",
        "motivos",
        "caracteristicas_principales",
    }
    assert {c["nombre"] for c in body["componentes"]} == {
        "motor_de_reglas",
        "probabilidad_ml",
        "deteccion_de_anomalias",
        "estructural",
        "metadatos",
        "consistencia_de_entidad",
        "integridad_de_firma",
    }


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


def test_files_listing_reflects_uploads(client: TestClient) -> None:
    client.post(
        "/training-data/genuine",
        data={"entity": "ANSES"},
        files={"file": ("doc.pdf", _pdf_bytes(), "application/pdf")},
    )
    client.post(
        "/training-data/confirmed-fraud",
        data={"entity": "ANSES"},
        files={"file": ("fraud.pdf", _pdf_bytes(1), "application/pdf")},
    )

    files = client.get("/training-data/files").json()["files"]
    assert len(files) == 2
    assert {"entity": "ANSES", "is_genuine": True, "filename": "doc.pdf"} in files
    assert {"entity": "ANSES", "is_genuine": False, "filename": "fraud.pdf"} in files


def test_delete_genuine_file_removes_it_and_updates_summary(client: TestClient) -> None:
    upload = client.post(
        "/training-data/genuine",
        data={"entity": "ANSES"},
        files={"file": ("doc.pdf", _pdf_bytes(), "application/pdf")},
    )
    filename = Path(upload.json()["saved_to"]).name

    files = client.get("/training-data/files").json()["files"]
    assert {"entity": "ANSES", "is_genuine": True, "filename": filename} in files

    response = client.delete(f"/training-data/genuine/ANSES/{filename}")
    assert response.status_code == 200
    assert response.json() == {"deleted": True}

    summary = client.get("/training-data/summary").json()
    assert summary["entities"] == {}
    assert client.get("/training-data/files").json()["files"] == []


def test_delete_confirmed_fraud_file_removes_it(client: TestClient) -> None:
    upload = client.post(
        "/training-data/confirmed-fraud",
        data={"entity": "ANSES"},
        files={"file": ("fraud.pdf", _pdf_bytes(), "application/pdf")},
    )
    filename = Path(upload.json()["saved_to"]).name

    response = client.delete(f"/training-data/confirmed-fraud/ANSES/{filename}")
    assert response.status_code == 200

    summary = client.get("/training-data/summary").json()
    assert summary["entities"] == {}


def test_delete_missing_file_returns_404(client: TestClient) -> None:
    response = client.delete("/training-data/genuine/ANSES/does_not_exist.pdf")
    assert response.status_code == 404


def test_recategorize_moves_file_to_new_entity_and_nature(client: TestClient) -> None:
    upload = client.post(
        "/training-data/genuine",
        data={"entity": "ANSES"},
        files={"file": ("doc.pdf", _pdf_bytes(), "application/pdf")},
    )
    filename = Path(upload.json()["saved_to"]).name

    response = client.post(
        "/training-data/recategorize",
        data={
            "entity": "ANSES",
            "filename": filename,
            "is_genuine": "true",
            "new_entity": "MUNICIPALIDAD_DE_JUJUY",
            "new_is_genuine": "false",
        },
    )
    assert response.status_code == 200
    new_path = Path(response.json()["saved_to"])
    assert new_path.is_file()
    assert new_path.parent.name == "MUNICIPALIDAD_DE_JUJUY"
    assert new_path.parent.parent.name == "confirmed_fraud"

    summary = client.get("/training-data/summary").json()
    assert summary["entities"] == {"MUNICIPALIDAD_DE_JUJUY": {"genuine": 0, "confirmed_fraud": 1}}


def test_recategorize_with_no_actual_change_does_not_duplicate_file(client: TestClient) -> None:
    upload = client.post(
        "/training-data/genuine",
        data={"entity": "ANSES"},
        files={"file": ("doc.pdf", _pdf_bytes(), "application/pdf")},
    )
    filename = Path(upload.json()["saved_to"]).name

    response = client.post(
        "/training-data/recategorize",
        data={
            "entity": "ANSES",
            "filename": filename,
            "is_genuine": "true",
            "new_entity": "ANSES",
            "new_is_genuine": "true",
        },
    )
    assert response.status_code == 200
    assert response.json()["saved_to"].endswith(filename)

    files = client.get("/training-data/files").json()["files"]
    assert len(files) == 1
    assert files[0]["filename"] == filename


def test_recategorize_missing_file_returns_404(client: TestClient) -> None:
    response = client.post(
        "/training-data/recategorize",
        data={
            "entity": "ANSES",
            "filename": "does_not_exist.pdf",
            "is_genuine": "true",
            "new_entity": "ANSES",
            "new_is_genuine": "false",
        },
    )
    assert response.status_code == 404


def test_suggest_entity_with_no_trained_model_returns_none(client: TestClient) -> None:
    response = client.post(
        "/training-data/suggest-entity",
        files={"file": ("doc.pdf", _pdf_bytes(), "application/pdf")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["suggested_entity"] is None
    assert body["confidence"] is None
    assert isinstance(body["risk_score"], int)
    assert isinstance(body["suggested_genuine"], bool)


def test_suggest_entity_rejects_non_pdf(client: TestClient) -> None:
    response = client.post(
        "/training-data/suggest-entity",
        files={"file": ("doc.pdf", b"not a pdf", "application/pdf")},
    )
    assert response.status_code == 422


def test_suggest_entity_uses_retrained_model(client: TestClient) -> None:
    for i in range(3):
        client.post(
            "/training-data/genuine",
            data={"entity": "ANSES"},
            files={"file": (f"doc{i}.pdf", _pdf_bytes(i), "application/pdf")},
        )
    client.post("/retrain")

    response = client.post(
        "/training-data/suggest-entity",
        files={"file": ("doc.pdf", _pdf_bytes(0), "application/pdf")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["suggested_entity"] == "ANSES"
    assert body["confidence"] > 0


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
    assert len(body["entidades_predichas"]) == 1
    assert body["entidades_predichas"][0]["entidad_predicha"] == "ANSES"


def test_verify_with_entity_override_replaces_the_prediction(client: TestClient) -> None:
    for i in range(3):
        client.post(
            "/training-data/genuine",
            data={"entity": "ANSES"},
            files={"file": (f"doc{i}.pdf", _pdf_bytes(i), "application/pdf")},
        )
    client.post("/retrain")

    response = client.post(
        "/verify",
        data={"entity": "MUNICIPALIDAD_DE_JUJUY"},
        files={"file": ("doc.pdf", _pdf_bytes(0), "application/pdf")},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["entidades_predichas"]) == 1
    assert body["entidades_predichas"][0]["entidad_predicha"] == "MUNICIPALIDAD_DE_JUJUY"
    # The classifier only ever saw ANSES — it has no structural confidence
    # for JUJUY at all, so this is 0.0, not an artificial 1.0.
    assert body["entidades_predichas"][0]["confianza"] == 0.0
