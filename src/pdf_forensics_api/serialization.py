"""Converts domain report objects into JSON-serializable dicts for the API.

Deliberately the one place translation happens: the domain/application
layers stay in English internally (their own tests, docs, and the sibling
CLI scripts all read that), but this platform's actual users are
Spanish-speaking, so everything that crosses the API boundary — keys,
enum labels, generated text — is translated here rather than in the
domain types themselves.
"""

from __future__ import annotations

from typing import Any

from pdf_forensics.application.document_scoring.score_document_use_case import (
    DocumentScoringResult,
)
from pdf_forensics.domain.signature_verification.signature_coverage import SignatureCoverage

_COVERAGE_LABELS = {
    SignatureCoverage.ENTIRE_FILE: "archivo_completo",
    SignatureCoverage.PARTIAL: "parcial",
    SignatureCoverage.UNCLEAR: "no_determinada",
}

#: Component names as produced by `GenerateRiskReportUseCase` -> label shown to users.
_COMPONENT_LABELS = {
    "rule_engine": "motor_de_reglas",
    "ml_probability": "probabilidad_ml",
    "anomaly_detection": "deteccion_de_anomalias",
    "structural": "estructural",
    "metadata": "metadatos",
    "entity_consistency": "consistencia_de_entidad",
    "signature_integrity": "integridad_de_firma",
}


def serialize_scoring_result(result: DocumentScoringResult) -> dict[str, Any]:
    fingerprint = result.fingerprint
    return {
        "huella_digital": {
            "generador": fingerprint.generator,
            "productor": fingerprint.producer,
            "version_pdf": fingerprint.pdf_version,
            "hash_xref": fingerprint.xref_hash,
            "hash_estructura": fingerprint.structure_hash,
            "hash_metadatos": fingerprint.metadata_hash,
            "hash_fuente": fingerprint.font_hash,
            "hash_arbol_paginas": fingerprint.page_tree_hash,
            "hash_caracteristicas": fingerprint.feature_hash,
            "algoritmo": fingerprint.algorithm,
            "version_esquema": fingerprint.schema_version,
        },
        "entidades_predichas": [
            {
                "clasificador": prediction.classifier_id,
                "entidad_predicha": prediction.predicted_entity,
                "confianza": prediction.confidence,
                "probabilidades": dict(prediction.probabilities),
            }
            for prediction in result.entity_report
        ],
        "firmas": [
            {
                "campo": signature.field_name,
                "digest_integro": signature.digest_intact,
                "criptograficamente_valida": signature.cryptographically_valid,
                "cobertura": _COVERAGE_LABELS[signature.coverage],
                "firmante": signature.signer_subject,
                "fecha_firma": (
                    signature.signing_time.isoformat() if signature.signing_time else None
                ),
            }
            for signature in result.signature_report
        ],
        "puntaje_riesgo": result.risk_report.risk_score,
        "componentes": [
            {
                "nombre": _COMPONENT_LABELS.get(component.name, component.name),
                "puntaje": component.score,
                "peso": component.weight,
            }
            for component in result.risk_report.components
        ],
        "motivos": list(result.explanation.reasons),
        "caracteristicas_principales": [
            {
                "modelo": feature.model_id,
                "caracteristica": feature.feature_name,
                "valor_shap": feature.shap_value,
            }
            for feature in result.explanation.top_features
        ],
    }
