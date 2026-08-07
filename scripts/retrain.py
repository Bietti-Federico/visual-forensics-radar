"""
Retrains the model bundle from the production training corpus (see
`infrastructure/training_corpus/filesystem_training_corpus_reader.py` for
the expected directory layout). Thin CLI wrapper around
`RetrainModelsUseCase`, the same use case the API's `/retrain` endpoint
calls.

Usage:
    poetry run python scripts/retrain.py <training_corpus_dir> <model_store_path>
"""

from __future__ import annotations

import sys
from pathlib import Path

from pdf_forensics.application.model_training.retrain_models_use_case import (
    RetrainModelsUseCase,
)


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    corpus_dir = Path(sys.argv[1])
    model_store_path = Path(sys.argv[2])

    summary = RetrainModelsUseCase(model_store_path).execute(corpus_dir)

    print(f"Total entries: {summary.total_entries}")
    print(f"Skipped: {summary.skipped_count}")
    print("Per entity:")
    for entity_summary in summary.per_entity:
        print(
            f"  {entity_summary.entity}: genuine={entity_summary.genuine_count} "
            f"confirmed_fraud={entity_summary.confirmed_fraud_count} "
            f"anomaly_detection_fitted={entity_summary.anomaly_detection_fitted} "
            f"ml_ensemble_ready={entity_summary.ml_ensemble_ready}"
        )
    print(f"Saved to {model_store_path}")


if __name__ == "__main__":
    main()
