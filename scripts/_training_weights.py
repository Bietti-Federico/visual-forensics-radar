"""
Shared helper for scripts/train_and_save_models.py and
scripts/train_and_score_real_documents.py: computes a per-sample training
weight so genuinely real documents (and their direct transformations) count
more than synthetic field-substituted variants generated from them while a
real training corpus is still small — see
pdf-forensics-benchmark/examples/generate_field_variants.py for how those
variants are produced and named.

Not part of the tested `pdf_forensics` package: this is glue specific to how
these two scripts happen to build their benchmark dataset (recognizing
"real" documents by their exact `external:<filename>` generator string),
not a general platform concept.
"""

from __future__ import annotations

# Must match the `external:<stem>` generator string DatasetBuilder assigns to
# each of examples/input/real_documents/*.pdf when passed as an
# `external_pdf_paths` root — NOT the field-substituted variants generated
# from them (those are named "external:<template>__variant_<seed>__<hash>").
REAL_DOCUMENT_GENERATORS = frozenset(
    {
        "external:Recibo-ANSES",
        "external:Recibo Municipalidad La Rioja",
        "external:Recibo Municipalidad de Jujuy",
    }
)


def compute_sample_weight(dataset, real_weight: float) -> list[float]:
    """
    `real_weight` for every sample whose `original_id` traces back to one of
    the genuinely real base documents (the root itself AND every
    transformation derived from it) — 1.0 for everything derived from a
    synthetic field-substituted variant.
    """
    trusted_original_ids = {
        sample.source.original_id
        for sample in dataset.samples
        if sample.source.is_original and sample.source.generator in REAL_DOCUMENT_GENERATORS
    }
    return [
        real_weight if sample.source.original_id in trusted_original_ids else 1.0
        for sample in dataset.samples
    ]
