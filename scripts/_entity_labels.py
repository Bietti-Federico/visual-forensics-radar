"""
Shared helper for the training/scoring scripts: maps a benchmark sample's
`generator` string (or a raw filename, for scoring a fresh target file) to
one of the known entities this platform has real documents for.

Not part of the tested `pdf_forensics` package — same reasoning as
scripts/_training_weights.py: this is glue specific to the filenames chosen
for examples/input/real_documents/*.pdf in the sibling benchmark project,
not a general platform concept.
"""

from __future__ import annotations

# Substring patterns (matched case-insensitively against a generator's
# `external:<stem>` string, or a plain filename stem) that identify each
# known entity. Kept as single broad substrings (not per-document patterns)
# specifically so a new real document or field-substitution template for an
# entity already listed here needs no change — e.g. "anses10__variant..."
# and "anses11__variant..." both already match "anses" without being added
# explicitly. Only add a new dict entry when a genuinely new entity shows up.
ENTITY_PATTERNS: dict[str, tuple[str, ...]] = {
    "ANSES": ("anses",),
    "LA_RIOJA": ("la rioja", "la_rioja", "mrioj"),
    "JUJUY": ("jujuy", "mjujuy"),
}


def entity_for_generator(generator: str) -> str | None:
    stem = generator.removeprefix("external:").lower()
    for entity, patterns in ENTITY_PATTERNS.items():
        if any(pattern in stem for pattern in patterns):
            return entity
    return None


def compute_entity_labels(dataset) -> list[str | None]:
    """
    `entity_for_generator` only has a `generator` string for ROOT samples
    (`is_original=True`) — every sample (root or transformed) is labeled via
    its `original_id`'s root, since a transformation never changes which
    entity produced the underlying document.
    """
    entity_by_original_id: dict[str, str | None] = {}
    for sample in dataset.samples:
        if sample.source.is_original:
            entity_by_original_id[sample.source.id] = entity_for_generator(sample.source.generator)

    return [entity_by_original_id.get(sample.source.original_id) for sample in dataset.samples]
