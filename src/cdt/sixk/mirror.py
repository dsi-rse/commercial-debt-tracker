"""Where CDT keeps its own copy of a 6-K submission.

A 6-K document row points at one submission through ``resource_uri``, the way an
8-K row points at the scraper's complete submission text file. The 6-K path
cannot point at such a file, because the scraper stores a filing as one object
*per document* with no whole-submission object to name. So ingest assembles one
and writes it here, and the row names it — which is what keeps the row shape,
and every stage that reads it, identical across the two genres.
"""

from __future__ import annotations

from cdt.storage import ArtifactPath, join_artifact_path, normalize_artifact_path

MIRROR_DATASET_NAME = "raw-documents"
MIRROR_GENRE = "sixk"


def mirror_root(artifact_root: ArtifactPath) -> str:
    """Return the root of CDT's own copies of 6-K submissions."""
    return join_artifact_path(
        normalize_artifact_path(artifact_root), MIRROR_DATASET_NAME, MIRROR_GENRE
    )


def mirror_path(
    artifact_root: ArtifactPath, *, filing_date: str, accession_number: str
) -> str:
    """Return the mirror path for one submission.

    Gzipped, and read back through ``ingest.decode_document_bytes``, which
    sniffs the gzip magic — so the stage that resolves this URI needs to know
    nothing about the compression, and the scraper's copies are stored the same
    way. Date-prefixed for navigability and so a storage lifecycle rule can
    address the old ones.

    Doubles as the resume ledger: a filing whose mirror exists is not read or
    assembled again, so re-running a range costs one existence check per filing
    and no document reads.
    """
    return join_artifact_path(
        mirror_root(artifact_root),
        f"date={filing_date}",
        f"{accession_number}.txt.gz",
    )
