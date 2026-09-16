"""Where CDT keeps its own copy of a 6-K submission.

A 6-K document row points at one submission through ``resource_uri``, the way an
8-K row points at the scraper's complete submission text file. Neither 6-K
source can point at such a file: EDGAR serves one but outside CDT's storage, and
the scraper stores a filing as one object *per document* with no whole-submission
object to name. Both therefore write a submission here and name it, which is
what keeps the row shape — and every stage that reads it — indifferent to where
the filing came from.
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

    One path for both sources, deliberately: the mirror is each source's resume
    ledger (a filing whose mirror exists is not fetched again), so sharing it
    makes the cutover from EDGAR to the scraper free. Filings EDGAR already
    supplied keep the bytes it served rather than being re-acquired.
    """
    return join_artifact_path(
        mirror_root(artifact_root),
        f"date={filing_date}",
        f"{accession_number}.txt.gz",
    )
