"""Where CDT keeps its own assembled copy of each 6-K submission.

A 6-K document row's ``resource_uri`` names its mirror copy, as an 8-K row's
names the scraper's complete submission file.
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

    The file is gzipped (``ingest.decode_document_bytes`` detects that on
    read) and lives under ``date={filing_date}/`` so lifecycle rules can target
    old partitions. Its existence marks the filing as already acquired.
    """
    return join_artifact_path(
        mirror_root(artifact_root),
        f"date={filing_date}",
        f"{accession_number}.txt.gz",
    )
