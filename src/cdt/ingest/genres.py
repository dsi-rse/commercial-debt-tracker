"""Ingest any registered genre into its own documents dataset."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pandas as pd

from cdt.datasets import GENRE_6K, GENRE_8K, GENRES
from cdt.ingest.core import IngestConfig, IngestRunResult, S3Client
from cdt.ingest.eightk import acquire_eightk_documents
from cdt.ingest.sixk import acquire_scraped_sixk_documents
from cdt.lease import throttled


def genre_config(config: IngestConfig, genre: str) -> IngestConfig:
    """Return ``config`` narrowed to one genre's forms and documents dataset.

    ``download`` is kept only for a genre that inlines bodies.

    Raises:
        KeyError: If ``genre`` is not registered.
    """
    record = GENRES[genre]
    return replace(
        config,
        form_types=record.form_types,
        dataset_name=record.document_dataset,
        download=config.download and record.inlines_bodies,
    )


def ingest_genre(
    genre: str,
    config: IngestConfig,
    *,
    ciks: set[str] | None = None,
    s3_client: S3Client | None = None,
    return_documents: bool = False,
    renew: Callable[[], None] | None = None,
) -> tuple[pd.DataFrame, IngestRunResult]:
    """Ingest one genre's filings for ``ciks`` over ``config``'s window.

    ``config`` is narrowed with :func:`genre_config`; the genre's candidate
    source does the rest. ``renew`` extends the caller's writer lease through
    the scan and the writes; it is throttled here, so it renews at most every
    :data:`cdt.lease.RENEW_INTERVAL_SECONDS`. Returns what
    ``run_ingest_pipeline`` returns.

    Raises:
        KeyError: If ``genre`` is not registered.
    """
    # Looked up at call time, so each genre's acquire function is patchable here.
    acquire = {
        GENRE_8K: acquire_eightk_documents,
        GENRE_6K: acquire_scraped_sixk_documents,
    }[genre]
    return acquire(
        genre_config(config, genre),
        ciks=ciks,
        s3_client=s3_client,
        return_documents=return_documents,
        renew=None if renew is None else throttled(renew),
    )
