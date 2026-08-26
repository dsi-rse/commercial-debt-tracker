# Design: completion keyed on row outcomes (#49, #62)

## The two bugs, one root cause

The pipeline records "done" at the wrong granularity and from the wrong signal:

- **#49** — extract's completion registry unions every *visited/claimed*
  partition, even when rows ended `ERROR` on provider failures (the 402 storm:
  3,465 rows permanently skipped). The failure registry records those rows but
  nothing reads it back. Infrastructure errors also burn per-row retries across
  the whole corpus instead of stopping the run.
- **#62** — every stage skips a source partition when its *target exists* or its
  *path* is registered complete, but ingest merges late-arriving rows into
  existing partition files in place, so those rows are never processed.

Both reduce to: completion must be keyed on **what was actually processed**
(which rows, from which version of the source), not on "we looked at this path
once".

## Design

### 1. Row states already carry the needed distinction

`ExtractionRowState.state` ends in `SUCCESS` (extracted), `FAILED` (validation
exhausted — a real content verdict), or `ERROR` (an exception: provider,
network, preprocess). The rule everywhere becomes:

> **A row counts toward completion when it reached any terminal state — and
> infrastructure failures never produce one.** The live path aborts the run on
> the first infra error (rows stay non-terminal, hence pending); the batch path
> requeues under the resubmission cap (#84). Deterministic content errors
> (`FAILED` validation, parse/preprocess `ERROR`s) stay terminal so partitions
> still converge instead of retrying forever; the failure registry records them
> for operators.

Registry entries carry an explicit ``complete`` flag: a partition interrupted
mid-pass keeps its terminal ``item_ids`` (never re-paid) but stays pending.

### 2. Versioned, row-aware completion registry (v2)

`runs/<stage>/completed-partitions.json` becomes:

```json
{
  "version": 2,
  "partitions": {
    "<source partition path>": {
      "fingerprint": "<source object version at processing time>",
      "item_ids": ["..."]        // extract only: content-terminal rows
    }
  }
}
```

- **fingerprint** is the source object's version captured during the discovery
  LIST (S3 ETag; size+mtime locally) — zero extra requests on top of the #83
  tier-1 listing. Ingest's in-place merge rewrites the object, so a grown
  partition's fingerprint no longer matches and the partition is pending again.
  That is the whole #62 fix.
- **item_ids** (extract only) makes re-processing row-level: when a partition is
  pending (new, fingerprint-changed, or previously ERROR-interrupted), only
  relevant rows *not* already content-terminal are extracted. Successful rows
  from a partial run are never re-paid, and the write path merges new mentions
  into the existing target partition. This is also the #49 re-drive: nothing new
  to run — ERROR rows are simply still pending, and the next daily/poll pass
  picks them up.
- The v1 format (a bare path list) loads as v2 entries with no fingerprint and
  no item_ids, meaning "complete as recorded, but reprocess if the source ever
  changes" — no migration step, fingerprints accrue from the next run.

Itemize/classify use the same v2 registry with fingerprints only (no item_ids):
they are cheap and deterministic, so a grown partition is simply recomputed
whole and its target rewritten.

The target-exists check (`existing_*_ids`) remains as a fast-path only for
partitions with **no registry entry at all** (pre-registry data); when an entry
exists, the fingerprint decides.

### 3. Infrastructure errors abort, not iterate (#49a)

A new `InfrastructureError` classification in the extractor client layer:
connection errors and HTTP 402/408/429/5xx. The live workflow re-raises it
instead of terminating the row; `extract_pending_items` stops the run at the
first occurrence (partitions already finished stay finished; in-flight rows stay
pending via §2). The batch path already retries batch-level failures under the
resubmission cap (#84); per-request infra error *lines* now also requeue the row
(clear pending) instead of `record_stage_error`, under the same cap.

### 4. Batch job creation/finalize become row-aware

- `collect_pending_extract_items` claims (partition, rows-to-do) pairs using §2
  — previously content-terminal item_ids are excluded from the job.
- `_finalize_job` records per-partition item_ids for content-terminal rows and a
  fingerprint only if every relevant row in that partition is now
  content-terminal **and** the fingerprint at claim time is recorded; ERROR rows
  leave their partition pending for the next job.
- Mentions for a partially-completed partition are merged into the existing
  target partition rather than overwritten.

## Cost/consistency notes

- The registry file grows by ~one id list per partition (~2 relevant items per
  partition at current volume); it stays one JSON object, rewritten as today.
- Deterministic recompute of itemize/classify on fingerprint change can emit
  updated rows for already-extracted items; extract's item_ids exclusion means
  changed *text* for an already-extracted item is NOT re-extracted. That is
  today's behavior too (path-level skip) — text-change reprocessing is out of
  scope; #62 is about *new* rows.
- Writers of the registry already run under the pipeline-writer lease on the
  batch path; the live path acquired it in the hardening pass.
