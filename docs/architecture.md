# CDT Architecture

Commercial Debt Tracker (CDT) turns SEC 8-K filings into a canonical, queryable history of debt instruments without maintaining an application database.

## End-to-End Flow

The pipeline runs in five stages:

1. `ingest`
   Reads scraper-managed filing manifests from S3, selects 8-K complete submission text files for the configured CIK set, and writes canonical `documents` partitions.
2. `itemize`
   Extracts only the 8-K items CDT cares about today: `1.01`, `1.02`, `2.03`, `2.04`, `7.01`, and `8.01`.
3. `classify`
   Uses a local TF-IDF plus linear SVC model to mark item sections as relevant or irrelevant before any LLM call.
4. `extract`
   Uses OpenRouter-backed chat completions to extract structured debt-instrument mentions from relevant items, and writes a full per-run audit log.
5. `match`
   Consolidates mention rows into debt instruments and writes instrument-level outputs partitioned by CIK shard.

The stage-oriented CLI is `cdt`. The deployment-oriented entrypoint is `cdt-orchestrator`, which simply resolves defaults and runs the same pipeline code used locally.

## Why CDT Is File-Native

CDT stores canonical state as Parquet, JSON, and JSONL artifacts under one root instead of using SQLite or another mutable database.

That choice is visible throughout the code:

- stage completion is inferred from partition presence
- partitions are rewritten deterministically
- stage manifests are sidecar metadata, not the source of truth
- the same pipeline can target either local paths or `s3://` URIs

This makes the system easier to rerun, inspect, diff, and backfill in batch environments like ECS.

## Why The Pipeline Is Split This Way

The stage boundaries are mostly cost and recovery boundaries.

- `ingest` is pure acquisition and can be rerun without recomputing extraction.
- `itemize` reduces each filing to the sections the downstream pipeline cares about.
- `classify` keeps LLM cost under control by filtering out obviously irrelevant items first.
- `extract` is isolated because it is the most expensive and least deterministic stage; it also records `extractor-runs/run_id=.../full.jsonl` for auditability.
- `match` is deterministic and cheap enough to rerun from mention outputs.

## Daily Versus Historical Runs

Two execution modes exist:

- `daily`
  Defaults to yesterday's filing date when no dates are provided.
- `historical`
  Requires explicit dates in the deployed orchestrator and is intended for backfills.

The scheduler only runs `daily`. Historical runs are manual by design so wide backfills are deliberate, observable operations.

## Classifier Design

The classifier is a saved local model under `data/models/classifier/tfidf-linear-svc/`. It is trained with threshold selection aimed at a high recall target, which reflects the product decision to prefer over-inclusion before extraction rather than miss potentially relevant debt disclosures.

In practical terms:

- false positives cost extra extraction work
- false negatives can permanently hide debt activity from downstream datasets

## Extractor Design

The extractor uses a multi-step validation workflow rather than accepting raw model output directly.

- the first stage produces XML-tagged NER output and validates that the text is preserved exactly
- the next stages convert those tagged spans into structured instrument mentions
- each row can retry validation failures up to `max_attempts`
- every attempt is recorded in the extractor audit log

This is why CDT can tolerate LLM use in a batch pipeline without treating the model output as unverified truth.

## Matcher Design

Matcher outputs are partitioned by `cik_shard` instead of by filing date because the matching problem is company-scoped. Mentions for the same issuer need to be considered together across time.

The matcher uses deterministic surfaces derived from:

- normalized instrument names
- normalized dates
- normalized amounts
- lender signatures
- one-hop lineage cues like `amendment_of` and `split_of`

This is a pragmatic middle ground: simpler than a graph database or long-lived entity service, but enough to build useful instrument histories from noisy filing text.

## Optional Dashboard Publishing

When the R2 environment variables are present, the pipeline publishes a denormalized dashboard snapshot after matching succeeds. The publisher writes:

- `index.json`
- `companies/<cik>.json`
- `debt-instruments/<instrument_id>.json`

It only uploads objects whose content changed, which keeps publish runs cheap and idempotent.
