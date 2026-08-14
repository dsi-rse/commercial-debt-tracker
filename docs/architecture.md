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
   Extracts structured debt-instrument mentions from relevant items and writes a full per-run audit log. Two backends exist: a synchronous `live` backend (OpenRouter chat completions) and the deployed `batch` backend (OpenAI Batch API). See "Extractor Design" below.
5. `match`
   Consolidates mention rows into debt instruments and writes instrument-level outputs partitioned by CIK shard.

The stage-oriented CLI is `cdt`. The deployment-oriented entrypoint is `cdt-orchestrator`, which simply resolves defaults and runs the same pipeline code used locally.

After matching, the pipeline can optionally materialize four final snapshot tables for downstream consumers:

- `items/latest.parquet`
- `debt-instruments/latest.parquet`
- `debt-instrument-mentions/latest.parquet`
- `mention-cluster-edges/latest.parquet`

Those files are written only when a final database root is configured.

## Why CDT Is File-Native

CDT stores canonical state as Parquet, JSON, and JSONL artifacts under one root instead of using SQLite or another mutable database.

That choice is visible throughout the code:

- stage completion is inferred from partition presence
- partitions are rewritten deterministically
- stage manifests are sidecar metadata, not the source of truth
- the same pipeline can target either local paths or `s3://` URIs

This makes the system easier to rerun, inspect, diff, and backfill in batch environments like ECS.

The final snapshot parquet files are intentionally not used as pipeline inputs. They are convenience exports derived from the canonical partition datasets.

## Why The Pipeline Is Split This Way

The stage boundaries are mostly cost and recovery boundaries.

- `ingest` is pure acquisition and can be rerun without recomputing extraction.
- `itemize` reduces each filing to the sections the downstream pipeline cares about.
- `classify` keeps LLM cost under control by filtering out obviously irrelevant items first.
- `extract` is isolated because it is the most expensive and least deterministic stage; it also records `extractor-runs/run_id=.../full.jsonl` for auditability.
- `match` is deterministic and cheap enough to rerun from mention outputs.

## Daily Versus Historical Versus Poll Runs

Three execution modes exist:

- `daily`
  Defaults to yesterday's filing date when no dates are provided. With the default
  `batch` backend it runs ingest → itemize → classify and refreshes match/final
  snapshots, but does not run the LLM extract stage; extraction is submitted and
  advanced asynchronously by `poll`.
- `poll`
  Runs on an hourly schedule and advances the OpenAI batch extract job by one tick
  (see "Extractor Design"). It never ingests; it only moves extraction forward and,
  when a job completes, re-runs match + finalize.
- `historical`
  Requires explicit dates and is intended for backfills. Historical always uses the
  synchronous `live` extract backend so a single command produces final outputs.

The scheduler runs `daily` (once a day) and `poll` (hourly). Historical runs are manual
by design so wide backfills are deliberate, observable operations. `daily
--extractor-backend live` restores the original fully synchronous single-run behavior.

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

A row that ends non-SUCCESS produces no mentions, but its partition is still marked
completed, so it is never revisited. Both backends therefore record dropped rows in
`failures/extract/failures.json` (see [schema.md](schema.md)) with their source partition,
stage, and error. That registry is diagnostic — nothing reads it to schedule work — but it
turns "silently short a few mentions" into a countable list. Retrying those rows remains a
manual, partition-granular `--force` operation.

### Live versus batch backends

The stage objects (`preprocess`/`validate`/`postprocess`/`early_stop`/`build_retry_message`)
are pure and backend-agnostic. Two backends drive them:

- The `live` backend (`OpenRouterChatClient`) runs the workflow synchronously, one item
  and one LLM call at a time. It is used for local development, `cdt extract`,
  `cdt pipeline`, and `historical` runs.
- The `batch` backend (`OpenAIBatchClient`) drives the same stages asynchronously through
  OpenAI's Batch API (~50% cheaper, up to 24h per round). This is the deployed default.

Because the two backends talk to different APIs, the request parameters they share are
resolved in one place so the same model cannot behave differently depending on which
backend ran it:

- Sampling comes from `sampling_params(model)`. Reasoning models (`gpt-5`, `o1`, `o3`,
  `o4` families, matched on the native id) reject `temperature != 1`, so temperature is
  sent only to models that can honor it; for those it is pinned to `0.0`.
- Reasoning effort is configured in OpenRouter's vocabulary and translated for OpenAI by
  `openai_reasoning_effort`: `none` → `minimal` and `xhigh` → `high`. Translating rather
  than dropping matters — an omitted `reasoning_effort` leaves a `gpt-5` batch on the API
  default (`medium`) while the live backend ran with reasoning off. An effort neither
  vocabulary accepts raises on the poll tick before a job is created.

### The batch extract state machine

Because each item flows through several sequential stages, each retryable, and each batch
round can take up to a day, a single run can span many days. Rather than block an ECS task
for days, extraction is a resumable, file-native state machine advanced by the hourly
`poll` tick. Every poll tick runs under a single `pipeline-writer` lease (`cdt.lease`,
built on S3 conditional writes), so a tick that outlives its hour — or an EventBridge
retry — cannot overlap the next one, and `daily`'s match/finalize cannot interleave with a
completing poll's; a run that finds the lease held reports `locked` and skips its turn.

Each tick:

1. reconciles any OpenAI batch tagged with the active job but not yet recorded (crash recovery),
2. folds results from any completed batch into per-item states via the same
   validate/postprocess/retry logic as the live backend — a result only applies if it
   matches the request the item is recorded as waiting on, so replaying a stale batch
   is a no-op,
3. submits the next batch for every item still needing a call (advancing and retrying
   together), chunked by both request count and bytes to stay under OpenAI's per-file
   limits,
4. persists item states and batch records at every transition, so a crash at any point
   is recoverable,
5. and, when every item is terminal, writes the `mentions` partitions, audit log, and
   completion registry, then clears the active-job marker so match + finalize can run.

Whole-batch failures terminate their items rather than looping forever, recording the
per-request reasons from the batch's error file (or, failing that, its batch-level
errors); expired batches salvage whatever completed and re-submit the rest, up to a
per-item cap of consecutive expired rounds. Job state lives under `extract-batches/`
(see [schema.md](schema.md)).

If `active.json` names a job directory that was deleted or only partially written, the
tick cannot load it. Rather than raising the same way forever, it logs the reason, clears
the marker, and reports `reset`; the next tick starts a fresh job over the same
classification partitions, which are still unclaimed because the completion registry is
only written when a job finishes. Any batch the abandoned job left in flight is lost
spend — orphan reconciliation is scoped per job, so a new job will not adopt it.

Two `cdt` commands cover the same ground manually:

- `cdt show-extract-job` reports the active job (rows, terminal rows, in-flight batches,
  claimed partitions), or `corrupt` with the reason. Read-only, and exits non-zero on a
  corrupt job so a health check fails loudly.
- `cdt reset-extract-job --yes` clears the marker immediately, taking the
  `pipeline-writer` lease first so it cannot race a running tick. Without `--yes` it only
  reports what a reset would abandon. The job directory is left in place for inspection.

## Matcher Design

Matcher outputs are partitioned by `cik_shard` instead of by filing date because the matching problem is company-scoped. Mentions for the same issuer need to be considered together across time.

The matcher uses deterministic surfaces derived from:

- normalized instrument names
- normalized dates
- normalized amounts
- lender signatures
- one-hop lineage cues like `amendment_of` and `split_of`

This is a pragmatic middle ground: simpler than a graph database or long-lived entity service, but enough to build useful instrument histories from noisy filing text.

## Dashboard Handoff

After matching succeeds, CDT can write final parquet snapshots for dashboard and database consumers:

- `items/latest.parquet`
- `debt-instruments/latest.parquet`
- `debt-instrument-mentions/latest.parquet`
- `mention-cluster-edges/latest.parquet`

The processor does not publish Cloudflare R2 JSON directly. The `../commercial-debt-tracker-dashboard` repository owns the publisher that reads these final parquet snapshots and writes `generated/*` JSON to R2.
