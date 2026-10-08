# Pipeline, ingest and publish: design decisions

This page explains why the pipeline's stage sequence, whole runs (`cdt run`)
with their lease and batch poll loop, the CLI, ingest, 8-K segmenting, the item
classifier and the final-snapshot publish work the way they do, and records the measurements behind them. Code
docstrings give the current contract; this page gives the reasons. Sections are
headed by the function or constant they explain (module in parentheses), so you
can search for a name you saw in the code.

## Stage orchestration (`cdt.pipeline`)

The `FINAL_OUTPUT_*` constants and `finalize_after_match` below live in `cdt.publish`, and `GENRE_8K` / `GENRE_6K` in `cdt.datasets`; they are described here because they shape the run.

### `FINAL_OUTPUT_TABLES`

The published `items` table unions the 8-K segmenter's 8-K item sections with the
6-K snippets. Both are "the unit of text a mention was extracted from", and
every consumer joins a mention to its unit by `item_id`. If only the 8-K units
were published, every 6-K mention would have no row to join to. The website
reads item text, the filing's SEC URL and its accession number from that row,
so a 6-K instrument would show none of them (#172).

### `FINAL_OUTPUT_TABLE_COLUMNS`

A 6-K snippet row has the classifier's three columns and the triage stage's six
on top of the 8-K segmenter's sixteen. The published `items` table keeps the
8-K segmenter's shape, so the extra columns are dropped rather than widening the
table with columns that are null on every 8-K row. The snippet's span and
verdict can still be queried in the `sixk-snippets` dataset.

### `FINAL_OUTPUT_TABLE_FORM_TYPES` / `FORM_TYPE_COLUMN`

`items` is the one published table where a consumer otherwise cannot tell the
genres apart: none of the 8-K segmenter's sixteen columns records which kind of
filing a row came from. The column is called `form_type` because the documents
dataset already uses that name for the same two values, so one vocabulary
covers both ends.

### `GENRE_8K` / `GENRE_6K` / `DEFAULT_GENRES`

A genre is a form family plus the stages that turn it into rows the extractor
can read. Both go ingest → segment → classify and meet at extract. 8-K segments
into item sections, which the item classifier scores. A 6-K has no items, so it
segments into window spans (`sixk-windows`), and its classify stage is the
two-stage triage, which writes rows in the same classified-item columns
(`sixk-snippets`). `cdt.pipeline.segment_genre` and `classify_genre` dispatch a
stage to its genre's module, for both the CLI's stage commands and a whole run.

Each genre is one record in `cdt.datasets.GENRES`: its SEC forms, and the
datasets its upstream stages write (documents, segments, the rows it contributes
to the published `items` table, the rows the extractor reads). The registry lives in
the leaf `datasets.py` and holds data only, so every stage can read it without
importing another stage. What depends on the set of genres is derived from it
rather than listed by hand: the extractor's `CLASSIFICATION_SOURCES`, the
`items` table's `form_type` stamps, the CLI default and the pipeline's prepare
loop. Adding a genre is a record, a branch in `segment_genre` and
`classify_genre`, and its stage modules.

The CLI defaults to every genre. A run is asked for CIKs and a date range, and
the caller should not have to know, or keep in sync with the scraper's
coverage, which forms those filers happened to file. `--genres` narrows the
run when it is deliberately about one genre.

### `PipelineConfig.genres`

The dataclass default is 8-K only, narrower than `DEFAULT_GENRES`. `cdt run`
passes its `--genres` default through, so scheduled and hand-typed runs
still prepare both (`test_scheduled_runs_prepare_both_genres_by_default` pins
this). Building a config in code is not a request for a full run, though, and
the 6-K chain scrapes the network and calls a paid model before it does
anything else. A caller that never mentions genres should get the stages it
named and nothing that spends money for it.

### One CIK list for every genre

A run searches every selected genre for the same CIK list: the caller names
issuers, not forms. A separate 6-K list (`--sixk-cik-file`) existed because a
list chosen for 8-K coverage may contain no foreign private issuers; it was
removed in favour of putting those issuers in the one list (2026-10).

### `Pipeline._setup` (genre validation)

A config built in code (a test, a notebook) skips the CLI's
parsing. A genre list that matches nothing would then run extract and finalize
over whatever the last run left behind and report success, so `_setup`
validates the list itself.

### `normalize_genres`

An unknown genre is a typo. If a run quietly prepared nothing, or less than was
asked, it would look like a corpus with no filings, so unknown genres raise.
The order is normalized so 8-K prepares first however the caller spelled the
list.

### `Pipeline._prepare_genres`

The two chains are independent until extract. They read different documents
datasets and write different classification sources, and work selection is by
source-partition fingerprint per dataset (#62). A genre that fails therefore
cannot corrupt the other genre's state. It only leaves its own partitions
pending for the next run.

That is why each genre's chain runs in its own `try` (2026-10): one genre
failing, for example the 6-K triage provider running out of credit, no longer
takes down the other genre's prepare, extract and publish. A partial success
is still a failed run. It exits nonzero and skips the daily heartbeat line, so
the heartbeat alarm and the ECS task-failure alarm fire as before.
`LeaseLostError` is re-raised: a run that lost its lease must stop writing.

### `Genre.inlines_bodies` (6-K never `download`)

`cdt.ingest.genres.genre_config` keeps `download` only for a genre whose
record sets `inlines_bodies`, and the 6-K record does not. A 6-K row points
at the assembled submission in CDT's mirror. Inlining bodies into the
documents partition would make every read of the partition pay for every
body (#69).

### `Pipeline._renew`

Historical runs outlast the lease TTL by hours. Renewing between stages stops
the run from being stolen mid-write, and the hook raises `LeaseLostError` if
the lease has already been stolen (#89). Every stage also renews inside
itself, because one stage alone can outlast the TTL. Segment, classify and
match renew per partition (or per shard). Ingest renews per scanned day, per
manifest and per candidate: an 8-K backfill fetches every manifest in its
window before it writes anything. Live extraction renews before each item's
model calls and before each partition write. Ingest and live extraction call
their hook per item, so they wrap it in `cdt.lease.throttled`, which renews at
most every `RENEW_INTERVAL_SECONDS` (5 minutes). A lease renewed that recently
cannot have expired, so it cannot have been stolen, and skipping the call in
between loses no safety.

### `DAILY_LOOKBACK_DAYS` / `resolve_mode_dates`

Daily mode scans a rolling window, the `DAILY_LOOKBACK_DAYS` (5) filing dates
ending yesterday, not a single day. A
manifest that the scraper writes or repairs after CDT's morning pass would
otherwise never be scanned again: a permanent gap that nothing reports (#90).
Ingest deduplicates by accession, so the re-scan costs only LIST/GET requests,
and the fingerprint registries carry late merges downstream.

### `finalize_after_match`

Every whole run that matches and publishes finishes through this one
function (`cdt match` runs the lineage pass itself and does not publish;
`cdt publish` calls `publish_final_tables`, the publish half, alone). When the
lineage pass was wired into only one of three entry points, production
published 537 of 542 instruments as lineage heads (#170).

Amendment lineage spans filings, so it can only be derived after every shard
has matched. That makes it a post-pass over the whole corpus. It re-derives
every pointer it has ever inferred (#204). It is skipped only when match
produced nothing, because it would then read three empty datasets to write
none.

## The publish (`cdt.publish`)

### `write_final_output_tables` (atomic generation)

Writing four separate `<table>/latest.parquet` objects in a loop can never be
consistent as a set (#91):

- a consumer polling mid-loop reads mixed generations, such as mentions that
  refer to instruments that do not exist yet;
- a crash between writes leaves that mixed state published for good;
- an accidentally empty dataset overwrites a good snapshot with zero rows.

So each publish first writes the whole generation under an immutable
`final-snapshots/snapshot=<run_id>/` prefix under the artifact root. Then it
replaces a single `latest.json` pointer, as the last atomic step. Consumers
that need a consistent four-table generation should resolve that pointer. Only
after that are the per-table `latest.parquet` objects under the final database
root refreshed. Each of those objects is atomic on its own, but the set is not
consistent mid-publish.

### `final_snapshots_root`

The snapshots live under the artifact root, not the final database root. The
final database prefix is a parquet-only contract for downstream consumers, so
the control metadata (`latest.json`) and the immutable generation copies stay
with the pipeline's other artifacts.

### `_prune_old_snapshots`

Two generations stay readable, so a consumer that resolved the previous pointer
moments ago can still finish reading it.

### `FINAL_SNAPSHOT_GUARD_RATIO` / `_guard_against_shrinkage`

A table that shrinks below half its published row count blocks the publish
unless `--force-publish` is passed. The likeliest causes are a bug or a half-built artifact
root, not a real mass deletion of filings. The prior counts come from the
published `latest.parquet` footers, not from the pointer. The pointer lives
with the artifact root, so a half-built or newly pointed artifact root has no
pointer, and that is exactly the case that must not overwrite a good database.
Footer metadata gives the counts without decoding any columns.

### `publish_would_republish_nothing` (the skip gate)

The publish is the most expensive step in the pipeline, and its cost is set by
request count, not bytes. Measured in production, publishing a delta of 14
documents took 25 minutes and 21,214 sequential GETs at about 70 ms each. The
publish pays that whether or not the run produced anything, and it runs twice
per batch cycle, because both `run.run_prepare_then_publish` and `run.run_poll` finalize.

The gate asks whether anything the publish reads has changed since the
generation the pointer names. It compares a digest of the source dataset roots
with the one recorded when that generation was written. Gating on "did match
produce instruments" answers a different question. `match_pending_mentions`
returns every shard's full instrument table, not a delta, so that frame is
empty only on a corpus that has never produced an instrument. Such a gate would
never have fired on the 542-instrument root #110 was measured against. It also
cannot see `items`, which segment and 6-K classify write before match runs.

Three things stop the gate from blocking every publish:

- `--force-publish` (`force_publish`) overrides it.
- A pointer with no recorded digest publishes, which records one for next
  time.
- A final database root missing any `latest.parquet` publishes regardless.
  Otherwise a newly pointed output root would stay empty until someone passed
  `--force-publish`. The check costs four HEAD requests on objects the publish would
  write anyway.

A run that crashed between writing a dataset and publishing it needs no
override to recover: the datasets changed, so the digest changed, so the next
run publishes.

The override is its own flag, not `--force`. `--force` means "reprocess
partitions", and an operator reaches for it on exactly the runs most likely to
leave a root half-built. If it also lowered `_guard_against_shrinkage`, such a
run could publish that root over a good database. So `--force` reaches match but
never the publish, and only `--force-publish` lowers either publish guard. A crash during the publish works the same way (see the
next section).

With no final database root the gate answers True without listing anything:
`write_final_output_tables` would return before reading.

### `PUBLISH_SOURCE_DIGEST_KEY` (recorded last)

The digest is written into the pointer only after all four `latest.parquet`
objects are written (#222). When it was recorded with the first pointer write,
a crash in the table loop left a pointer whose digest matched while the
database root still held the previous tables, and every later run skipped the
publish. Now a crash leaves no digest, which the gate treats as "unknown" and
publishes.

`write_final_output_tables` takes the digest before it reads the sources, so a
source written during the reads counts as changed. That costs at most one
redundant publish next time, instead of being treated as already published.

### `publish_source_digest`

This works like `pending_source_partitions` (#62): compare a stored source
version. But it uses `artifact_content_versions`, which is byte-based on both
backends. The registries' mtime-based version would be wrong here. The matcher
rewrites every shard on every run, almost always with identical content, so an
mtime digest would change on every run and the gate would never fire.

The bytes are not the whole input. The same partitions published by a
different publisher are a different snapshot, so the digest also covers which
roots feed which table, `MATCHER_SCHEMA_VERSION`, and `PUBLISH_FORMAT_VERSION`
(#223). On S3 the digest costs one LIST per root, and the ETag comes with it.

### `PUBLISH_FORMAT_VERSION`

The source digest cannot see code. If the publish changes what it writes for
the same source bytes (a column projection, the `form_type` stamp,
`normalize_snapshot_text`, any reshaping in `write_final_output_tables`)
without a version bump, the gate keeps skipping, and the published tables keep
the old shape until some partition happens to change (#223).

### `_read_published_table`

Empty frames are left out of the concat. A dataset with no partitions reads
back with all-object columns, and pandas would widen the integer columns of the
frames next to it to float. That would change a published table's types on any
run where one genre produced nothing.

`form_type` is stamped at publish time, not read from the source, because
neither dataset carries it. The 8-K segmenter deliberately does not copy it from the
documents dataset (see `ITEM_DOCUMENT_COLUMNS`).

### `normalize_snapshot_text` / `normalize_snapshot_cell`

Partitions written before a text column existed, or by a stage that turned a
missing value into a string, contain literal text such as `nan`. Dashboard
consumers read the snapshots directly, so these are nulled on the way out.
Only string cells are coerced: an object column can hold booleans, and passing
everything through the text helper would publish `True` as `"True"`. Nulling
happens once, up front, so the immutable snapshot and the database root
publish the same values.

## Whole runs and the batch poll loop (`cdt.run`)

### Module shape and the `pipeline-writer` lease

A single lease serializes every writer of extract job state and the
match/final snapshots. A poll tick that runs past its hour, or an EventBridge
retry, cannot overlap the next tick. `daily`'s match/finalize cannot interleave
with a completing poll's. `historical` has the same shape as `daily`: with the
batch backend it prepares its date range and the next poll tick claims the
pending partitions.

### `run_prepare_then_publish`

The prepare stages rewrite the same completion registries that a poll tick's
finalize does, so the whole run holds the lease, not just match/finalize (#88).
After prepare, the run publishes from whatever mentions already exist, and the
in-flight batch job publishes again when it completes. Prepare can outlast the
TTL, so the run renews before match. If the lease was stolen, the snapshots
belong to another run.

### `LEASE_WAIT_SECONDS`

Daily and historical runs must not silently skip a day's work because an
hourly poll tick briefly held the lease. A normal tick takes minutes, so a
bounded wait absorbs it. A stuck holder outlasts the wait, and the run fails
loudly (#88). A poll tick does not wait: it skips its turn and prints `locked`,
and the next scheduled tick picks the job up.

### `run_poll` (renew at phase boundaries)

The lease is renewed at the tick's phase boundaries, so a long fold or submit
is not stolen by the next hourly run. The hook raises `LeaseLostError` if the
lease was already stolen, which aborts the tick (#89).

### `--max-rows-per-job`

This caps the rows claimed into one job, so a job after a backfill cannot run
the poll task out of memory. Deferred partitions form the next job (#92).

### The force backlog (`extract-batches/force-backlog.json`)

`--force` on a poll tick lists every classification partition into a backlog
file. Each new job claims backlog partitions as forced (no rows counted done)
and removes the ones it claimed, so a forced re-extract reaches every partition
even when `--max-rows-per-job` splits it across jobs, and a force given while a
job is active waits for that job instead of being dropped (#264). The backlog
is updated after the new job's marker is written: a crash in between re-forces
that job's partitions later rather than losing them from the request.

### Infrastructure failures in a batch job (deferred rows)

A provider 429/5xx on one request, or an output file that cannot be read, says
nothing about the filing, so it never becomes a verdict. Treating either as an
`ERROR` row (as the resubmission cap once did) marked the partition complete
and lost the row for good, while the live backend left the same row pending
(#264). An unreadable output file keeps its batch in flight, retried each tick
for `MAX_RESULT_DOWNLOAD_TICKS`: the results exist, and resubmitting pays for
them again. A row's 429/5xx rounds count separately from expiries. At
`DEFAULT_MAX_INFRASTRUCTURE_ROUNDS` the row is deferred: the job finishes
without it, its partition's registry entry is written incomplete, and the next
job claims just that row. This is the batch form of the live backend's abort,
except that one bad request does not hold up the rest of the job.

### `MODE_DEADLINE_HOURS` / `start_runtime_watchdog`

ECS has no task-level timeout: a Fargate task runs, and bills, until its
process exits. A run stuck past every client timeout must therefore end itself
(#93). Deadlines are per mode. A poll tick takes minutes, and its 2h matches
the lease TTL. Daily gets 12h, well inside its 24h cadence. Historical
backfills can legitimately run long, and get 72h. The CLI arms the watchdog in
`cdt run` only; a stage command has no deadline.

`os._exit` deliberately skips `finally` blocks. Every artifact write is already
crash-safe, and an unreleased lease is exactly the crash case the TTL/steal
design recovers from. A clean shutdown of a stuck process is not possible
anyway.

### `WATCHDOG_EXIT_CODE`

70 is EX_SOFTWARE. It separates a watchdog exit from an ordinary crash in the
task's stoppedReason. Any nonzero exit trips the task-failure alarm (#85).

### `RUN_COMPLETE_MESSAGE` / `POLL_TICK_COMPLETE_MESSAGE` (heartbeat log lines)

The `Run complete: mode=...` and `Poll tick complete: ...` lines
feed the daily-heartbeat and poll-liveness CloudWatch alarms (#85). A day with
no `mode=daily` completion means the run crashed, got stuck or never launched.
Keep the literals in sync with `pulumi/infra/alerts.py`.

### `reject_placeholder_secrets`

Without this check, a task launched before the one-time
`aws ssm put-parameter` would run a full ingest/segment/classify and then die
on a provider 401 that looks like a revoked key.

## CLI (`cdt.cli`)

### One command line, commands named after modules

`cdt` is the only console script. Each stage command is named after the module
it runs (`ingest`, `segment`, `classify`, `extract`, `match`, `publish`), and
`cdt run daily|historical|poll` runs the whole pipeline the way ECS does. A
separate deployment entry point and a separate `cdt pipeline` command both
called the same pipeline code but had drifted apart in their defaults, lease
handling, flags and watchdog; one parser removes the second place for those to
live. `cdt run historical --extractor-backend live` is the synchronous
end-to-end run.

### `main` (`configure_s3_profile`, `configure_logging`)

The S3 profile is set once per process, before any other credentialed call, so
artifact reads, writes and the lease all use `--aws-profile` and not the
ambient credentials (#71). Every command takes the flag. Logging is configured
on the root logger, and `cdt.shared.get_logger` routes every module's logger
through it, so `--quiet` and `--log-file` reach every module and a run has one
log format.

### `ENVIRONMENT_DEFAULTS`

Every option that has an environment default reads it in one place, so a flag
means the same thing on every command: the flag, then its variable, then the
built-in default. This is what lets ECS pass only `run daily` and configure the
rest through the task definition's environment.

### `_common_options(suppress=True)`

A nested command (`classify train`, `extract job show`) repeats its parent's
shared options with suppressed defaults. Otherwise the nested parser's default
would overwrite a value given before the nested command's name.

### `_final_database_root_option`

Only commands that publish accept `--final-database-root`. A stage command that
accepted the flag and ignored it would mislead anyone redirecting final output
for a single stage run (#72).

### `_with_writer_lease`

Stage commands rewrite the same completion registries and datasets as the
scheduled runs. Writers that are not serialized lose registry updates and
strand partitions (#88).

### `run_match` (lineage always)

An amend-and-restate chain spans filings, so its links exist only after every
shard has matched (#170, #204). Every inferred pointer is re-derived, never
carried over.

## Ingest (`cdt.ingest`)

### `run_ingest_pipeline` (`candidate_source`)

`candidate_source` is a factory, not a source, because the failure registry is
created inside ingest: two registries over one `failures.json` would overwrite
each other's entries. Everything after the candidates is shared: accession
dedup, batched partition merges, the read-back window and the run manifest.
That makes switching sources cheap. The S3 client is built only when needed,
so a run that never touches S3 does not need a profile.

### `run_ingest_pipeline` (`return_documents`, read-back)

Only the row count is read back unless the caller asks for the documents.
Building the frame meant deserializing every body in the window, including
`text`, one partition at a time; on a historical run that is the whole corpus
(#224). No production caller uses the frame.

The read-back covers the run's date window plus any partitions the run wrote
outside it, not the whole dataset. Reading the whole dataset deserialized every
historical 8-K body a second time on every run (#69). The union matters for a
source whose candidates can be dated outside the window. On the scraper-manifest
path it adds nothing, because candidates come from per-filing-date prefixes. No
row-level date filter is needed: `_write_document_partitions` groups on the
date column, so every row in `date=D/shard=S` has date D.

### `run_ingest_pipeline` (`--force` and the failure registry)

`--force` retries even failures registered as permanent. Otherwise a variant
document label or a since-fixed scraper bug would block a filing forever, and
hand-editing `failures.json` would be the only fix (#67). A registered failure
that does not reproduce is removed, so normal runs stop skipping that filing.

### `_existing_accessions` (the dedup window)

This used to read the whole `documents` dataset, projected to
`accession_number`. On S3, projection saves only deserialization: `read_table`
GETs the entire object before pandas applies `columns=`. Measured on
`data/genwindow-eval-apr`, `documents` is 12.2 GB across 1,640 partitions at
1.345 MB/row, almost all of it the `text` column. The accession numbers alone
are 0.2988 MB of compressed column chunks. Every ingest, even one with nothing
to do, moved the whole corpus to build a set of 9,077 strings (#190).

The window `[start_date, end_date]` is exact. A row is written to the partition
for its `candidate.date`, which is the manifest's `filing_date`, which is also
the day prefix `_iter_manifest_keys` found it under. So every accession the
run can be offered is stored under a date inside the window.

The exception is a manifest whose `filing_date` changed between runs (a scraper
repair). The stored copy is then under the old date, outside the window, and
this set misses it. The cost is bounded: the document is re-downloaded and
rewritten under its new date, because this set is an optimization, not the
uniqueness guarantee. Uniqueness within a partition comes from
`_write_document_partitions`, which merges and then runs
`drop_duplicates(subset=["accession_number"], keep="last")`.

The set is built with one parallel `read_partitions` scan, not a `read_table`
per partition. Measured on `data/genwindow-eval-apr/documents`, 1,640
partitions took 26.45 s one at a time and 1.75 s as one scan, for the same
9,077 accessions. The daily five-day window bounds the loop, but `cdt ingest`
given only `--end-date` starts at 1994, which is the whole corpus.

### `_write_document_partitions`

Each partition holds exactly one file at a known path, so it is read directly.
Listing the dataset to find it would cost a full LIST per group per flush.

### `_document_shard`

Python's built-in `hash` is salted per process, so the same accession used to
land in different shards across runs. A forced re-ingest then wrote a second
copy into a new partition that per-partition dedup could never see (#61). The
crc32 scheme in `datasets.shard_label` pins the assignment.

### `IngestConfig.dataset_name` / `SIXK_DOCUMENT_DATASET_NAME`

Each genre gets its own documents dataset. Mixing forms in one dataset would
merge new rows into partitions the 8-K path has already processed. Every
downstream stage selects work by source-partition fingerprint (#62), so a 6-K
backfill would make the whole 8-K corpus pending again. The name
`documents-sixk` matches the genre modules, `*.sixk`. The partition contract
reads a path's date and shard, never its dataset segment, so the name has no
effect on behaviour.

### `SIXK_FORM_TYPES` / `DEFAULT_FORM_TYPES`

Ingest itself does not care about form type. `DEFAULT_FORM_TYPES` is the 8-K
genre's forms and `SIXK_FORM_TYPES` the 6-K genre's; which genres a run covers
is chosen by `--genres` (default every genre), and each genre's ingest narrows
the config to its own forms (`ingest.genres.genre_config`). Both form tuples are
defined in `cdt.datasets` with the genre registry, so an `IngestConfig` built in
code without forms defaults to 8-K.

### `DOCUMENT_COLUMNS` (`form_type`, `source`)

Both columns are provenance, not keys. `source` is recorded per row, so a row's
origin survives any change in how a genre is acquired and is never inferred
from when the row was written.

### `MANIFEST_KEY_CIK_INDEX_FROM_END`

The CIK is counted from the end of the key, so a multi-segment `--s3-prefix`
cannot shift it (#73).

### `_candidate_from_filing` (`form_type`)

The candidate takes the manifest's `form_type`, not the key prefix. The prefix
spells "8-K/A" as "8-K_A", and turning it back would mean guessing which
underscore was a slash.

### `filing_from_manifest_key`

Both genres share this function. An unreadable manifest, an unparseable one
and one the scraper marked failed mean the same thing for any form, and one
implementation keeps them classified the same way in `failures.json`.

### `IngestFailureClassifier.do_not_retry` (`MALFORMED_DOCUMENT`)

Reading the same object again returns the same bytes, so a document that is
not in dissemination format never becomes one on retry.

## Segmenting 8-K text (`cdt.segmenter`)

### `VALID_ITEM_NUMBERS` / `leading_item_numbers`

Only real 8-K item numbers can be headings. Even those are rejected when they
read as money or a rate: `$1.05 billion` in a heading line, or a coupon such as
`5.25%`. HTML table cells become their own lines, so `5.25% Senior Notes due
2029` used to be read as a heading and cut the enclosing item section off right
at the debt text this pipeline targets (#63).

### `extract_items_from_document` (one row per key)

`item_id` is accession plus item number. A header that repeats an
`ITEM INFORMATION` line (SEC headers do this), or two labels that map to one
number, would produce duplicate primary keys and break downstream joins (#74).
The first occurrence of each key is kept.

### `ITEM_DOCUMENT_COLUMNS`

These columns are fixed in the 8-K segmenter, not taken from
`ingest.DOCUMENT_COLUMNS`. Items, classifications, mentions and the published
items snapshot all take their schema from this list. If it were derived, a
column added to the documents dataset would reshape four datasets and the
dashboard's contract as a side effect of an ingest change.

### `document_text_for_record` / `ensure_s3_client`

These are public because both genres resolve a documents row the same way. A
row carries either inline text (from `download=True` ingest) or a
`resource_uri` pointing at the stored submission: the scraper's copy for 8-K,
CDT's own mirror for 6-K. A local run with mirrored bodies must not need AWS
credentials to resolve them.

### `run_partition_stage` (fingerprint selection and checkpoints)

`cdt.partition_stage.run_partition_stage` is the one loop behind the four
whole-partition stages: 8-K and 6-K segment, 8-K and 6-K classify. A source
partition that changed (ingest merged new rows into a documents partition, or
segment rewrote a windows partition) becomes pending again and is recomputed
whole. Segmenting and the 8-K classifier are cheap and deterministic, so
row-level diffing is not worth the complexity (#62). A stage can report a
partition incomplete (`PartitionOutput.complete`), which writes nothing and
records no completion, so the next run retries it.

Completion is checkpointed by time: saved at most every
`CHECKPOINT_INTERVAL_SECONDS` (300 s) and once at the end. When the registry was
written only at stage end, any interruption threw away the whole run's progress:
up to 2.5 h of itemizing on the real corpus (#111). A count of partitions is the
wrong unit for the interval, because what a partition costs varies by stage and
will vary by partition layout. A time interval bounds the loss directly. A save
is a few S3 requests, so at 300 s it costs nothing worth measuring. Repeated
saves are safe under concurrency, because only dirty entries are merged, by
compare-and-swap (#88). The lease is renewed after every partition, throttled
to `RENEW_INTERVAL_SECONDS`, because a stage that outlasts the TTL would
otherwise be stolen mid-run.

## The classifier (`cdt.classifier`)

### `classify_items` (`artifacts`) / `classify_pending_items`

The model is unpickled once per run and passed to each partition as
`artifacts`. Loading it per partition dominated large backfills with repeated
deserialization and S3 GETs (#76).

### `load_training_artifacts` (scikit-learn version check)

A pickle trained under a different scikit-learn minor version is refused. The
model is the relevance gate that decides which items reach the LLM, and its
threshold is calibrated against the training version's scoring. scikit-learn
only warns on a mismatch, and a drift of a few percent in relevance would look
like normal variation in corpus size (#109).
