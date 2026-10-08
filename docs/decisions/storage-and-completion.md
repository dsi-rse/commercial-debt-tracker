# Storage, completion registry, leases and settings: design decisions

This page records why `cdt.storage`, `cdt.datasets`, `cdt.lease`,
`cdt.settings` and `cdt.shared` are built the way they are: the trade-offs
behind each design and the measurements that settled them. Docstrings in the
code give each function's current contract; this page is the reasoning behind
it. Each section is headed by the name it explains, so you can search for a
name from the code. Measurements name the data root they were taken on; rerun
them before relying on them for a new decision.

## `cdt.storage` (`objects.py`, `tables.py`, `columns.py`)

### `configure_s3_profile`, `s3_client`, `boto3_session`

The AWS profile is chosen in one place for the whole process. Before this,
`--aws-profile` was passed by hand to whichever client factory ingest happened
to call. A run against a non-default account then read manifests with the
right credentials and wrote artifacts with the wrong ones, or failed on the
first write with no sign that the flag had been ignored (#71).
`boto3_session` exists separately from `s3_client` because the Arrow read path
needs a second credentialed object (`pyarrow.fs.S3FileSystem`) built from the
same profile. With one Session per profile, both objects get their credentials
from a single resolution.

Clients and Sessions are memoized per profile. Building one resolves
credentials and discovers endpoints, and a partition scan makes thousands of
S3 calls per run (#83).

### `S3_CLIENT_CONFIG`, `_STREAMING_READ_ERRORS`, `_get_object_with_body`

Before `S3_CLIENT_CONFIG` was set explicitly, every client used botocore's
legacy retry mode (2 attempts) and had no timeout on a stalled socket. That
config only covers the API call itself. Once `get_object` has returned,
botocore does not retry failures in the streaming body read, and a single
mid-stream timeout killed a 2.5 h itemize run (#112). A stream cannot be
resumed partway through, so `_get_object_with_body` retries by re-issuing the
whole GET.

### `artifact_content_versions` vs `list_artifacts_with_versions`

There are two version functions because they answer different questions.
`list_artifacts_with_versions` uses mtime locally, so rewriting a file with
identical bytes counts as a change. That is the right trade for the question
"does this partition need reprocessing?", because a false positive costs only
one redundant partition. It is the wrong trade for "may we skip re-reading the
whole corpus?". The matcher rewrites every shard on every run, usually with
byte-identical content, so an mtime-based check would differ every time and
the skip would never fire. A guard that silently always says "changed" is the
failure `artifact_content_versions` was written to replace. Locally it hashes
the bytes, which is cheaper than the parquet decode the skip can avoid. On S3
it costs nothing extra, because the ETag of a single-part `put_object` (which
is what `write_table` issues) is already the content MD5.

### `DECLARED_COLUMN_TYPES`, `declared_column_type`

Column types are declared, not inferred (#187). `pa.Table.from_pandas` infers
an object column's type from its values. A column with no values in one
partition was therefore written as parquet `null`, and as `string` in the
next. In `debt-instruments`, 23 of 42 columns did this. As a result,
`pyarrow.dataset`, `pq.read_table`, `ParquetDataset` and `pandas.read_parquet`
all failed on the directory with "Unsupported cast from string to null". The
problem went unnoticed because only this module's own per-file pandas
`read_dataset` could read the directory. An empty frame is worse still: every
column is inferred as `null`, including counts and flags.

Types are keyed by column name, which keeps `write_table` generic. That works
because a column name (`cik`, for example) means the same thing in every
dataset.

An undeclared column is text, and two inferred types are mapped to that
default: `null`, and floating where every value is null (#268). The second is
what a pandas `reindex` fills a missing column with. `synthesized_by` is null
on every model-emitted mention, so a mentions partition with no synthesized
row wrote it as `double` and the next as `string`, unification failed, and
every match read fell back to per-file. Every real float column (the model
scores) is declared, so the rule cannot retype one. Undeclared integer
columns are pandas `Int64`, which keeps `int64` when all-null.

Money and rates are stored as exact decimals (#185). Float is not an option:
`float("372246148.11")` is not exactly that number, and printing it at fixed
precision exposes the error. That error is what made every amount with cents
publish as null (#119). `interest_rate_pct` uses scale 4, which holds basis
points with room to spare; the corpus uses at most three decimal places.

### `decimal_column_values`

This function quantizes values in Python instead of casting through Arrow. A
partition written before #119 stores float error in its text, such as
`372246148.110000014305`. Arrow rejects that rescale because it would lose
data. But the extra digits are the error, not the value, so dropping them is
correct, and a replay must not fail on them. Placeholder text (`nan`, the
empty string) must also become null rather than raise an Arrow error.

### `apply_declared_column_types`

Declared decimal columns are converted to canonical text before Arrow sees
them. Rewriting an existing partition mixes rows read from parquet (which
carry `Decimal`) with rows built in memory (which carry the parser's text).
`Table.from_pandas` rejects an object column holding two Python types before
the decimal path can quantize either one (#203).

### `write_gzip_text_artifact`

The extract job state holds full item text and message histories, and the
pipeline rewrites it many times. Gzip makes that object roughly ten times
smaller (#86).

### Arrow read path, `_ARROW_READ_ERRORS`

Reads used to GET the whole object, wrap it in a `BytesIO`, and then pass a
`columns=` list to pandas. The column list saved deserialization time but not
a single byte of transfer (#190). Measured on `data/genwindow-eval-apr`: the
`documents` dataset is 12,206.6 MB over 1,640 partitions, at 1.345 MB per row.
Its `accession_number` column is 0.2988 MB of compressed column chunks, so the
projection a dedup scan wants is 0.0024% of the bytes the old path moved
(40,856x less). Reading through a pyarrow filesystem turns the projection into
ranged GETs.

`pa.ArrowException` is the common base of `ArrowInvalid`, `ArrowTypeError` and
`ArrowNotImplementedError`. Those are the three ways that partitions with
different physical types break a dataset scan, and all three get the same
answer (the per-file fallback), so the code catches the base class.
`FileNotFoundError` is caught as well. It is an `OSError`, not an
`ArrowException`, and before it was caught, a partition deleted between the
listing and the scan made the whole read fail, while the per-file loop would
have returned an empty frame and carried on. pyarrow's `ArrowIOError`, which
is how auth and transport failures surface, is deliberately left uncaught.
Those failures must stay fatal, not turn into a slow fallback that is silent
and gives the wrong reason.

### `_SCHEMA_WORKERS`, `_unified_schema`

`ds.dataset` takes its schema from the first fragment alone. If the first file
was written before a column existed, the dataset reads as though that column
never existed, and its later values are silently dropped. This pipeline has
that shape on purpose (`form_type` and `source` were added to `documents`
later), so the explicit schema merge is needed for correctness, not just
speed.

Each `fragment.physical_schema` call opens the file and reads its footer, so
the merge costs one round trip per partition. These reads happen before the
scanner exists, so the scanner's own threads do not cover them. Measured on
`data/genwindow-eval-apr/items` (1,554 files, warm cache, local disk): 0.523 s
serially vs 0.184 s threaded, and the `unify_schemas` call itself takes
0.017 s. Locally the difference is lost in the noise. On S3 every read is a
sequential round trip: the publish step reads 21,214 objects, so a serial pass
is 21,214 round trips at about 70 ms each before any data is read. That serial
pass alone costs more than the parallel scan that follows it. The worker count
of 32 follows the sizing in #110 (option 1, "parallelize the reads"). Arrow
releases the GIL during the read, so the threads really do overlap.

### `_CREDENTIAL_LIFETIME_WARN_SECONDS`, `_warn_on_short_credential_lifetime`

`S3FileSystem` is given a fixed key triple. Unlike a boto3 client, it cannot
re-sign requests when a temporary credential rolls over in the middle of a
scan. `get_frozen_credentials` refreshes, so each new filesystem starts with
fresh keys, but a full-corpus read can run for minutes to hours. When the keys
expire, every later request fails with pyarrow's opaque
`AWS Error UNKNOWN (HTTP status 400)`, which does not look like an expiry. The
warning does not extend the credentials; it names the likely cause in advance.
The threshold is 900 s because that is botocore's advisory refresh window, and
so the most a freshly frozen credential is guaranteed to last.

### `arrow_filesystem`, `strip_s3_scheme`

Local paths get `None`. pyarrow resolves local paths itself, and passing a
`LocalFileSystem` through would only add one more way to get it wrong.

The S3 filesystem is built fresh on each call, not memoized, because a
Session's frozen credentials can be temporary (SSO, assume-role) and a
historical backfill outlives them.

Timeouts and retries are set to match `S3_CLIENT_CONFIG`. pyarrow's defaults
are `connect_timeout=-1` and `request_timeout=-1` (no limit) with 3 attempts.
Left at those defaults, the path that now does nearly all the reading would
lose both halves of the #112 fix.

The function raises when no credentials resolve. With no keys,
`S3FileSystem()` is not anonymous: it resolves credentials through its own
chain. That chain can disagree with boto3's, because botocore ignores
environment credentials when a profile is set explicitly, while pyarrow checks
the environment first. Reading as one identity while writing as another is
exactly the #71 failure.

`strip_s3_scheme` is separate from `arrow_filesystem` because turning a list
of paths into Arrow form only needs the stripped path. Calling
`arrow_filesystem` once per path built and threw away one `S3FileSystem` per
partition, each with its own SDK client and connection pool.

### `read_table`

`columns` used to shape only the empty fallback: both real branches still
deserialized every column (#69). Once projection worked, the S3 branch still
downloaded the whole object first (#190). The absent-column case is decided
from the footer schema, not by catching an exception, because
`ParquetFile.read` does not raise for a missing column. It silently returns
the columns the file does have, so relying on an exception dropped exactly the
column this behaviour exists to keep.

There is no existence check before the open: pyarrow's open already sends a
HEAD to size the object, so a separate `artifact_exists` check would double
the requests for every read of a partition that exists.

### `count_table_rows`, `count_partition_rows`

The row count lives in the footer alone. The S3 branch used to GET the whole
object just to read its last few KB, which on `documents` meant moving
1.345 MB per row to learn one integer (#190). The publish guard compares
against this count, so it runs once per final table on every publish.

`count_partition_rows` exists because ingest's read-back used to load every
partition in its window, `text` column included, only to call `len()` on the
result (#224). Measured on `data/genwindow-eval-apr`: 65.44 s vs 0.67 s for the
same 9,077 rows. It uses threads for the same reason as `_unified_schema`.

### `_read_dataset_with_arrow`

The scan runs in parallel, which makes it worth having as a separate path from
looping over `read_table`. Measured on the 1,554-file `items` dataset in
`data/genwindow-eval-apr`, a full read takes 4.836 s looping pandas, 2.348 s
looping `pq.read_table` and concatenating, and 0.599 s as one dataset. The
gain comes from the scanner's threads, not from Arrow's decoding.

The function returns its exception alongside the result. Reporting every
fallback as "pre-#187 types" was a confident but wrong diagnosis when the real
cause was a corrupt or missing partition.

### `read_dataset`, `read_partitions` (per-file fallback)

The per-file pandas fallback is kept rather than removed for two reasons.
First, partitions written before #187 still have per-partition physical types,
and the production root has not been rebuilt (#107). On
`data/lineage-probe/debt-instruments`, 26 of 41 columns still vary by
partition, and unification fails with
`ArrowTypeError: Unable to merge: Field amendment_inferred_by has incompatible types: double vs string`.
Making the Arrow path mandatory would mean the rebuild has to happen first.
Second, it is a permanent safety net: #187 fixes the types of declared columns
only, so an undeclared object column can still infer a different type in
different partitions. A partition can also disappear between the listing and
the scan, because the snapshot prune deletes live partitions, and the per-file
path handles that case correctly.

`read_partitions` exists for callers that already know their partitions.
`ingest._existing_accessions` limits itself to a date window. Without
`read_partitions` it would loop over `read_table` and lose the parallel scan:
26.45 s vs 1.75 s over 1,640 partitions.

### `_ORPHANED_TEMP_RE`, `write_table` temp suffix

Before the current temp naming, `NamedTemporaryFile` gave temp files a
`.parquet` suffix. A crash between creating the file and renaming it left an
empty `tmp*.parquet` inside a partition directory, and every `**/*.parquet`
reader failed on it (#68). `write_table` now uses `.parquet.tmp`. Readers
still skip the old pattern, because existing roots may contain such files.

## `completion.py` (the registry) and `datasets.py` (paths and shards)

### `completion_registry_root`, `_REGISTRY_SHARD_DATE_CHARS` (registry sharding)

The registry is a prefix of year-month shards, not one object (#191). It used
to be a single `runs/<stage>/completed-partitions.json`, which every batch
boundary read and rewrote in full under compare-and-swap. At full corpus scale
(8,533 business days x about 52 occupied shards = 440,000 entries), that
object serializes to 56.8 MB, at 129 B per entry. (The real registry on
`data/genwindow-eval-apr` measures 146 B per entry over 1,640 entries.) One
compare-and-swap cycle moves 113.5 MB and spends 3.2 s in JSON, and one
itemize pass runs 4,400 cycles: 499 GB moved and 3.9 h of JSON CPU before any
real work happens.

Sharding works because of locality. Stages walk pending partitions in sorted
path order, which is date order, so one 100-partition chunk covers one or two
consecutive dates, and therefore one or two shards. Each cycle then touches
about 0.2 to 0.5 MB.

The legacy single object is neither read nor migrated: nothing in that format
is kept during beta. On S3 the shard prefix also matches the legacy key, so
the loader reads only `date=` files.

Naming: in this module, prefixes are `*_root` (`dataset_root`, `items_root`,
`mentions_root`, `mirror_root`) and single objects are `*_path`
(`run_manifest_path`, `failure_registry_path`, `active_job_path`,
`final_pointer_path`). `completion_registry_root` and
`completion_registry_shard_path` follow that rule (#227).

### `"completion_registry"` in run manifests

The five run manifests' `"completion_registry"` key names the registry's
directory (`completion_registry_root`), while the neighbouring `"audit_path"`
and `"failure_registry"` keys name files. The key keeps its name so existing
manifests stay comparable.

### `load_completion_registry`, `_REGISTRY_LOAD_CONCURRENCY`

Loading is the only operation that reads every shard. It runs once or twice
per run, against the 4,400 saves per itemize pass that sharding was built
for. The shard reads run concurrently, because sharding turned one GET into
one GET per occupied month. A serial loop over them would repeat #110's
problem in a new place (#227). At full-corpus shape (393 shards), a load is
one LIST plus 393 GETs. Serially, at the 70 ms round trip #110 measured on the
same 1-vCPU stack, that is 27.5 s per load. A pipeline run does five loads,
and the shard count grows by one every month. (#191 said the old design
"cannot be fixed by parallelising reads, because it is one object"; that
stopped being true once there were 393 objects.) Merge order is unchanged:
`map` returns results in submission order, and `list_artifacts` returns sorted
paths.

Concurrency is capped at 10, botocore's default `max_pool_connections`. All
the GETs share one cached client, and threads beyond the pool size only queue
and log "connection pool is full".

### `_relative_registry_key` (root-relative keys)

Keys used to store the whole path, root included. That made each entry larger
(129 B vs 105 B, measured over 440,000 full-corpus-shaped entries). Less
obviously, it also made an artifact root non-portable. Copy a root, and every
key still carries a prefix that no longer exists, so the copy's registry
matches nothing and the whole corpus reads as unprocessed: #107's failure,
arriving through `cp -r`. Keys are relativized only when they will read back
through `_absolute_registry_key`. That keeps the two functions exact inverses,
and the tests assert that property on this one-key pair.

### `_REGISTRY_PREFIX_PROBE` (hoisted key prefixes)

Both key prefixes are computed once per object read or written, not once per
key (#227). Computed per key, `join_artifact_path` built a `pathlib.Path` and
`_relative_registry_key` rebuilt a normalized prefix and an f-string, inside
loops over every entry. On a 440,000-entry registry (local disk, best of
three), that made the load 2.32 s, against 1.06 s for the old single object,
and `_absolute_registry_key` took 56% of the save time under cProfile. With
the prefixes hoisted, the load takes 1.08 s. That matches the old object on a
short root and beats it under a long absolute root (1.18 s vs 1.27 s), because
the per-key cost no longer grows with the root's length.

### `_registry_payload`

Entries are sorted on the relativized key alone. The collision that would make
a tuple sort raise cannot happen through `save_completion_registry` today,
because `_registry_entries` turns every stored key back into a whole path
first. But a free sort key is a better trade than a crash on a collision, and
`json.dumps(sort_keys=True)` orders the written bytes anyway.

### `save_completion_registry`

The registry has several concurrent writers (the daily run, poll ticks, manual
CLI runs; #88). A blind overwrite would lose every entry another writer saved
since this run loaded its copy, and losing some of those entries silently
strands partitions or marks them complete when they are not. So only the
dirty entries are written, overlaid on the freshest saved state.

Each shard's compare-and-swap is independent, which keeps the checkpointed
saves from #111 durable: if a save is interrupted after three of five shards,
those three stay saved. The segment and classify stages of both genres
checkpoint every `CHECKPOINT_INTERVAL_SECONDS` (they share
`cdt.partition_stage.run_partition_stage`). Live extract commits at every
partition end, and within a partition on the same interval. Each commit writes,
in order:
1. the mentions so far, pruning rows that left the source on every write. The
   entry the commit saves no longer names those rows, so no later run would
   prune them.
2. the audit records;
3. the failure registry;
4. the registry.

So the registry never marks a row done before its mentions, its audit record
and its failure record are on disk. An
incomplete entry carries the rows already terminal, so an interrupted run's
successor pays only for the rest.

The dirty set is cleared shard by shard. Without clearing, the dirty set would
grow to the whole run's write set, so batch k would rewrite every shard that
batches 1 to k-1 touched, and the cost would grow quadratically with run
length, which is the problem #191 removed.

### `CompletedPartition.item_ids`, `pending_source_partitions`

Partitions are tracked by source fingerprint, not by whether the target
exists, because ingest merges late rows into existing partition files in
place. A check on the target path alone would strand those rows forever (#62).
`item_ids` makes reprocessing row-level, so rows that already have a real
outcome are never paid for twice (#49).

### `existing_date_shard_partition_ids`

The alternative, one HeadObject per candidate target, costs one sequential
round trip for every partition ever written. Once the dataset is large, that
dominates stage runtime (#83).

### `shard_label`, `normalize_cik`, `shard_for_cik`

Changing the shard function strands every existing partition (#61), so it is
defined in one place. CIKs are published zero-padded so they join against
anything keyed on SEC's canonical CIKs (#153). Partitions written before
padding hashed the bare string (for example `707605`), so `shard_for_cik`
hashes the unpadded form.

## lease.py

### `release_lease`, `_EXPIRED`, `_log_takeover`

EventBridge starts the poll schedule every hour whether or not the previous
tick has finished. Both `daily` and `poll` rewrite the match and final
snapshots, so writers have to be serialized. Releasing a lease stamps it as
expired instead of deleting it, so in steady state the lease object exists
and is free. That means a normal handoff and the rescue of a holder that died
go through the same compare-and-swap. The exact `_EXPIRED` stamp tells them
apart. A handoff is logged at DEBUG, because it happens on every tick and
logging it at WARNING would bury real takeovers under about 24 false alarms a
day. The "Stole lease" warning feeds a CloudWatch metric-filter alarm (#85).

### `DEFAULT_LEASE_TTL_SECONDS`

Releases happen in a `finally` block, so the TTL only matters after a crash.
It is set well above a normal tick (2 h, where a tick is usually minutes).

### `renew_lease`, `renewer`, `LeaseLostError`

Folding a large job's results can take longer than the TTL. Renewing at phase
boundaries keeps a long but legitimate holder from being taken over by the
next scheduled run in the middle of a write. `renewer` raises instead of
returning a bool because a discarded `renew_lease` return value once let a run
keep writing on a lease another process had already taken (#89).

## settings.py

### `DEFAULT_EXTRACTOR_MODEL`

The model id is kept undated. OpenRouter's dated alias
(`openai/gpt-5.6-terra-20260709`) normalizes to `gpt-5.6-terra-20260709`, and
the OpenAI API rejects that id with a 400 on every request in a batch.

### `SIXK_TRIAGE_MODEL`, `SIXK_TRIAGE_PROVIDER`

Triage reads a lot of text and returns a list of ids, so its model is chosen
for cost at volume. Extraction returns structured records, so its model is
chosen for accuracy. The provider can be switched because the shared
OpenRouter account has hit its credit limit before. OpenRouter reserves an
estimated maximum cost for each in-flight request, so this stage, which keeps
many large requests in flight, is the first to fail. Switching provider turns
that into a settings change rather than a blocked run.

## shared.py

### `FailureRegistry.discard`

Without `discard`, a filing that succeeds on a `--force` retry stays
registered forever: every later normal run skips it, and `failures.json`
over-reports. Delete the subclass once `idi-ftm2j-shared` ships its own
discard method.
