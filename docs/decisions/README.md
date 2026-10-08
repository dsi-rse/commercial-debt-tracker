# Design decisions

These documents record why the code is built the way it is: the reasoning behind design choices, and the evaluations and measurements that justified them. Module docstrings describe what the code does; these describe why.

**By area** (headed by the module, function or constant each section explains):

- [storage-and-completion.md](storage-and-completion.md): the storage layer, the completion registry, the writer lease and settings.
- [extraction.md](extraction.md): the extractor's stages, validation and normalization rules, salvage, provider aborts and prior-state minting.
- [matching-and-lineage.md](matching-and-lineage.md): matcher scoring, name compatibility, the lifecycle rollup and amendment-lineage inference.
- [pipeline-ingest-and-publish.md](pipeline-ingest-and-publish.md): the pipeline and its whole runs (`cdt run`), the batch poll loop, ingest, segmenting, the classifier and snapshot publishing.
- The 6-K triage design and its evaluation are in [../sixk-two-stage-triage.md](../sixk-two-stage-triage.md).

**Dated decision records** (a snapshot of a design at the time it was decided, not a description of the current code):

- [2026-08-completion-keyed-on-row-outcomes.md](2026-08-completion-keyed-on-row-outcomes.md)
- [2026-09-wiring-the-6k-path.md](2026-09-wiring-the-6k-path.md)
