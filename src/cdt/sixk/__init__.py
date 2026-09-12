"""Form 6-K triage: window a filing, then select snippets worth extracting.

The 8-K path classifies numbered items. A 6-K has no item structure, so this
path windows the document instead and runs two stages over the windows: a cheap
recall-oriented classifier, then an LLM that sees a whole filing's admitted
windows at once and prunes them. Between the two, an admitted window is expanded
backwards into the context the 400-token crop cut off, which stage 1 must not
see: its threshold is calibrated on the crop.

Developed and evaluated in ``uchicago-dsi/commercial-debt-tracker-models``; see
``docs/sixk-two-stage-triage.md`` for the measurements.
"""

from cdt.sixk.triage import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_STAGE1_THRESHOLD,
    DEFAULT_STAGE2_MODEL,
    DEFAULT_STAGE2_REASONING,
    SYSTEM_PROMPT,
    FilingVerdict,
    Snippet,
    build_retry_message,
    build_snippet_message,
    default_model_dir,
    load_stage1_model,
    stage1_admit,
    triage_filing,
    validate_verdict,
)
from cdt.sixk.windows import (
    DEBT_KEYWORDS,
    MAX_EXPANSION_TOKENS,
    MIN_EXPANSION_TOKENS,
    WINDOW_TOKENS,
    ExpandedWindow,
    TextWindow,
    count_tokens,
    expand_admitted_windows,
    has_debt_keyword,
    matched_debt_keywords,
    prepare_filing,
    split_into_windows,
    strip_inline_xbrl_prologue,
)

__all__ = [
    "DEBT_KEYWORDS",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_STAGE1_THRESHOLD",
    "DEFAULT_STAGE2_MODEL",
    "DEFAULT_STAGE2_REASONING",
    "MAX_EXPANSION_TOKENS",
    "MIN_EXPANSION_TOKENS",
    "SYSTEM_PROMPT",
    "ExpandedWindow",
    "FilingVerdict",
    "Snippet",
    "TextWindow",
    "WINDOW_TOKENS",
    "build_retry_message",
    "build_snippet_message",
    "count_tokens",
    "expand_admitted_windows",
    "default_model_dir",
    "has_debt_keyword",
    "load_stage1_model",
    "matched_debt_keywords",
    "prepare_filing",
    "split_into_windows",
    "stage1_admit",
    "strip_inline_xbrl_prologue",
    "triage_filing",
    "validate_verdict",
]
