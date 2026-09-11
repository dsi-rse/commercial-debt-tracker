"""Settings for the Commercial Debt Tracker project."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def resolve_path(path: Path) -> Path:
    """Resolve a path to an absolute path.

    If the path is not absolute, it is assumed to be relative to the project root.
    """
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


# Load environment variables from .env file if it exists
load_dotenv()

# Set the data directory
DATA_DIR = resolve_path(Path(os.environ.get("DATA_DIR", str(DEFAULT_DATA_DIR))))
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
    "OPENROUTER_API_TOKEN"
)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# Single source of truth for the extractor model id, as an OpenRouter slug. The
# batch backend strips the provider prefix (``normalize_batch_model``), so both
# backends stay on the same model when only this value changes.
#
# Keep the id undated. OpenRouter and the OpenAI API both expose this model
# undated, so one value serves both backends; OpenRouter's dated alias
# (`openai/gpt-5.6-terra-20260709`) normalizes to `gpt-5.6-terra-20260709`,
# which the OpenAI API rejects with a 400 on every request in a batch.
DEFAULT_EXTRACTOR_MODEL = "openai/gpt-5.6-terra"
EXTRACTOR_MODEL = os.environ.get("EXTRACTOR_MODEL") or DEFAULT_EXTRACTOR_MODEL
EXTRACTOR_REASONING = os.environ.get("EXTRACTOR_REASONING", "none")
# The OpenAI Batch API uses a reasoning_effort vocabulary distinct from
# OpenRouter's, so the batch backend gets its own reasoning knob. It defaults to
# the same model as the live backend.
EXTRACTOR_BATCH_MODEL = os.environ.get("EXTRACTOR_BATCH_MODEL") or EXTRACTOR_MODEL
EXTRACTOR_BATCH_REASONING = os.environ.get("EXTRACTOR_BATCH_REASONING", "none")
# Model id for the 6-K stage-2 triage, as an OpenRouter slug. Separate from the
# extractor's because the two jobs want opposite trade-offs: triage reads a lot
# of text and returns a list of ids, so it is priced for volume, while
# extraction returns structured records and is priced for accuracy.
DEFAULT_SIXK_TRIAGE_MODEL = "openai/gpt-5.6-luna"
SIXK_TRIAGE_MODEL = os.environ.get("SIXK_TRIAGE_MODEL") or DEFAULT_SIXK_TRIAGE_MODEL
DEFAULT_SIXK_TRIAGE_REASONING = "none"
SIXK_TRIAGE_REASONING = (
    os.environ.get("SIXK_TRIAGE_REASONING") or DEFAULT_SIXK_TRIAGE_REASONING
)
# Which API the stage-2 triage call goes to: "openrouter" (the default, matching
# the live extractor) or "openai". Stage 2 is priced for volume and the shared
# OpenRouter account has hit its credit limit before — and OpenRouter reserves
# an estimated maximum cost per in-flight request, so it fails first under
# exactly this stage's shape. This makes that a setting change, not a blocked
# run.
DEFAULT_SIXK_TRIAGE_PROVIDER = "openrouter"
SIXK_TRIAGE_PROVIDER = (
    os.environ.get("SIXK_TRIAGE_PROVIDER") or DEFAULT_SIXK_TRIAGE_PROVIDER
)
# SEC fair-access policy requires a declared contact in the User-Agent of every
# EDGAR request: "Sample Company Name AdminContact@example.com". Requests
# without one are answered with a 403 and an "Undeclared Automated Tool" page
# rather than the file, so cdt.sixk.edgar refuses to start without this set
# instead of persisting that page as filing text. Only the direct-EDGAR 6-K
# fallback reads it; nothing else in the pipeline talks to sec.gov.
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "")
