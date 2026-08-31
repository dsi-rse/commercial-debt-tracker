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
DEFAULT_EXTRACTOR_MODEL = "openai/gpt-5.4"
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
