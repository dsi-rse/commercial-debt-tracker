"""Settings for the Commercial Debt Tracker project."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
# The committed model artifacts. Not under DATA_DIR: DATA_DIR is where a
# developer's artifacts live, and the models are part of the code.
MODELS_DIR = PROJECT_ROOT / "data" / "models"


def resolve_path(path: Path) -> Path:
    """Return ``path`` as an absolute, resolved path, expanding ``~``.

    A relative path is taken relative to the project root.
    """
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


load_dotenv()

DATA_DIR = resolve_path(Path(os.environ.get("DATA_DIR", str(DEFAULT_DATA_DIR))))
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
    "OPENROUTER_API_TOKEN"
)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# The extractor model id, as an OpenRouter slug; the batch backend strips the
# provider prefix (``native_model_id``), so one value serves both backends. Keep
# it undated: OpenRouter's dated alias normalizes to an id the OpenAI API
# rejects with a 400.
DEFAULT_EXTRACTOR_MODEL = "openai/gpt-5.6-terra"
EXTRACTOR_MODEL = os.environ.get("EXTRACTOR_MODEL") or DEFAULT_EXTRACTOR_MODEL
EXTRACTOR_REASONING = os.environ.get("EXTRACTOR_REASONING", "none")
# The batch backend's model (default: the live model) and its own reasoning
# knob, since the OpenAI Batch API's reasoning_effort vocabulary differs.
EXTRACTOR_BATCH_MODEL = os.environ.get("EXTRACTOR_BATCH_MODEL") or EXTRACTOR_MODEL
EXTRACTOR_BATCH_REASONING = os.environ.get("EXTRACTOR_BATCH_REASONING", "none")
# Model id for the 6-K stage-2 triage, as an OpenRouter slug. Separate from the
# extractor's: triage is priced for volume, extraction for accuracy.
DEFAULT_SIXK_TRIAGE_MODEL = "openai/gpt-5.6-luna"
SIXK_TRIAGE_MODEL = os.environ.get("SIXK_TRIAGE_MODEL") or DEFAULT_SIXK_TRIAGE_MODEL
DEFAULT_SIXK_TRIAGE_REASONING = "none"
SIXK_TRIAGE_REASONING = (
    os.environ.get("SIXK_TRIAGE_REASONING") or DEFAULT_SIXK_TRIAGE_REASONING
)
# Which API the stage-2 triage call goes to: "openrouter" (default) or "openai".
# "openai" is the fallback when OpenRouter credit runs short; OpenRouter reserves
# a maximum cost per in-flight request, so this high-volume stage fails first.
DEFAULT_SIXK_TRIAGE_PROVIDER = "openrouter"
SIXK_TRIAGE_PROVIDER = (
    os.environ.get("SIXK_TRIAGE_PROVIDER") or DEFAULT_SIXK_TRIAGE_PROVIDER
)
