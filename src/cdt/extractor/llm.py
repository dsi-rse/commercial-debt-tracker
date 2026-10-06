"""Model access: the OpenRouter client, sampling parameters and response parsing for both backends."""

# ruff: noqa: ANN101, ANN102, D102, D105, D107

from __future__ import annotations

import json
from importlib import resources
from typing import cast

from cdt import settings
from cdt.extractor.state import CompletionResult

DEFAULT_MODEL = settings.DEFAULT_EXTRACTOR_MODEL


DEFAULT_REASONING_EFFORT = "none"


REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


EXTRACTOR_TEMPERATURE = 0.0


# Reasoning models take a reasoning_effort and reject temperature != 1, so both
# backends must decide sampling params the same way or the same model produces
# different output live versus in batch. Prefixes are matched against the native
# id, so both "gpt-5.4" and "openai/gpt-5.4" resolve identically.
REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


# Bounds every live chat call, so one non-responsive provider socket cannot
# wedge the synchronous run. Generous because reasoning models legitimately
# take minutes per response.
LIVE_REQUEST_TIMEOUT_SECONDS = 600


class OpenRouterChatClient:
    """Native OpenRouter client wrapper used by the extractor."""

    def __init__(self, *, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required for cdt extractor. "
                "OPENROUTER_API_TOKEN is also accepted as a compatibility alias."
            )

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        reasoning_effort: str,
    ) -> CompletionResult:
        """Run one OpenRouter chat completion."""
        from openrouter import OpenRouter

        request_kwargs: dict[str, object] = {
            "messages": messages,
            "model": model,
            "stream": False,
            **sampling_params(model),
        }
        if reasoning_effort:
            request_kwargs["reasoning"] = {"effort": reasoning_effort}

        async with OpenRouter(
            api_key=self.api_key,
            x_open_router_title="commercial-debt-tracker",
            x_open_router_categories="cli-agent",
            timeout_ms=LIVE_REQUEST_TIMEOUT_SECONDS * 1000,
        ) as client:
            response = await client.chat.send_async(**request_kwargs)
        return completion_result_from_response(response)


def is_content_filter_abort(completion: CompletionResult | None) -> bool:
    """Whether the provider aborted this response instead of the model ending it.

    An abort is remedied by resending the same request; a model stop by
    changing it. False for None.
    """
    return completion is not None and completion.finish_reason == "content_filter"


def load_prompt(name: str) -> str:
    """Load one extractor prompt from the local package."""
    return (
        resources.files("cdt.extractor.prompts")
        .joinpath(f"{name}.md")
        .read_text(encoding="utf-8")
    )


def extract_response_text(response: object) -> str:
    """Extract the assistant text from an OpenRouter SDK response."""
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("OpenRouter response did not include choices.")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise RuntimeError("OpenRouter response did not include a message.")
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            else:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    text_parts.append(text)
        return "".join(text_parts)
    raise RuntimeError("OpenRouter response content was not text.")


def extract_batch_response_text(line: dict[str, object]) -> str:
    """Extract the assistant text from one OpenAI Batch output JSONL line.

    Batch results are plain JSON dicts (not SDK objects). Per-request failures
    surface either as a top-level ``error`` or a non-200 ``response.status_code``;
    both raise so the caller can mark the row terminal ERROR.
    """
    error = line.get("error")
    if error:
        raise RuntimeError(f"Batch request error: {error}")
    response = cast(dict[str, object], line.get("response") or {})
    status_code = response.get("status_code")
    if status_code != 200:  # noqa: PLR2004
        raise RuntimeError(
            f"Batch request returned status {status_code}: {response.get('body')}"
        )
    body = cast(dict[str, object], response.get("body") or {})
    choices = cast(list[dict[str, object]], body.get("choices") or [])
    if not choices:
        raise RuntimeError("Batch response did not include choices.")
    message = cast(dict[str, object], choices[0].get("message") or {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        return "".join(text_parts)
    raise RuntimeError("Batch response content was not text.")


def as_plain_dict(value: object) -> dict[str, object] | None:
    """Return one SDK model or mapping as a plain JSON-serializable dict.

    Falls back to the instance attributes rather than returning None, so a
    provider SDK that stops using pydantic models does not silently start
    logging no usage at all.
    """
    if value is None:
        return None
    candidate: object = value
    if not isinstance(value, dict):
        dump = getattr(value, "model_dump", None)
        candidate = dump() if callable(dump) else getattr(value, "__dict__", None)
    if not isinstance(candidate, dict):
        return None
    return cast("dict[str, object]", json.loads(json.dumps(candidate, default=str)))


def completion_result_from_response(response: object) -> CompletionResult:
    """Build one completion result from an OpenRouter SDK response."""
    choices = getattr(response, "choices", None) or []
    choice = choices[0] if choices else None
    message = getattr(choice, "message", None)
    return CompletionResult(
        text=extract_response_text(response),
        finish_reason=getattr(choice, "finish_reason", None),
        refusal=getattr(message, "refusal", None),
        usage=as_plain_dict(getattr(response, "usage", None)),
        response_id=getattr(response, "id", None),
        served_model=getattr(response, "model", None),
    )


def completion_result_from_batch_line(line: dict[str, object]) -> CompletionResult:
    """Build one completion result from an OpenAI Batch output JSONL line.

    The batch route reaches OpenAI directly rather than through OpenRouter, so
    the usage block carries token counts but no `cost`. A filtered response
    arrives as a normal `200` with a body, so it never reaches the
    infrastructure-error path; callers detect it by `finish_reason`
    (`is_content_filter_abort`). Raises as ``extract_batch_response_text`` does.
    """
    text = extract_batch_response_text(line)
    response = cast(dict[str, object], line.get("response") or {})
    body = cast(dict[str, object], response.get("body") or {})
    choices = cast(list[dict[str, object]], body.get("choices") or [])
    choice = choices[0] if choices else {}
    message = cast(dict[str, object], choice.get("message") or {})
    refusal = message.get("refusal")
    finish_reason = choice.get("finish_reason")
    return CompletionResult(
        text=text,
        finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        refusal=refusal if isinstance(refusal, str) else None,
        usage=as_plain_dict(body.get("usage")),
        response_id=body.get("id") if isinstance(body.get("id"), str) else None,
        served_model=body.get("model") if isinstance(body.get("model"), str) else None,
    )


def normalize_reasoning_effort(reasoning_effort: str | None) -> str:
    """Resolve and validate configured reasoning effort."""
    resolved = (
        reasoning_effort or settings.EXTRACTOR_REASONING or DEFAULT_REASONING_EFFORT
    )
    if resolved not in REASONING_EFFORTS:
        allowed = ", ".join(sorted(REASONING_EFFORTS))
        raise ValueError(
            f"Unsupported reasoning effort {resolved!r}; expected one of {allowed}"
        )
    return resolved


def native_model_id(model: str) -> str:
    """Strip any provider prefix so an OpenRouter slug becomes a native id."""
    return model.split("/", 1)[1] if "/" in model else model


def is_reasoning_model(model: str) -> bool:
    """Return True for model families that take reasoning_effort over temperature."""
    return native_model_id(model).lower().startswith(REASONING_MODEL_PREFIXES)


def sampling_params(model: str) -> dict[str, object]:
    """Return the sampling params to send with one extract call.

    Shared by both backends. Reasoning models reject ``temperature != 1``, so
    temperature is only sent to models that can honor it; for those, pinning it
    to 0 keeps extraction as reproducible as the provider allows.
    """
    if is_reasoning_model(model):
        return {}
    return {"temperature": EXTRACTOR_TEMPERATURE}
