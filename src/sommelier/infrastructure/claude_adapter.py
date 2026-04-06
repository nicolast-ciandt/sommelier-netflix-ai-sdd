"""ClaudeAdapter — Infrastructure adapter implementing LLMPort.

Wraps the Anthropic Python SDK and dispatches LLMRequest to the
appropriate Claude model based on the semantic role field.

Tasks covered:
  3.1 — complete(), model routing, serialization   ← this file
  3.2 — error handling and observability           ← added in task 3.2
"""

from __future__ import annotations

import os
import sys
import time

import anthropic

from sommelier.domain.models import LLMUnavailableError
from sommelier.ports.interfaces import LLMRequest, LLMResponse

def _model_map() -> dict[str, str]:
    return {
        "extraction": os.environ.get("EXTRACTION_MODEL", "anthropic.claude-4-6-sonnet"),
        "generation": os.environ.get("GENERATION_MODEL", "anthropic.claude-4-6-sonnet"),
    }


class ClaudeAdapter:
    """LLMPort implementation backed by the Anthropic Messages API."""

    def __init__(self, api_key: str | None = None) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic(api_key=resolved_key)

    # ── LLMPort ───────────────────────────────────────────────────────────────

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute a completion and return a typed LLMResponse.

        Raises LLMUnavailableError on any Anthropic SDK or network error.
        """
        model_id = _model_map()[request.model]
        messages = [
            {"role": m.role, "content": m.content}
            for m in request.messages
        ]

        start = time.monotonic()
        try:
            response = self._client.messages.create(
                model=model_id,
                system=request.system_prompt,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        except anthropic.APIError as exc:
            raise LLMUnavailableError(str(exc)) from exc
        except Exception as exc:
            raise LLMUnavailableError(str(exc)) from exc

        elapsed_ms = (time.monotonic() - start) * 1000
        content = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        print(
            f"[ClaudeAdapter] model={model_id} "
            f"in={input_tokens} out={output_tokens} "
            f"latency={elapsed_ms:.0f}ms",
            file=sys.stderr,
        )

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
