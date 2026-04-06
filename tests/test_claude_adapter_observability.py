"""Task 3.2 — Verify ClaudeAdapter error handling and observability.

Covers:
- Log line format: model name, input/output token counts, latency in ms
- Exhaustive error taxonomy: APIError subtypes, connection errors, generic exceptions
- LLMUnavailableError message includes the original error context
"""

from unittest.mock import MagicMock, patch

import pytest

import anthropic

from sommelier.domain.models import LLMUnavailableError, Message
from sommelier.infrastructure.claude_adapter import ClaudeAdapter
from sommelier.ports.interfaces import LLMRequest


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sdk_response(text="ok", input_tokens=5, output_tokens=15):
    r = MagicMock()
    r.content = [MagicMock(text=text)]
    r.usage.input_tokens = input_tokens
    r.usage.output_tokens = output_tokens
    return r


def _request(model="extraction"):
    return LLMRequest(
        system_prompt="sys",
        messages=[Message(role="user", content="hi")],
        model=model,  # type: ignore[arg-type]
        max_tokens=128,
        temperature=0.3,
    )


def _adapter(side_effect=None, return_value=None):
    mock_client = MagicMock()
    if side_effect is not None:
        mock_client.messages.create.side_effect = side_effect
    else:
        mock_client.messages.create.return_value = return_value or _sdk_response()
    with patch("sommelier.infrastructure.claude_adapter.anthropic.Anthropic", return_value=mock_client):
        adapter = ClaudeAdapter(api_key="sk-test")
    return adapter, mock_client


# ── Observability: log format ─────────────────────────────────────────────────


class TestObservabilityLog:
    def test_log_written_to_stderr_on_success(self, capsys):
        adapter, _ = _adapter(return_value=_sdk_response("hi", 10, 20))
        adapter.complete(_request())
        assert "[ClaudeAdapter]" in capsys.readouterr().err

    def test_log_contains_model_id(self, capsys):
        adapter, _ = _adapter(return_value=_sdk_response())
        adapter.complete(_request(model="extraction"))
        assert "claude-haiku-4-5-20251001" in capsys.readouterr().err

    def test_log_contains_generation_model_id(self, capsys):
        adapter, _ = _adapter(return_value=_sdk_response())
        adapter.complete(_request(model="generation"))
        assert "claude-sonnet-4-6" in capsys.readouterr().err

    def test_log_contains_input_token_count(self, capsys):
        adapter, _ = _adapter(return_value=_sdk_response(input_tokens=42))
        adapter.complete(_request())
        assert "42" in capsys.readouterr().err

    def test_log_contains_output_token_count(self, capsys):
        adapter, _ = _adapter(return_value=_sdk_response(output_tokens=99))
        adapter.complete(_request())
        assert "99" in capsys.readouterr().err

    def test_log_contains_latency_in_ms(self, capsys):
        adapter, _ = _adapter(return_value=_sdk_response())
        adapter.complete(_request())
        err = capsys.readouterr().err
        assert "ms" in err

    def test_log_latency_is_numeric(self, capsys):
        import re
        adapter, _ = _adapter(return_value=_sdk_response())
        adapter.complete(_request())
        err = capsys.readouterr().err
        # e.g. "latency=3ms" or "latency=3.0ms"
        match = re.search(r"latency=(\d+)", err)
        assert match is not None, f"No numeric latency found in log: {err!r}"

    def test_no_log_written_on_error(self, capsys):
        adapter, _ = _adapter(side_effect=anthropic.APIConnectionError(request=MagicMock()))
        with pytest.raises(LLMUnavailableError):
            adapter.complete(_request())
        assert "[ClaudeAdapter]" not in capsys.readouterr().err


# ── Error handling: full taxonomy ─────────────────────────────────────────────


class TestErrorHandling:
    def test_api_status_error_wrapped(self):
        exc = anthropic.APIStatusError(
            message="429 rate limit",
            response=MagicMock(status_code=429),
            body=None,
        )
        adapter, _ = _adapter(side_effect=exc)
        with pytest.raises(LLMUnavailableError):
            adapter.complete(_request())

    def test_api_connection_error_wrapped(self):
        adapter, _ = _adapter(side_effect=anthropic.APIConnectionError(request=MagicMock()))
        with pytest.raises(LLMUnavailableError):
            adapter.complete(_request())

    def test_api_timeout_error_wrapped(self):
        adapter, _ = _adapter(side_effect=anthropic.APITimeoutError(request=MagicMock()))
        with pytest.raises(LLMUnavailableError):
            adapter.complete(_request())

    def test_generic_exception_wrapped(self):
        adapter, _ = _adapter(side_effect=RuntimeError("unexpected crash"))
        with pytest.raises(LLMUnavailableError):
            adapter.complete(_request())

    def test_error_preserves_original_message(self):
        exc = anthropic.APIStatusError(
            message="503 service unavailable",
            response=MagicMock(status_code=503),
            body=None,
        )
        adapter, _ = _adapter(side_effect=exc)
        with pytest.raises(LLMUnavailableError, match="503"):
            adapter.complete(_request())

    def test_generic_error_preserves_message(self):
        adapter, _ = _adapter(side_effect=RuntimeError("disk full"))
        with pytest.raises(LLMUnavailableError, match="disk full"):
            adapter.complete(_request())

    def test_llm_unavailable_error_has_cause(self):
        cause = anthropic.APIConnectionError(request=MagicMock())
        adapter, _ = _adapter(side_effect=cause)
        with pytest.raises(LLMUnavailableError) as exc_info:
            adapter.complete(_request())
        assert exc_info.value.__cause__ is cause
