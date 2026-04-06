"""Task 3.1 — Verify ClaudeAdapter implements LLMPort via Anthropic SDK.

All tests use pytest-mock to stub out the anthropic SDK — no real API calls.
"""

from unittest.mock import MagicMock, patch

import pytest

from sommelier.domain.models import LLMUnavailableError, Message
from sommelier.ports.interfaces import LLMRequest, LLMResponse


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_sdk_response(content: str, input_tokens: int = 10, output_tokens: int = 20):
    """Build a minimal mock that looks like anthropic.types.Message."""
    response = MagicMock()
    response.content = [MagicMock(text=content)]
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    return response


def _make_request(
    model: str = "extraction",
    system_prompt: str = "You are a helpful assistant.",
    messages: list | None = None,
    max_tokens: int = 256,
    temperature: float = 0.3,
) -> LLMRequest:
    if messages is None:
        messages = [Message(role="user", content="Hello")]
    return LLMRequest(
        system_prompt=system_prompt,
        messages=messages,
        model=model,  # type: ignore[arg-type]
        max_tokens=max_tokens,
        temperature=temperature,
    )


# ── Import and instantiation ──────────────────────────────────────────────────


class TestImportAndInstantiation:
    def test_claude_adapter_importable(self):
        from sommelier.infrastructure.claude_adapter import ClaudeAdapter  # noqa: F401

    def test_instantiates_with_api_key(self):
        from sommelier.infrastructure.claude_adapter import ClaudeAdapter
        with patch("sommelier.infrastructure.claude_adapter.anthropic.Anthropic"):
            adapter = ClaudeAdapter(api_key="sk-test-key")
            assert adapter is not None

    def test_instantiates_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-key")
        from sommelier.infrastructure.claude_adapter import ClaudeAdapter
        with patch("sommelier.infrastructure.claude_adapter.anthropic.Anthropic"):
            adapter = ClaudeAdapter()
            assert adapter is not None

    def test_implements_llm_port(self):
        from sommelier.infrastructure.claude_adapter import ClaudeAdapter
        from sommelier.ports.interfaces import LLMPort
        with patch("sommelier.infrastructure.claude_adapter.anthropic.Anthropic"):
            adapter = ClaudeAdapter(api_key="sk-test")
        assert isinstance(adapter, LLMPort)


# ── Model routing ─────────────────────────────────────────────────────────────


class TestModelRouting:
    def test_extraction_model_routes_to_haiku(self):
        from sommelier.infrastructure.claude_adapter import ClaudeAdapter
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_sdk_response("ok")
        with patch("sommelier.infrastructure.claude_adapter.anthropic.Anthropic", return_value=mock_client):
            adapter = ClaudeAdapter(api_key="sk-test")
        adapter.complete(_make_request(model="extraction"))
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"

    def test_generation_model_routes_to_sonnet(self):
        from sommelier.infrastructure.claude_adapter import ClaudeAdapter
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_sdk_response("ok")
        with patch("sommelier.infrastructure.claude_adapter.anthropic.Anthropic", return_value=mock_client):
            adapter = ClaudeAdapter(api_key="sk-test")
        adapter.complete(_make_request(model="generation"))
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-6"


# ── Request serialization ─────────────────────────────────────────────────────


class TestRequestSerialization:
    def _adapter_with_mock(self):
        from sommelier.infrastructure.claude_adapter import ClaudeAdapter
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_sdk_response("response text")
        with patch("sommelier.infrastructure.claude_adapter.anthropic.Anthropic", return_value=mock_client):
            adapter = ClaudeAdapter(api_key="sk-test")
        return adapter, mock_client

    def test_system_prompt_passed_as_top_level_field(self):
        adapter, mock_client = self._adapter_with_mock()
        adapter.complete(_make_request(system_prompt="Be concise."))
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "Be concise."

    def test_messages_serialized_as_role_content_dicts(self):
        adapter, mock_client = self._adapter_with_mock()
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
        adapter.complete(_make_request(messages=messages))
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["messages"] == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

    def test_max_tokens_passed_correctly(self):
        adapter, mock_client = self._adapter_with_mock()
        adapter.complete(_make_request(max_tokens=512))
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 512

    def test_temperature_passed_correctly(self):
        adapter, mock_client = self._adapter_with_mock()
        adapter.complete(_make_request(temperature=0.7))
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7


# ── Response deserialization ──────────────────────────────────────────────────


class TestResponseDeserialization:
    def _adapter_and_client(self, sdk_response):
        from sommelier.infrastructure.claude_adapter import ClaudeAdapter
        mock_client = MagicMock()
        mock_client.messages.create.return_value = sdk_response
        with patch("sommelier.infrastructure.claude_adapter.anthropic.Anthropic", return_value=mock_client):
            adapter = ClaudeAdapter(api_key="sk-test")
        return adapter

    def test_returns_llm_response(self):
        adapter = self._adapter_and_client(_make_sdk_response("hello"))
        result = adapter.complete(_make_request())
        assert isinstance(result, LLMResponse)

    def test_content_extracted_from_first_block(self):
        adapter = self._adapter_and_client(_make_sdk_response("extracted text"))
        result = adapter.complete(_make_request())
        assert result.content == "extracted text"

    def test_input_tokens_populated(self):
        adapter = self._adapter_and_client(_make_sdk_response("x", input_tokens=42))
        result = adapter.complete(_make_request())
        assert result.input_tokens == 42

    def test_output_tokens_populated(self):
        adapter = self._adapter_and_client(_make_sdk_response("x", output_tokens=99))
        result = adapter.complete(_make_request())
        assert result.output_tokens == 99


# ── Error handling (task 3.1 scope: errors propagate; wrapping in 3.2) ────────


class TestErrorPropagation:
    def test_api_error_raises_llm_unavailable_error(self):
        import anthropic
        from sommelier.infrastructure.claude_adapter import ClaudeAdapter
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.APIError(
            message="rate limit", request=MagicMock(), body=None
        )
        with patch("sommelier.infrastructure.claude_adapter.anthropic.Anthropic", return_value=mock_client):
            adapter = ClaudeAdapter(api_key="sk-test")
        with pytest.raises(LLMUnavailableError):
            adapter.complete(_make_request())

    def test_network_error_raises_llm_unavailable_error(self):
        import anthropic
        from sommelier.infrastructure.claude_adapter import ClaudeAdapter
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.APIConnectionError(request=MagicMock())
        with patch("sommelier.infrastructure.claude_adapter.anthropic.Anthropic", return_value=mock_client):
            adapter = ClaudeAdapter(api_key="sk-test")
        with pytest.raises(LLMUnavailableError):
            adapter.complete(_make_request())

    def test_error_message_is_descriptive(self):
        import anthropic
        from sommelier.infrastructure.claude_adapter import ClaudeAdapter
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.APIError(
            message="quota exceeded", request=MagicMock(), body=None
        )
        with patch("sommelier.infrastructure.claude_adapter.anthropic.Anthropic", return_value=mock_client):
            adapter = ClaudeAdapter(api_key="sk-test")
        with pytest.raises(LLMUnavailableError, match="quota exceeded"):
            adapter.complete(_make_request())
