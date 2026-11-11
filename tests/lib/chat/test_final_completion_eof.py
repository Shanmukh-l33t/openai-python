"""Regression tests for issue #4063.

`ChatCompletionStream.get_final_completion()` (and its async twin) must not
silently return an incomplete completion when the stream terminates via an
early EOF / connection close instead of the `data: [DONE]` terminator.

Control pair, fully offline (httpx2.MockTransport + fake API key):
- completed stream: full tool-call, `finish_reason="tool_calls"`, usage, `[DONE]`
  -> `get_final_completion()` returns the complete completion
- early EOF: last tool-call arguments cut mid-string, no finish_reason, no `[DONE]`
  -> `get_final_completion()` raises a clear error instead of returning
     the incomplete accumulation (partial arguments, `finish_reason=None`).
"""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam


def _chunk(choices: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o",
        "choices": choices,
        **extra,
    }


def _sse_body(chunks: list[dict[str, Any]], *, done: bool) -> str:
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    if done:
        body += "data: [DONE]\n\n"
    return body


# Control: a full tool-call stream that terminates cleanly with `finish_reason`,
# usage and `[DONE]`.
_COMPLETED_TOOL_CALL_CHUNKS: list[dict[str, Any]] = [
    _chunk([{"index": 0, "delta": {"role": "assistant"}}]),
    _chunk(
        [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_value", "arguments": ""},
                        }
                    ]
                },
            }
        ]
    ),
    _chunk([{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"value":'}}]}}]),
    _chunk([{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "42}"}}]}}]),
    _chunk([{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]),
    _chunk([], usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}),
]

# Early EOF: same stream, but the final tool-call arguments are cut mid-string
# and the stream ends without `finish_reason`, usage or `[DONE]`.
_EARLY_EOF_TOOL_CALL_CHUNKS: list[dict[str, Any]] = [
    _chunk([{"index": 0, "delta": {"role": "assistant"}}]),
    _chunk(
        [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_value", "arguments": ""},
                        }
                    ]
                },
            }
        ]
    ),
    _chunk([{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"value":'}}]}}]),
]

# Early EOF, content-only case (as reproduced in the issue thread): a single
# content chunk with `finish_reason: null` followed by a clean EOF.
_EARLY_EOF_CONTENT_CHUNKS: list[dict[str, Any]] = [
    _chunk([{"index": 0, "delta": {"role": "assistant", "content": "partial answer"}}]),
]


def _client(body: str) -> OpenAI:
    return OpenAI(
        api_key="test-key",
        max_retries=0,
        base_url="https://example.test/v1",
        http_client=httpx2.Client(
            transport=httpx2.MockTransport(
                lambda _: httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})
            )
        ),
    )


def _async_client(body: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="test-key",
        max_retries=0,
        base_url="https://example.test/v1",
        http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                lambda _: httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})
            )
        ),
    )


MESSAGES: list[ChatCompletionMessageParam] = [{"role": "user", "content": "hi"}]


def test_get_final_completion_control_completed_stream_sync() -> None:
    with _client(_sse_body(_COMPLETED_TOOL_CALL_CHUNKS, done=True)) as client:
        with client.chat.completions.stream(model="gpt-4o", messages=MESSAGES) as stream:
            completion = stream.get_final_completion()

    assert completion.choices[0].finish_reason == "tool_calls"
    tool_calls = completion.choices[0].message.tool_calls
    assert tool_calls is not None
    assert tool_calls[0].function.arguments == '{"value":42}'
    assert completion.usage is not None
    assert completion.usage.total_tokens == 3


async def test_get_final_completion_control_completed_stream_async() -> None:
    async with _async_client(_sse_body(_COMPLETED_TOOL_CALL_CHUNKS, done=True)) as client:
        async with client.chat.completions.stream(model="gpt-4o", messages=MESSAGES) as stream:
            completion = await stream.get_final_completion()

    assert completion.choices[0].finish_reason == "tool_calls"
    tool_calls = completion.choices[0].message.tool_calls
    assert tool_calls is not None
    assert tool_calls[0].function.arguments == '{"value":42}'
    assert completion.usage is not None
    assert completion.usage.total_tokens == 3


def test_get_final_completion_raises_on_early_eof_tool_call_sync() -> None:
    with _client(_sse_body(_EARLY_EOF_TOOL_CALL_CHUNKS, done=False)) as client:
        with client.chat.completions.stream(model="gpt-4o", messages=MESSAGES) as stream:
            with pytest.raises(RuntimeError, match=r"\[DONE\]"):
                stream.get_final_completion()


async def test_get_final_completion_raises_on_early_eof_tool_call_async() -> None:
    async with _async_client(_sse_body(_EARLY_EOF_TOOL_CALL_CHUNKS, done=False)) as client:
        async with client.chat.completions.stream(model="gpt-4o", messages=MESSAGES) as stream:
            with pytest.raises(RuntimeError, match=r"\[DONE\]"):
                await stream.get_final_completion()


def test_get_final_completion_raises_on_early_eof_content_sync() -> None:
    with _client(_sse_body(_EARLY_EOF_CONTENT_CHUNKS, done=False)) as client:
        with client.chat.completions.stream(model="gpt-4o", messages=MESSAGES) as stream:
            with pytest.raises(RuntimeError, match=r"\[DONE\]"):
                stream.get_final_completion()


async def test_get_final_completion_raises_on_early_eof_content_async() -> None:
    async with _async_client(_sse_body(_EARLY_EOF_CONTENT_CHUNKS, done=False)) as client:
        async with client.chat.completions.stream(model="gpt-4o", messages=MESSAGES) as stream:
            with pytest.raises(RuntimeError, match=r"\[DONE\]"):
                await stream.get_final_completion()


def test_iteration_and_snapshot_unaffected_by_early_eof_sync() -> None:
    # Iterating the stream and reading `current_completion_snapshot` must keep
    # working on a truncated stream; only `get_final_completion()` guarantees
    # completeness.
    with _client(_sse_body(_EARLY_EOF_CONTENT_CHUNKS, done=False)) as client:
        with client.chat.completions.stream(model="gpt-4o", messages=MESSAGES) as stream:
            events = [event.type for event in stream]
            assert events
            assert stream.current_completion_snapshot.choices[0].message.content == "partial answer"


async def test_iteration_and_snapshot_unaffected_by_early_eof_async() -> None:
    async with _async_client(_sse_body(_EARLY_EOF_CONTENT_CHUNKS, done=False)) as client:
        async with client.chat.completions.stream(model="gpt-4o", messages=MESSAGES) as stream:
            events = [event.type async for event in stream]
            assert events
            assert stream.current_completion_snapshot.choices[0].message.content == "partial answer"
