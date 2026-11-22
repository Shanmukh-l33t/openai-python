from __future__ import annotations

from typing import Any, cast
from dataclasses import FrozenInstanceError

import pytest

from openai import OpenAIError
from openai.lib.live import TranscriptGrouper, AsyncTranscriptGrouper
from openai.types.live import InputTranscriptDeltaEvent, OutputTranscriptDeltaEvent

from .helpers import FakeClock, Recording, ClockGrouper, AsyncClockGrouper, push, text, close, event


@pytest.mark.parametrize("factory", [TranscriptGrouper, AsyncTranscriptGrouper])
@pytest.mark.parametrize(
    "key", ["min_turn_separation_ms", "assistant_silence_ms", "backchannel_max_duration_ms", "backchannel_isolation_ms"]
)
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), 2_147_483_648, True, "100", 10**400])
def test_invalid_options(
    factory: type[TranscriptGrouper] | type[AsyncTranscriptGrouper], key: str, value: object
) -> None:
    with pytest.raises(OpenAIError, match="finite number"):
        factory(**{key: cast(Any, value)})


@pytest.mark.parametrize("factory", [TranscriptGrouper, AsyncTranscriptGrouper])
@pytest.mark.parametrize(
    "value",
    [
        "cool",
        "",
        b"cool",
        b"",
        bytearray(b"cool"),
        memoryview(b"cool"),
        0,
        False,
        123,
        {},
        {"cool"},
        [123],
        [None],
        [b"cool"],
    ],
)
def test_invalid_additional_acknowledgments(
    factory: type[TranscriptGrouper] | type[AsyncTranscriptGrouper], value: object
) -> None:
    with pytest.raises(OpenAIError, match="sequence of strings"):
        factory(additional_acknowledgments=cast(Any, value))


@pytest.mark.parametrize("factory", [TranscriptGrouper, AsyncTranscriptGrouper])
@pytest.mark.parametrize("value", [None, (), [], ["cool"], ("cool", "okay")])
def test_additional_acknowledgments_accepts_sequences(
    factory: type[TranscriptGrouper] | type[AsyncTranscriptGrouper], value: object
) -> None:
    factory(additional_acknowledgments=cast(Any, value))


@pytest.mark.parametrize("async_mode", [False, True])
async def test_additional_acknowledgments_suppresses_whole_phrase(async_mode: bool) -> None:
    texts: list[str] = []
    for acknowledgments in (["cool"], None):
        clock = FakeClock()
        grouper = (AsyncClockGrouper if async_mode else ClockGrouper)(clock, acknowledgments)
        recording = Recording(grouper, clock)
        await push(grouper, text("Here is my question", item_id="u1", speaker="user"))
        await push(grouper, text("cool", start=300, end=500, item_id="a1"))
        await push(grouper, text("about the API.", start=600, end=1200, item_id="u2", speaker="user"))
        await close(grouper)
        texts.append(" | ".join(item.segment.text for item in recording.closed))
    assert texts == ["Here is my question about the API.", "Here is my question about the API. | cool"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("start_ms", -1),
        ("start_ms", 0.5),
        ("start_ms", float("nan")),
        ("start_ms", float("inf")),
        ("start_ms", 9_007_199_254_740_992),
        ("start_ms", True),
        ("start_ms", None),
        ("start_ms", "0"),
        ("end_ms", -1),
        ("end_ms", True),
        ("end_ms", 0.5),
        ("end_ms", 9_007_199_254_740_992),
        ("event_id", ""),
        ("event_id", 123),
        ("delta", None),
        ("delta", 123),
    ],
)
@pytest.mark.parametrize("async_mode", [False, True])
async def test_invalid_transcript_before_state_changes(field: str, value: object, async_mode: bool) -> None:
    clock = FakeClock()
    grouper = (AsyncClockGrouper if async_mode else ClockGrouper)(clock)
    recording = Recording(grouper, clock)
    data: dict[str, Any] = {
        "type": "session.input_transcript.delta",
        "start_ms": 0,
        "end_ms": 200,
        "event_id": "same",
        "delta": "Invalid",
    }
    data[field] = value
    incoming = event(data)
    assert isinstance(incoming, InputTranscriptDeltaEvent)
    # Pydantic may coerce bool/string times and (v1) numeric text/IDs. Mutate
    # the constructed model so the helper actually receives the malformed value.
    setattr(incoming, field, value)
    with pytest.raises(OpenAIError, match="Invalid public Live transcript"):
        await push(grouper, incoming)
    await push(grouper, text("Valid", item_id="same", speaker="user"))
    await close(grouper)
    assert [item.segment.text for item in recording.closed] == ["Valid"]


@pytest.mark.parametrize("async_mode", [False, True])
async def test_missing_fields_and_reversed_interval(async_mode: bool) -> None:
    grouper = (AsyncClockGrouper if async_mode else ClockGrouper)(FakeClock())
    for data in [
        {"type": "session.input_transcript.delta"},
        {
            "type": "session.input_transcript.delta",
            "start_ms": 200,
            "end_ms": 100,
            "event_id": "same",
            "delta": "x",
        },
    ]:
        with pytest.raises(OpenAIError, match="Invalid public Live transcript"):
            await push(grouper, event(data))
    await close(grouper)


@pytest.mark.parametrize("async_mode", [False, True])
async def test_copies_input_and_freezes_snapshots(async_mode: bool) -> None:
    clock = FakeClock()
    grouper = (AsyncClockGrouper if async_mode else ClockGrouper)(clock)
    recording = Recording(grouper, clock)
    incoming = text("Original")
    assert isinstance(incoming, OutputTranscriptDeltaEvent)
    await push(grouper, incoming)
    incoming.delta = "Changed"
    incoming.event_id = "changed-id"
    incoming.end_ms = 999
    await close(grouper)
    segment = recording.closed[0].segment
    assert segment.text == "Original"
    assert segment.end_ms == 200
    with pytest.raises(FrozenInstanceError):
        cast(Any, segment).text = "Mutated"
    with pytest.raises(FrozenInstanceError):
        cast(Any, recording.closed[0]).reason = "inactivity"


@pytest.mark.parametrize("async_mode", [False, True])
async def test_large_unicode_text_without_limits(async_mode: bool) -> None:
    clock = FakeClock()
    grouper = (AsyncClockGrouper if async_mode else ClockGrouper)(clock)
    recording = Recording(grouper, clock)
    value = "你好 😀" * 100_000
    await push(grouper, text(value))
    await push(grouper, text(" \n", start=200, end=400, item_id="second"))
    await close(grouper)
    assert recording.closed[0].segment.text == value + " \n"


@pytest.mark.parametrize("async_mode", [False, True])
async def test_idempotent_close_and_push_after_close(async_mode: bool) -> None:
    clock = FakeClock()
    grouper = (AsyncClockGrouper if async_mode else ClockGrouper)(clock)
    recording = Recording(grouper, clock)
    await push(grouper, text())
    await close(grouper)
    await close(grouper)
    for incoming in [text(), event({"type": "turn.created"}), event({"type": "session.closed"})]:
        with pytest.raises(OpenAIError, match="after closing"):
            await push(grouper, incoming)
    assert len(recording.closed) == 1
    assert not clock.pending


@pytest.mark.parametrize("async_mode", [False, True])
async def test_integer_valued_float_matches_javascript(async_mode: bool) -> None:
    clock = FakeClock()
    grouper = (AsyncClockGrouper if async_mode else ClockGrouper)(clock)
    recording = Recording(grouper, clock)
    incoming = text(speaker="user")
    assert isinstance(incoming, InputTranscriptDeltaEvent)
    # Number.isSafeInteger(0.0) is true in TypeScript. Model mutation avoids coercion.
    cast(Any, incoming).start_ms = 0.0
    cast(Any, incoming).end_ms = 200.0
    await push(grouper, incoming)
    await close(grouper)
    assert recording.closed[0].segment.start_ms == 0
    assert recording.closed[0].segment.end_ms == 200
