import pytest

from openai import OpenAI, AsyncOpenAI
from openai._types import NOT_GIVEN


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_stream_requires_input_for_new_response(is_async: bool) -> None:
    if is_async:
        async with AsyncOpenAI(api_key="My API Key") as client:
            with pytest.raises(ValueError, match="input must be provided"):
                client.responses.stream(model="gpt-4o-mini")
    else:
        with OpenAI(api_key="My API Key") as client:
            with pytest.raises(ValueError, match="input must be provided"):
                client.responses.stream(model="gpt-4o-mini")


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_stream_requires_response_id_for_existing_response(is_async: bool) -> None:
    if is_async:
        async with AsyncOpenAI(api_key="My API Key") as client:
            with pytest.raises(ValueError, match="must be provided when streaming an existing response"):
                client.responses.stream(response_id=NOT_GIVEN)
    else:
        with OpenAI(api_key="My API Key") as client:
            with pytest.raises(ValueError, match="must be provided when streaming an existing response"):
                client.responses.stream(response_id=NOT_GIVEN)
