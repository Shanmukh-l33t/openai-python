"""Regression tests for nullable `illicit` / `illicit/violent` moderation fields.

The live moderation API returns `null` for the `illicit` and `illicit/violent`
entries in both `categories` and `category_scores` (e.g. when the model does not
score those categories), see issue #1786. The typed models must accept those
`None` values both on the client parse path and under strict validation.
"""

from __future__ import annotations

from typing import Any, cast

import httpx2
import pytest

from openai import OpenAI, AsyncOpenAI
from openai.types import ModerationCreateResponse
from openai._compat import PYDANTIC_V1

# Wire-format payload captured from the live API (sanitized): the `illicit` and
# `illicit/violent` entries are JSON `null` in both `categories` and
# `category_scores`, while every other category carries a normal value.
_LIVE_RESULT: dict[str, Any] = {
    "flagged": False,
    "categories": {
        "harassment": False,
        "harassment/threatening": False,
        "hate": False,
        "hate/threatening": False,
        "illicit": None,
        "illicit/violent": None,
        "self-harm": False,
        "self-harm/instructions": False,
        "self-harm/intent": False,
        "sexual": False,
        "sexual/minors": False,
        "violence": False,
        "violence/graphic": False,
    },
    "category_scores": {
        "harassment": 0.000255020015174523,
        "harassment/threatening": 1.3588138244813308e-05,
        "hate": 2.8068381652701646e-05,
        "hate/threatening": 1.0663524108167621e-06,
        "illicit": None,
        "illicit/violent": None,
        "self-harm": 9.841909195529297e-05,
        "self-harm/instructions": 7.693658517382573e-06,
        "self-harm/intent": 7.031533459667116e-05,
        "sexual": 0.013590452261269093,
        "sexual/minors": 0.0031673426274210215,
        "violence": 0.00022930897830519825,
        "violence/graphic": 4.927426198264584e-05,
    },
    "category_applied_input_types": {
        "harassment": ["text"],
        "harassment/threatening": ["text"],
        "hate": ["text"],
        "hate/threatening": ["text"],
        "illicit": ["text"],
        "illicit/violent": ["text"],
        "self-harm": ["text", "image"],
        "self-harm/instructions": ["text", "image"],
        "self-harm/intent": ["text", "image"],
        "sexual": ["text", "image"],
        "sexual/minors": ["text"],
        "violence": ["text", "image"],
        "violence/graphic": ["text", "image"],
    },
}


def _response_payload() -> dict[str, Any]:
    return {
        "id": "modr-test",
        "model": "omni-moderation-latest",
        "results": [_LIVE_RESULT],
    }


def _strict_parse(payload: dict[str, Any]) -> ModerationCreateResponse:
    if PYDANTIC_V1:
        return ModerationCreateResponse.parse_obj(payload)
    return ModerationCreateResponse.model_validate(payload)


@pytest.mark.parametrize("sync", [True, False], ids=["sync", "async"])
async def test_client_parses_null_illicit_fields(sync: bool) -> None:
    """The client parse path must keep `None` for illicit/illicit_violent, not crash or invent values."""
    transport = httpx2.MockTransport(
        lambda _request: httpx2.Response(
            200, json=_response_payload(), headers={"content-type": "application/json"}
        )
    )
    if sync:
        with OpenAI(api_key="fake-test-key", http_client=httpx2.Client(transport=transport)) as client:
            response = client.moderations.create(input="text")
    else:
        async with AsyncOpenAI(
            api_key="fake-test-key", http_client=httpx2.AsyncClient(transport=transport)
        ) as async_client:
            response = await async_client.moderations.create(input="text")

    result = response.results[0]

    assert result.flagged is False

    assert result.categories.illicit is None
    assert result.categories.illicit_violent is None
    assert result.categories.harassment is False

    assert result.category_scores.illicit is None
    assert result.category_scores.illicit_violent is None
    assert result.category_scores.harassment == pytest.approx(0.000255020015174523)


def test_strict_validation_accepts_null_illicit_fields() -> None:
    """Strict validation must accept JSON null for illicit/illicit_violent (issue #1786)."""
    parsed = _strict_parse(_response_payload())
    result = parsed.results[0]

    assert result.categories.illicit is None
    assert result.categories.illicit_violent is None
    assert result.category_scores.illicit is None
    assert result.category_scores.illicit_violent is None
    assert result.category_scores.sexual == pytest.approx(0.013590452261269093)


def test_strict_validation_accepts_numeric_illicit_fields() -> None:
    """Models that do score illicit categories must still parse to plain bool/float values."""
    payload = _response_payload()
    result = cast(dict[str, Any], payload["results"][0])
    cast(dict[str, Any], result["categories"])["illicit"] = True
    cast(dict[str, Any], result["categories"])["illicit/violent"] = False
    cast(dict[str, Any], result["category_scores"])["illicit"] = 0.5
    cast(dict[str, Any], result["category_scores"])["illicit/violent"] = 0.25

    parsed = _strict_parse(payload)
    out = parsed.results[0]

    assert out.categories.illicit is True
    assert out.categories.illicit_violent is False
    assert out.category_scores.illicit == pytest.approx(0.5)
    assert out.category_scores.illicit_violent == pytest.approx(0.25)
