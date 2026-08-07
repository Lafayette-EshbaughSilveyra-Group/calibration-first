# text_score_adapter.py

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any

from openai import OpenAI, RateLimitError


MODEL = "gpt-4.1-mini"
MAX_RETRIES = 6

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Create and cache the OpenAI client."""

    global _client

    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set in the current environment."
        )

    _client = OpenAI(
        api_key=api_key,
    )

    return _client


def build_text_prompt(
    inspection_report: str,
) -> str:
    """Build the prompt for scoring one whole-home inspection note."""

    return f"""
You are an expert building-energy analyst.

Below is one narrative inspection report describing a whole home.

Estimate the need for each of the following retrofit categories:

1. HVAC upgrade
2. Insulation upgrade

Return one score for each category in the range [0, 1].

Scoring interpretation:
- 0 means the retrofit is definitely not needed.
- 1 means the retrofit is definitely needed.
- Intermediate values indicate partial need or uncertainty.

Evaluate HVAC and insulation separately using all relevant evidence in the
complete inspection report. Do not infer that both categories have the same
condition merely because they appear in the same report.

INSPECTION REPORT:
\"\"\"
{inspection_report}
\"\"\"

Return exactly one JSON object with this structure:

{{
  "hvac_retrofit_need": 0.5,
  "insulation_retrofit_need": 0.5
}}

Do not include Markdown, explanation, or commentary.
""".strip()


def safe_chat_response(
    prompt: str,
    client: OpenAI,
) -> str:
    """Call the model and return its textual response.

    Rate-limit errors are retried with exponential backoff.
    """

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You produce calibrated building-retrofit "
                            "judgments and return only valid JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.0,
                response_format={
                    "type": "json_object",
                },
            )

            if not response.choices:
                raise ValueError(
                    "The OpenAI response contained no choices."
                )

            content = response.choices[0].message.content

            if content is None or not content.strip():
                raise ValueError(
                    "The OpenAI response contained no message content."
                )

            return content.strip()

        except RateLimitError:
            if attempt == MAX_RETRIES - 1:
                raise

            delay_seconds = min(
                2 ** (attempt + 1),
                30,
            )

            print(
                "Rate limit encountered. "
                f"Retrying in {delay_seconds} seconds..."
            )

            time.sleep(delay_seconds)

    raise RuntimeError(
        "The text scorer exhausted its retry attempts."
    )


def parse_json_response(
    content: str,
) -> Mapping[str, Any]:
    """Parse and validate the model's JSON object."""

    cleaned = content.strip()

    # Defensive handling in case Markdown fences are returned despite the
    # response-format request.
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines and lines[0].strip() in {
            "```",
            "```json",
        }:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    try:
        result = json.loads(cleaned)

    except json.JSONDecodeError as error:
        raise ValueError(
            "The model did not return valid JSON.\n\n"
            f"Returned content:\n{content}"
        ) from error

    if not isinstance(result, Mapping):
        raise ValueError(
            "The model response must be a JSON object."
        )

    return result


def validate_score(
    value: Any,
    *,
    field_name: str,
) -> float:
    """Convert a score to float and require that it lies in [0, 1]."""

    try:
        score = float(value)

    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name!r} must be numeric; received {value!r}."
        ) from error

    if not 0.0 <= score <= 1.0:
        raise ValueError(
            f"{field_name!r} must lie in [0, 1]; received {score}."
        )

    return score


def score_text(
    note: str,
    record: Mapping[str, Any],
) -> dict[str, float]:
    """Score one whole-home note and return two concept judgments.

    The note is sent to the model once. The model returns separate HVAC and
    insulation retrofit-need scores.

    The record argument is accepted to match the experiment runner's scorer
    interface. It is not provided to the model.
    """

    del record

    if not isinstance(note, str) or not note.strip():
        raise ValueError(
            "The inspection note must be a non-empty string."
        )

    prompt = build_text_prompt(
        note.strip(),
    )

    content = safe_chat_response(
        prompt,
        get_client(),
    )

    result = parse_json_response(
        content,
    )

    required_fields = {
        "hvac_retrofit_need",
        "insulation_retrofit_need",
    }

    missing_fields = required_fields.difference(
        result
    )

    if missing_fields:
        raise ValueError(
            "The model response is missing required field(s): "
            f"{sorted(missing_fields)}.\n\n"
            f"Returned object: {dict(result)}"
        )

    return {
        "hvac_retrofit_need": validate_score(
            result["hvac_retrofit_need"],
            field_name="hvac_retrofit_need",
        ),
        "insulation_retrofit_need": validate_score(
            result["insulation_retrofit_need"],
            field_name="insulation_retrofit_need",
        ),
    }