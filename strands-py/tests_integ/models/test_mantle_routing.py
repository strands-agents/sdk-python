"""Drift detection for the Bedrock Mantle base-path table.

Mantle serves each model from exactly one base path (``/v1`` or ``/openai/v1``) and
rejects the other with HTTP 400 ``validation_error``. The path is a property of the
individual model and is *not* discoverable from the API: ``GET /v1/models`` reports
``status`` but not routing, and there is no ``/openai/v1/models``. So
:data:`strands.models._openai_bedrock._OPENAI_PATH_MODEL_IDS` is a hand-maintained
table, and it silently goes stale whenever Mantle onboards a model.

This test closes that gap. It lists the live catalog and, for every model, probes the
path the SDK would *not* use. A model that answers on the unused path is mis-routed, so
the test fails naming the offending ids. That turns "Mantle onboarded a model and the SDK
sends it to the wrong route" from a 400 in a user's application into a CI failure.

Failure means the table needs updating, not that the SDK is broken for existing models.
See https://github.com/strands-agents/harness-sdk/issues/3654.
"""

import concurrent.futures
import json
import urllib.error
import urllib.request

import pytest

from strands.models._openai_bedrock import _resolve_mantle_base_path, resolve_bedrock_client_args

_REGION = "us-east-1"
_BASE = f"https://bedrock-mantle.{_REGION}.api.aws"
_TIMEOUT = 30
_MAX_WORKERS = 8

# Models that answer on neither OpenAI-compatible base path. The Anthropic family is served
# from /anthropic/v1/messages (a different protocol, reached via AnthropicModel, not
# OpenAIModel), so it is out of scope for this table.
_NOT_OPENAI_COMPATIBLE_PREFIXES = ("anthropic.",)


def _token() -> str:
    """Mint a Mantle bearer token, or skip if the ambient chain can't produce one."""
    try:
        return resolve_bedrock_client_args({"region": _REGION})["api_key"]
    except Exception as e:  # noqa: BLE001 - any credential/import failure means "can't run here"
        pytest.skip(f"cannot mint a Bedrock Mantle token: {e}")


def _list_models(token: str) -> list[str]:
    request = urllib.request.Request(f"{_BASE}/v1/models", headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            pytest.skip(f"account lacks bedrock-mantle:ListModels ({e.code})")
        raise
    return sorted(model["id"] for model in payload["data"])


def _status(path: str, body: dict, token: str) -> int:
    """POST to a Mantle route and return the HTTP status (0 on timeout)."""
    request = urllib.request.Request(
        f"{_BASE}{path}",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            return response.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:  # noqa: BLE001 - a hung route is "not served", same as a 400
        return 0


def _serves(base_path: str, model_id: str, token: str) -> bool:
    """Whether Mantle serves ``model_id`` from ``base_path`` on either API surface."""
    chat = _status(
        f"{base_path}/chat/completions",
        {"model": model_id, "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 8},
        token,
    )
    if chat == 200:
        return True
    responses = _status(
        f"{base_path}/responses",
        {"model": model_id, "input": "hi", "max_output_tokens": 24},
        token,
    )
    return responses == 200


def test_mantle_base_path_table_matches_live_catalog():
    """Every live Mantle model is routed to the base path it is actually served from."""
    token = _token()
    models = [model_id for model_id in _list_models(token) if not model_id.startswith(_NOT_OPENAI_COMPATIBLE_PREFIXES)]
    assert models, "Mantle returned no OpenAI-compatible models"

    def check(model_id: str) -> tuple[str, str, bool]:
        resolved = _resolve_mantle_base_path(model_id)
        unused = "/openai/v1" if resolved == "/v1" else "/v1"
        # Probe only the path the SDK would *not* use: if that one answers, we are wrong.
        # Minting per model keeps long sweeps inside the token's lifetime.
        return model_id, resolved, _serves(unused, model_id, _token())

    with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        results = list(pool.map(check, models))

    misrouted = {model_id: resolved for model_id, resolved, wrong in results if wrong}
    assert not misrouted, (
        "Mantle serves these models from the base path the SDK does not use. Update "
        "_OPENAI_PATH_MODEL_IDS in strands/models/_openai_bedrock.py (and the TypeScript "
        f"mirror in strands-ts/src/models/openai/mantle.ts): {misrouted}"
    )


@pytest.mark.parametrize(
    "model_id",
    ["xai.grok-4.3", "google.gemma-4-31b", "google.gemma-3-27b-it", "openai.gpt-oss-120b"],
)
def test_mantle_resolved_base_path_is_served(model_id):
    """The resolved base path actually serves each regression-case model."""
    token = _token()
    if model_id not in _list_models(token):
        pytest.skip(f"{model_id} is not in the {_REGION} catalog")

    assert _serves(_resolve_mantle_base_path(model_id), model_id, token)
