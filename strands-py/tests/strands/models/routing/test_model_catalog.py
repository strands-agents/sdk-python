"""Tests for immutable, credential-safe routing model catalogs."""

import json
import operator

import pytest
import yaml

from strands.models import ModelCatalog


def test_catalog_is_immutable_and_overrides_create_a_new_snapshot():
    catalog = ModelCatalog({"model-a": {"max_input_tokens": 100, "supports_tool_calling": False}})

    with pytest.raises(TypeError):
        operator.setitem(catalog["model-a"], "max_input_tokens", 200)

    copied = catalog.as_dict()
    copied["model-a"]["max_input_tokens"] = 300
    updated = catalog.with_overrides(
        {
            "model-a": {"max_input_tokens": 200},
            "model-b": {"input_cost_per_token": 0.000001},
        }
    )

    assert catalog["model-a"] == {"max_input_tokens": 100, "supports_tool_calling": False}
    assert updated["model-a"] == {"max_input_tokens": 200, "supports_tool_calling": False}
    assert updated["model-b"] == {"input_cost_per_token": 0.000001}


@pytest.mark.parametrize("suffix", [".json", ".yaml"], ids=["json", "yaml"])
def test_from_file_loads_versioned_json_and_yaml(tmp_path, suffix):
    path = tmp_path / f"catalog{suffix}"
    document = {
        "version": 1,
        "models": {
            "provider/model": {
                "input_cost_per_token": 0.0000055,
                "output_cost_per_token_above_272k_tokens": 0.0000495,
                "cache_read_input_token_cost": 0.00000055,
                "max_input_tokens": 272_000,
                "supports_vision": True,
            }
        },
    }
    path.write_text(json.dumps(document) if suffix == ".json" else yaml.safe_dump(document), encoding="utf-8")

    assert ModelCatalog.from_file(path).as_dict() == document["models"]


@pytest.mark.parametrize("suffix", [".json", ".yaml"], ids=["json", "yaml"])
def test_from_file_rejects_duplicate_keys(tmp_path, suffix):
    path = tmp_path / f"catalog{suffix}"
    if suffix == ".json":
        content = '{"version": 1, "version": 1, "models": {}}'
    else:
        content = "version: 1\nversion: 1\nmodels: {}\n"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        ModelCatalog.from_file(path)


def test_from_file_rejects_litellm_proxy_configuration_shape(tmp_path):
    path = tmp_path / "catalog.yaml"
    path.write_text(
        """
model_list:
  - model_name: customer-facing-name
    litellm_params:
      model: azure/gpt-5.5-deployment
      api_key: os.environ/AZURE_API_KEY
      input_cost_per_token: 0.0000055
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only version and models"):
        ModelCatalog.from_file(path)


@pytest.mark.parametrize("field", ["api_key", "api_base", "extra_headers", "description"])
def test_native_catalog_rejects_unsupported_or_sensitive_fields(field):
    with pytest.raises(ValueError, match="unsupported metadata field"):
        ModelCatalog({"model": {field: "must-not-pass"}})


@pytest.mark.parametrize(
    ("document", "exception", "match"),
    [
        ({"version": 2, "models": {}}, ValueError, "version must be 1"),
        ({"version": True, "models": {}}, TypeError, "version must be an integer"),
        ({"version": 1, "models": []}, TypeError, "models must be a mapping"),
        ({"version": 1, "models": {"model": {}}}, ValueError, "must not be empty"),
    ],
)
def test_from_file_rejects_invalid_native_documents(tmp_path, document, exception, match):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(exception, match=match):
        ModelCatalog.from_file(path)


def test_from_file_rejects_oversized_documents(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_bytes(b" " * 1_000_001)

    with pytest.raises(ValueError, match="1000000 bytes"):
        ModelCatalog.from_file(path)
