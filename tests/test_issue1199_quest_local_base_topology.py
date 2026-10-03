"""Issue #1199: a local QuEST base is refused on topology before it is hashed.

``_validate_raw_topology`` runs inside calibration, after ``AutoModelForCausalLM``
has already read every byte of the base. A base that can never satisfy the route
is therefore SHA-256-hashed in full twice before the gate refuses it. These
tests pin the pre-hash refusal and its ordering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from soup_cli.utils import quest
from soup_cli.utils.quest import resolve_base_model_identity

COMPATIBLE_WIDTHS = {
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
}


def _config(layers: int = 24, **overrides: Any) -> dict[str, Any]:
    return {
        "model_type": "llama",
        "num_hidden_layers": layers,
        **COMPATIBLE_WIDTHS,
        **overrides,
    }


def _base(directory: Path, config: dict[str, Any], *, weights: bytes = b"safe-weights") -> Path:
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "model.safetensors").write_bytes(weights)
    return directory


@pytest.fixture
def hashed_paths(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every file whose content is read through ``_hash_local_file``."""
    reads: list[str] = []
    original = quest._hash_local_file

    def recording(path: Path) -> tuple[int, str]:
        reads.append(path.name)
        return original(path)

    monkeypatch.setattr(quest, "_hash_local_file", recording)
    return reads


def test_wrong_layer_count_is_refused_before_a_weight_is_read(tmp_path: Path, hashed_paths):
    root = _base(tmp_path / "base", _config(layers=30))

    with pytest.raises(ValueError, match="24 blocks"):
        resolve_base_model_identity(str(root))

    assert hashed_paths == []


def test_narrow_width_is_refused_before_a_weight_is_read(tmp_path: Path, hashed_paths):
    """Qwen2.5-0.5B has exactly 24 layers; width 896 is what disqualifies it."""
    root = _base(
        tmp_path / "base",
        {
            "model_type": "qwen2",
            "num_hidden_layers": 24,
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_attention_heads": 14,
        },
    )

    with pytest.raises(ValueError, match="power-of-two input width"):
        resolve_base_model_identity(str(root))

    assert hashed_paths == []


@pytest.mark.parametrize(
    ("field", "value"),
    [("hidden_size", 768), ("intermediate_size", 3000)],
)
def test_every_measured_width_is_gated_before_a_weight_is_read(
    tmp_path: Path, hashed_paths, field: str, value: int
):
    root = _base(tmp_path / "base", _config(**{field: value}))

    with pytest.raises(ValueError, match="power-of-two input width"):
        resolve_base_model_identity(str(root))

    assert hashed_paths == []


def test_the_derived_attention_width_is_the_hidden_size(tmp_path: Path, hashed_paths):
    """With no explicit ``head_dim``, ``o_proj`` is as wide as ``hidden_size``.

    transformers derives ``head_dim = hidden_size // num_attention_heads``, so
    the attention width the gate measures is exactly the hidden width. Gating
    the head count separately would only add a way to refuse a base the
    post-load gate accepts.
    """
    root = _base(tmp_path / "base", _config(hidden_size=768, num_attention_heads=12))

    with pytest.raises(ValueError, match="power-of-two input width"):
        resolve_base_model_identity(str(root))

    assert hashed_paths == []


def test_a_below_group_width_is_refused_before_a_weight_is_read(tmp_path: Path, hashed_paths):
    root = _base(tmp_path / "base", _config(hidden_size=64))

    with pytest.raises(ValueError, match="power-of-two input width"):
        resolve_base_model_identity(str(root))

    assert hashed_paths == []


def test_nested_text_config_widths_are_gated(tmp_path: Path, hashed_paths):
    """A multimodal config carries the text stack under ``text_config``."""
    root = _base(
        tmp_path / "base",
        {
            "model_type": "qwen3_vl",
            "text_config": {"num_hidden_layers": 24, **COMPATIBLE_WIDTHS, "hidden_size": 896},
        },
    )

    with pytest.raises(ValueError, match="power-of-two input width"):
        resolve_base_model_identity(str(root))

    assert hashed_paths == []


def test_a_compatible_base_is_still_fingerprinted(tmp_path: Path, hashed_paths):
    root = _base(tmp_path / "base", _config())

    identity = resolve_base_model_identity(str(root))

    assert identity.startswith(quest.LOCAL_BASE_PREFIX)
    assert hashed_paths == ["config.json", "model.safetensors"]


def test_an_explicit_head_dim_is_gated(tmp_path: Path, hashed_paths):
    """``o_proj`` takes ``num_attention_heads * head_dim``; a config may set it."""
    root = _base(tmp_path / "base", _config(head_dim=300))

    with pytest.raises(ValueError, match="power-of-two input width"):
        resolve_base_model_identity(str(root))

    assert hashed_paths == []


def test_a_sharded_compatible_base_is_still_fingerprinted(tmp_path: Path, hashed_paths):
    root = tmp_path / "base"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(_config()), encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors"}}),
        encoding="utf-8",
    )
    (root / "model-00001-of-00002.safetensors").write_bytes(b"first")

    identity = resolve_base_model_identity(str(root))

    assert identity.startswith(quest.LOCAL_BASE_PREFIX)
    assert hashed_paths == [
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
    ]


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({"model_type": "llama"}, id="no-topology-keys"),
        pytest.param({"model_type": "llama", "num_hidden_layers": 24}, id="widths-absent"),
        pytest.param(
            {"model_type": "llama", "num_hidden_layers": True, **COMPATIBLE_WIDTHS},
            id="bool-is-not-a-layer-count",
        ),
        pytest.param(
            {"model_type": "llama", "num_hidden_layers": "24", **COMPATIBLE_WIDTHS},
            id="string-layer-count",
        ),
        pytest.param(
            {"model_type": "llama", "num_hidden_layers": 24, "hidden_size": "1024"},
            id="string-width",
        ),
    ],
)
def test_an_unreadable_topology_is_left_to_the_post_load_gate(
    tmp_path: Path, hashed_paths, config: dict[str, Any]
):
    """Only a value we can positively disqualify refuses early; the rest loads.

    Refusing on a value we merely cannot parse would reject bases that the
    post-load gate would accept.
    """
    root = _base(tmp_path / "base", config)

    identity = resolve_base_model_identity(str(root))

    assert identity.startswith(quest.LOCAL_BASE_PREFIX)
    assert hashed_paths == ["config.json", "model.safetensors"]


def test_a_malformed_config_is_still_refused(tmp_path: Path, hashed_paths):
    """The fingerprint's own contract is unchanged by the topology gate."""
    root = tmp_path / "base"
    root.mkdir()
    (root / "config.json").write_text("{not json", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"safe-weights")

    with pytest.raises(ValueError, match="config.json is unreadable"):
        resolve_base_model_identity(str(root))

    assert hashed_paths == []


def test_a_hub_id_is_returned_unchanged(hashed_paths):
    assert resolve_base_model_identity("owner/model") == "owner/model"
    assert hashed_paths == []


def test_the_pre_hash_and_post_load_refusals_share_one_message():
    """One constant feeds both gates so the two messages cannot drift."""
    source = Path(quest.__file__).read_text(encoding="utf-8")
    assert quest.LAYER_COUNT_REFUSAL == "QuEST first slice requires the measured 24 blocks"
    assert quest.LAYER_COUNT_REFUSAL in source
    assert source.count('"QuEST first slice requires the measured 24 blocks"') == 1
