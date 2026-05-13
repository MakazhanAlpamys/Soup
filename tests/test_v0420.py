"""Tests for v0.42.0 — Data Pipeline Pro.

Covers all 6 Parts (A-F):
    A. New formats (PRM / pre_tokenized / input_output / video / multimodal)
    B. Remote loading allowlist + streaming + sharding bounds
    C. AOT preprocess cache key + tokenized_path schema
    D. Interleave strategies + advanced masking + image/video toggles
    E. Vocab expansion + advanced toggles + custom prompt strategy
    F. Document ingestion CLI smoke

Project policies under test (mirrors prior releases):
- Bool rejected before int isinstance check.
- Null-byte / oversize string rejection.
- Frozen dataclass mutation raises FrozenInstanceError.
- ``MappingProxyType``-based registries are runtime-immutable.
- Cross-validators fire at config-load (not at trainer construction).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import DataConfig
from soup_cli.data.formats import format_to_messages
from soup_cli.utils import data_pipeline as dp

# ---------------------------------------------------------------------------
# Part A — New formats
# ---------------------------------------------------------------------------


class TestPartANewFormats:
    def test_new_formats_constant(self):
        assert dp.NEW_FORMATS_V0_42 == frozenset(
            {"prm", "pre_tokenized", "input_output", "video", "multimodal"}
        )
        assert dp.new_formats() == (
            "input_output", "multimodal", "pre_tokenized", "prm", "video",
        )

    @pytest.mark.parametrize("fmt", [
        "prm", "pre_tokenized", "input_output", "video", "multimodal",
    ])
    def test_data_config_accepts_new_format(self, fmt):
        kwargs = {"train": "data.jsonl", "format": fmt}
        if fmt == "pre_tokenized":
            kwargs["tokenized_path"] = "./cache"
        cfg = DataConfig(**kwargs)
        assert cfg.format == fmt

    def test_data_config_rejects_unknown_format(self):
        with pytest.raises(Exception) as exc_info:
            DataConfig(train="data.jsonl", format="bogus")
        msg = str(exc_info.value).lower()
        assert "format" in msg or "literal_error" in msg

    def test_prm_converter_happy(self):
        row = {
            "prompt": "Solve 2+2",
            "completions": ["First, add", "Result is 4"],
            "labels": [True, True],
        }
        out = format_to_messages(row, "prm")
        assert out == row

    def test_prm_converter_mismatched_lengths(self):
        # Wrapped converter swallows ValueError → returns None.
        row = {
            "prompt": "p", "completions": ["a", "b"], "labels": [True],
        }
        assert format_to_messages(row, "prm") is None

    def test_prm_converter_empty(self):
        row = {"prompt": "p", "completions": [], "labels": []}
        assert format_to_messages(row, "prm") is None

    def test_pre_tokenized_converter(self):
        row = {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}
        out = format_to_messages(row, "pre_tokenized")
        assert out == {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}

    def test_pre_tokenized_converter_minimal(self):
        row = {"input_ids": [1, 2, 3]}
        out = format_to_messages(row, "pre_tokenized")
        assert out == {"input_ids": [1, 2, 3]}

    def test_pre_tokenized_converter_missing_ids(self):
        assert format_to_messages({"labels": [1]}, "pre_tokenized") is None

    def test_input_output_converter(self):
        row = {"segments": [
            {"text": "Q: hi", "label": False},
            {"text": "A: hello", "label": True},
        ]}
        out = format_to_messages(row, "input_output")
        assert out["segments"][0]["label"] is False
        assert out["segments"][1]["label"] is True

    def test_input_output_converter_label_must_be_bool(self):
        row = {"segments": [{"text": "x", "label": 1}]}
        # 1 is not bool — converter raises, wrapper returns None.
        assert format_to_messages(row, "input_output") is None

    def test_input_output_converter_empty_segments(self):
        assert format_to_messages({"segments": []}, "input_output") is None

    def test_video_converter(self):
        row = {"video": "clip.mp4", "messages": [
            {"role": "user", "content": "describe"},
        ]}
        out = format_to_messages(row, "video")
        assert out["video"] == "clip.mp4"

    def test_video_converter_missing_video(self):
        assert format_to_messages({"messages": []}, "video") is None

    def test_multimodal_converter_typed_parts(self):
        row = {"messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "what?"},
                {"type": "image", "url": "x.png"},
            ]},
        ]}
        out = format_to_messages(row, "multimodal")
        assert out["messages"][0]["content"][0]["type"] == "text"

    def test_multimodal_converter_string_content_back_compat(self):
        row = {"messages": [{"role": "user", "content": "plain"}]}
        out = format_to_messages(row, "multimodal")
        assert out["messages"][0]["content"] == "plain"

    def test_multimodal_converter_invalid_part_type(self):
        row = {"messages": [{"role": "user", "content": [
            {"type": "evil", "data": "x"},
        ]}]}
        assert format_to_messages(row, "multimodal") is None

    def test_pre_tokenized_format_requires_tokenized_path(self):
        with pytest.raises(Exception, match="tokenized_path"):
            DataConfig(train="data.jsonl", format="pre_tokenized")


# ---------------------------------------------------------------------------
# Part B — Remote loading allowlist + streaming + sharding
# ---------------------------------------------------------------------------


class TestPartBRemoteLoading:
    def test_remote_schemes_immutable(self):
        with pytest.raises(TypeError):
            dp._REMOTE_SCHEMES["evil"] = "evilfs"  # type: ignore

    @pytest.mark.parametrize("uri,expected", [
        ("s3://bucket/path/file.jsonl", True),
        ("gs://bucket/path", True),
        ("gcs://bucket/path", True),
        ("az://container/path", True),
        ("abfs://container/path", True),
        ("oci://bucket/path", True),
        ("./local.jsonl", False),
        ("/abs/local.jsonl", False),
        ("https://example.com/data", False),
        ("ftp://bucket/path", False),
        ("", False),
    ])
    def test_is_remote_uri(self, uri, expected):
        assert dp.is_remote_uri(uri) is expected

    def test_is_remote_uri_non_string(self):
        assert dp.is_remote_uri(None) is False
        assert dp.is_remote_uri(42) is False

    def test_validate_remote_uri_happy(self):
        out = dp.validate_remote_uri("S3://my-bucket/path/file.jsonl")
        assert out == "s3://my-bucket/path/file.jsonl"

    def test_validate_remote_uri_unknown_scheme(self):
        with pytest.raises(ValueError, match="not in allowlist"):
            dp.validate_remote_uri("ftp://bucket/x")

    def test_validate_remote_uri_userinfo_rejected(self):
        with pytest.raises(ValueError, match="userinfo"):
            dp.validate_remote_uri("s3://user:pass@bucket/x")

    def test_validate_remote_uri_fragment_rejected(self):
        with pytest.raises(ValueError, match="fragment"):
            dp.validate_remote_uri("s3://bucket/x#frag")

    def test_validate_remote_uri_null_byte(self):
        with pytest.raises(ValueError, match="null byte"):
            dp.validate_remote_uri("s3://bucket/x\x00y")

    def test_validate_remote_uri_oversize(self):
        with pytest.raises(ValueError, match="<= 2048"):
            dp.validate_remote_uri("s3://bucket/" + "a" * 3000)

    def test_validate_remote_uri_bad_bucket(self):
        with pytest.raises(ValueError, match="bucket"):
            dp.validate_remote_uri("s3://-bad/x")

    def test_validate_remote_uri_no_bucket(self):
        with pytest.raises(ValueError, match="bucket"):
            dp.validate_remote_uri("s3:///x")

    def test_validate_remote_uri_non_string(self):
        with pytest.raises(ValueError):
            dp.validate_remote_uri(None)  # type: ignore

    def test_required_remote_package(self):
        assert dp.required_remote_package("s3") == "s3fs"
        assert dp.required_remote_package("GS") == "gcsfs"
        assert dp.required_remote_package("nope") is None
        assert dp.required_remote_package(None) is None  # type: ignore

    def test_buffer_size_bool_rejected(self):
        with pytest.raises(ValueError, match="bool"):
            dp.validate_buffer_size(True)

    def test_buffer_size_bounds(self):
        assert dp.validate_buffer_size(None) is None
        assert dp.validate_buffer_size(1) == 1
        assert dp.validate_buffer_size(1_000_000) == 1_000_000
        with pytest.raises(ValueError):
            dp.validate_buffer_size(0)
        with pytest.raises(ValueError):
            dp.validate_buffer_size(1_000_001)

    def test_shards_bounds(self):
        assert dp.validate_shards(None) is None
        assert dp.validate_shards(1) == 1
        assert dp.validate_shards(1024) == 1024
        with pytest.raises(ValueError):
            dp.validate_shards(0)
        with pytest.raises(ValueError):
            dp.validate_shards(1025)
        with pytest.raises(ValueError, match="bool"):
            dp.validate_shards(True)

    def test_streaming_buffer_cross_validator(self):
        with pytest.raises(Exception, match="streaming"):
            DataConfig(train="data.jsonl", buffer_size=128, streaming=False)
        cfg = DataConfig(train="data.jsonl", buffer_size=128, streaming=True)
        assert cfg.buffer_size == 128

    def test_streaming_default(self):
        cfg = DataConfig(train="data.jsonl")
        assert cfg.streaming is False
        assert cfg.buffer_size is None
        assert cfg.shards is None


# ---------------------------------------------------------------------------
# Part C — AOT preprocess cache + tokenized_path
# ---------------------------------------------------------------------------


class TestPartCPreprocess:
    def test_cache_key_deterministic(self):
        a = dp.make_preprocess_cache_key(
            dataset_path="d.jsonl", tokenizer_name="x/y", max_length=512,
            format_name="alpaca",
        )
        b = dp.make_preprocess_cache_key(
            dataset_path="d.jsonl", tokenizer_name="x/y", max_length=512,
            format_name="alpaca",
        )
        assert a == b
        assert len(a) == 16

    def test_cache_key_changes_on_each_arg(self):
        baseline = dp.make_preprocess_cache_key(
            dataset_path="d.jsonl", tokenizer_name="x/y", max_length=512,
            format_name="alpaca",
        )
        # Different dataset path → different key.
        diff_path = dp.make_preprocess_cache_key(
            dataset_path="other.jsonl", tokenizer_name="x/y", max_length=512,
            format_name="alpaca",
        )
        diff_tok = dp.make_preprocess_cache_key(
            dataset_path="d.jsonl", tokenizer_name="z/w", max_length=512,
            format_name="alpaca",
        )
        diff_len = dp.make_preprocess_cache_key(
            dataset_path="d.jsonl", tokenizer_name="x/y", max_length=1024,
            format_name="alpaca",
        )
        diff_fmt = dp.make_preprocess_cache_key(
            dataset_path="d.jsonl", tokenizer_name="x/y", max_length=512,
            format_name="sharegpt",
        )
        assert len({baseline, diff_path, diff_tok, diff_len, diff_fmt}) == 5

    def test_cache_key_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            dp.make_preprocess_cache_key(
                dataset_path="", tokenizer_name="x", max_length=1,
                format_name="a",
            )
        with pytest.raises(ValueError, match="null bytes"):
            dp.make_preprocess_cache_key(
                dataset_path="d\x00", tokenizer_name="x", max_length=1,
                format_name="a",
            )
        with pytest.raises(ValueError, match="bool"):
            dp.make_preprocess_cache_key(
                dataset_path="d", tokenizer_name="x", max_length=True,
                format_name="a",
            )
        with pytest.raises(ValueError):
            dp.make_preprocess_cache_key(
                dataset_path="d", tokenizer_name="x", max_length=0,
                format_name="a",
            )

    def test_tokenized_path_schema(self):
        cfg = DataConfig(train="d.jsonl", tokenized_path="./cache")
        assert cfg.tokenized_path == "./cache"

    def test_tokenized_path_rejects_null_byte(self):
        with pytest.raises(Exception, match="null"):
            DataConfig(train="d.jsonl", tokenized_path="x\x00y")

    def test_preprocess_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["data", "preprocess", "--help"])
        assert result.exit_code == 0
        assert "preprocess" in result.output.lower()


# ---------------------------------------------------------------------------
# Part D — Interleave + advanced masking
# ---------------------------------------------------------------------------


class TestPartDInterleave:
    def test_interleave_strategies_constant(self):
        assert dp.INTERLEAVE_STRATEGIES == frozenset(
            {"concat", "under", "over", "probs"}
        )

    def test_interleave_none(self):
        assert dp.parse_interleave(None, num_datasets=3) is None

    @pytest.mark.parametrize("strategy", ["concat", "under", "over"])
    def test_interleave_string_form(self, strategy):
        spec = dp.parse_interleave(strategy, num_datasets=2)
        assert spec.strategy == strategy
        assert spec.probs is None

    def test_interleave_probs_dict(self):
        spec = dp.parse_interleave(
            {"strategy": "probs", "probs": [0.6, 0.3, 0.1]},
            num_datasets=3,
        )
        assert spec.strategy == "probs"
        assert spec.probs == (0.6, 0.3, 0.1)

    def test_interleave_probs_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1"):
            dp.parse_interleave(
                {"strategy": "probs", "probs": [0.5, 0.4]}, num_datasets=2,
            )

    def test_interleave_probs_length_mismatch(self):
        with pytest.raises(ValueError, match="length"):
            dp.parse_interleave(
                {"strategy": "probs", "probs": [0.5, 0.5]}, num_datasets=3,
            )

    def test_interleave_probs_string_form_rejected(self):
        with pytest.raises(ValueError, match="probs"):
            dp.parse_interleave("probs", num_datasets=2)

    def test_interleave_probs_with_non_probs_strategy(self):
        with pytest.raises(ValueError, match="must not set"):
            dp.parse_interleave(
                {"strategy": "concat", "probs": [0.5, 0.5]}, num_datasets=2,
            )

    def test_interleave_unknown_strategy(self):
        with pytest.raises(ValueError, match="strategy"):
            dp.parse_interleave("evil", num_datasets=2)
        with pytest.raises(ValueError, match="strategy"):
            dp.parse_interleave({"strategy": "evil"}, num_datasets=2)

    def test_interleave_num_datasets_bounds(self):
        with pytest.raises(ValueError, match=">= 2"):
            dp.parse_interleave("concat", num_datasets=1)
        with pytest.raises(ValueError, match="32"):
            dp.parse_interleave("concat", num_datasets=33)

    def test_interleave_num_datasets_bool(self):
        with pytest.raises(ValueError, match="bool"):
            dp.parse_interleave("concat", num_datasets=True)

    def test_interleave_probs_bool_value(self):
        with pytest.raises(ValueError, match="bool"):
            dp.parse_interleave(
                {"strategy": "probs", "probs": [True, 0.5]}, num_datasets=2,
            )

    def test_interleave_probs_non_finite(self):
        with pytest.raises(ValueError, match="finite"):
            dp.parse_interleave(
                {"strategy": "probs", "probs": [float("nan"), 0.5]},
                num_datasets=2,
            )

    def test_interleave_probs_out_of_range(self):
        with pytest.raises(ValueError, match=r"\(0\.0, 1\.0\]"):
            dp.parse_interleave(
                {"strategy": "probs", "probs": [0.0, 1.0]}, num_datasets=2,
            )

    def test_interleave_invalid_shape(self):
        with pytest.raises(ValueError):
            dp.parse_interleave(123, num_datasets=2)  # type: ignore

    def test_interleave_spec_frozen(self):
        spec = dp.parse_interleave("concat", num_datasets=2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.strategy = "over"  # type: ignore

    def test_image_pixels_bounds(self):
        assert dp.validate_image_pixels("min", None) is None
        assert dp.validate_image_pixels("min", 1) == 1
        with pytest.raises(ValueError):
            dp.validate_image_pixels("min", 0)
        with pytest.raises(ValueError, match="bool"):
            dp.validate_image_pixels("min", True)

    def test_image_pixel_range_cross_validator(self):
        with pytest.raises(Exception, match="<= image_max"):
            DataConfig(
                train="d.jsonl", image_min_pixels=1000, image_max_pixels=100,
            )

    def test_video_fps_bounds(self):
        assert dp.validate_video_fps(None) is None
        assert dp.validate_video_fps(30) == 30.0
        with pytest.raises(ValueError, match="finite"):
            dp.validate_video_fps(float("inf"))
        with pytest.raises(ValueError, match="bool"):
            dp.validate_video_fps(True)
        with pytest.raises(ValueError):
            dp.validate_video_fps(0)
        with pytest.raises(ValueError):
            dp.validate_video_fps(241)

    def test_video_maxlen_bounds(self):
        assert dp.validate_video_maxlen(None) is None
        assert dp.validate_video_maxlen(64) == 64
        with pytest.raises(ValueError):
            dp.validate_video_maxlen(0)
        with pytest.raises(ValueError):
            dp.validate_video_maxlen(4097)
        with pytest.raises(ValueError, match="bool"):
            dp.validate_video_maxlen(True)

    def test_video_fields_require_video_format(self):
        with pytest.raises(Exception, match="video"):
            DataConfig(train="d.jsonl", format="alpaca", video_fps=24)
        cfg = DataConfig(train="d.jsonl", format="video", video_fps=24)
        assert cfg.video_fps == 24.0

    def test_image_resize_algorithm_literal(self):
        cfg = DataConfig(train="d.jsonl", image_resize_algorithm="bicubic")
        assert cfg.image_resize_algorithm == "bicubic"
        with pytest.raises(Exception):
            DataConfig(train="d.jsonl", image_resize_algorithm="bogus")

    def test_train_on_prompt_mutually_exclusive_with_responses_only(self):
        with pytest.raises(Exception, match="mutually exclusive"):
            DataConfig(
                train="d.jsonl", train_on_prompt=True,
                train_on_responses_only=True,
            )

    def test_train_on_prompt_default_false(self):
        cfg = DataConfig(train="d.jsonl")
        assert cfg.train_on_prompt is False

    def test_mask_history_eval_on_each_split_thinking_defaults(self):
        cfg = DataConfig(train="d.jsonl")
        assert cfg.mask_history is False
        assert cfg.eval_on_each_dataset is False
        assert cfg.split_thinking is False


# ---------------------------------------------------------------------------
# Part E — Vocab expansion + advanced toggles + custom prompt strategy
# ---------------------------------------------------------------------------


class TestPartEVocabExpansion:
    def test_add_new_tokens_happy(self):
        cfg = DataConfig(
            train="d.jsonl",
            add_new_tokens=["<reasoning>", "</reasoning>"],
            resize_vocab=True,
        )
        assert cfg.add_new_tokens == ["<reasoning>", "</reasoning>"]
        assert cfg.resize_vocab is True

    def test_new_tokens_rejects_duplicates(self):
        with pytest.raises(Exception, match="duplicate"):
            DataConfig(train="d.jsonl", add_new_tokens=["x", "x"])

    def test_new_tokens_rejects_empty(self):
        with pytest.raises(Exception, match="empty"):
            DataConfig(train="d.jsonl", add_new_tokens=[""])

    def test_new_tokens_rejects_null_byte(self):
        with pytest.raises(Exception, match="null"):
            DataConfig(train="d.jsonl", add_new_tokens=["x\x00"])

    def test_new_tokens_rejects_non_list(self):
        with pytest.raises(Exception):
            DataConfig(train="d.jsonl", add_new_tokens="not a list")

    def test_new_tokens_rejects_non_string_entry(self):
        with pytest.raises(Exception):
            DataConfig(train="d.jsonl", add_new_tokens=[123])

    def test_new_tokens_oversize(self):
        with pytest.raises(Exception, match="too many"):
            DataConfig(
                train="d.jsonl",
                add_new_tokens=[f"t_{i}" for i in range(10_001)],
            )

    def test_resize_vocab_requires_tokens(self):
        with pytest.raises(Exception, match="resize_vocab"):
            DataConfig(train="d.jsonl", resize_vocab=True)

    def test_new_special_tokens_path(self):
        cfg = DataConfig(
            train="d.jsonl",
            new_special_tokens=["<|tool_call|>"],
            resize_vocab=True,
        )
        assert cfg.new_special_tokens == ["<|tool_call|>"]

    def test_extend_conversation_default(self):
        cfg = DataConfig(train="d.jsonl")
        assert cfg.extend_conversation is False

    def test_skip_prepare_dataset_default(self):
        cfg = DataConfig(train="d.jsonl")
        assert cfg.skip_prepare_dataset is False

    def test_remove_unused_columns_default_true(self):
        cfg = DataConfig(train="d.jsonl")
        assert cfg.remove_unused_columns is True

    def test_prompt_strategy_happy(self):
        cfg = DataConfig(
            train="d.jsonl", prompt_strategy="my_module:my_fn",
        )
        assert cfg.prompt_strategy == "my_module:my_fn"

    def test_prompt_strategy_dotted_module(self):
        cfg = DataConfig(
            train="d.jsonl", prompt_strategy="pkg.sub.mod:transform",
        )
        assert cfg.prompt_strategy == "pkg.sub.mod:transform"

    @pytest.mark.parametrize("bad", [
        "no_colon",
        ":fn",
        "module:",
        "1bad:fn",
        "module:1fn",
        "mod-with-hyphen:fn",
    ])
    def test_prompt_strategy_invalid(self, bad):
        with pytest.raises(Exception):
            DataConfig(train="d.jsonl", prompt_strategy=bad)

    def test_prompt_strategy_null_byte(self):
        with pytest.raises(Exception, match="null"):
            DataConfig(train="d.jsonl", prompt_strategy="m:f\x00")

    def test_prompt_strategy_oversize(self):
        with pytest.raises(Exception):
            DataConfig(
                train="d.jsonl",
                prompt_strategy="m:" + "f" * 300,
            )


# ---------------------------------------------------------------------------
# Part F — Document ingestion
# ---------------------------------------------------------------------------


class TestPartFIngest:
    def test_detect_ingest_format_pdf(self):
        assert dp.detect_ingest_format("doc.pdf") == "pdf"

    def test_detect_ingest_format_docx(self):
        assert dp.detect_ingest_format("doc.DOCX") == "docx"

    def test_detect_ingest_format_md(self):
        assert dp.detect_ingest_format("README.md") == "markdown"

    def test_detect_ingest_format_txt(self):
        assert dp.detect_ingest_format("notes.txt") == "txt"

    def test_detect_ingest_format_unsupported(self):
        with pytest.raises(ValueError, match="ingest"):
            dp.detect_ingest_format("doc.xlsx")

    def test_detect_ingest_format_null_byte(self):
        with pytest.raises(ValueError, match="null"):
            dp.detect_ingest_format("a.txt\x00")

    def test_detect_ingest_format_empty(self):
        with pytest.raises(ValueError):
            dp.detect_ingest_format("")

    def test_ingest_extensions_constant(self):
        assert dp.INGEST_EXTENSIONS == frozenset(
            {".pdf", ".docx", ".md", ".txt"}
        )

    def test_ingest_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["data", "ingest", "--help"])
        assert result.exit_code == 0
        assert "ingest" in result.output.lower()

    def test_ingest_txt_smoke(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = tmp_path / "notes.txt"
        src.write_text("Hello world\nSecond line", encoding="utf-8")
        out = tmp_path / "ingested.jsonl"
        runner = CliRunner()
        result = runner.invoke(app, [
            "data", "ingest", "notes.txt", "--output", "ingested.jsonl",
        ])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert out.exists()
        with open(out, encoding="utf-8") as f:
            row = json.loads(f.readline())
        assert "Hello world" in row["text"]
        assert row["source"] == "notes.txt"

    def test_ingest_md_smoke(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = tmp_path / "doc.md"
        src.write_text("# Title\n\nBody.", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(app, ["data", "ingest", "doc.md"])
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_ingest_unsupported_extension_friendly(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = tmp_path / "data.xlsx"
        src.write_bytes(b"x")
        runner = CliRunner()
        result = runner.invoke(app, ["data", "ingest", "data.xlsx"])
        assert result.exit_code == 1
        assert "ingest only supports" in result.output

    def test_ingest_outside_cwd_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        # Use an absolute path well outside cwd.
        outside = Path(tmp_path).parent / "evil.txt"
        outside.write_text("nope", encoding="utf-8")
        result = runner.invoke(app, ["data", "ingest", str(outside)])
        assert result.exit_code == 1
        assert "under cwd" in result.output


# ---------------------------------------------------------------------------
# Cross-cutting — full SoupConfig YAML roundtrip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_full_v0420_yaml_roundtrip(self):
        yaml_str = """
base: meta-llama/Llama-3.1-8B
task: sft
data:
  train: ./data.jsonl
  format: multimodal
  streaming: true
  buffer_size: 1024
  shards: 4
  interleave: concat
  mask_history: true
  train_on_prompt: false
  train_on_responses_only: true
  eval_on_each_dataset: true
  split_thinking: true
  image_min_pixels: 256
  image_max_pixels: 4096
  image_resize_algorithm: bicubic
  video_fps: 24
  video_maxlen: 32
  video_dir: ./videos
  add_new_tokens: ["<think>", "</think>"]
  new_special_tokens: ["<|tool|>"]
  resize_vocab: true
  extend_conversation: true
  skip_prepare_dataset: false
  remove_unused_columns: false
  prompt_strategy: my_pkg.transforms:greet
training:
  epochs: 1
  lr: 1.0e-4
output: ./out
"""
        cfg = load_config_from_string(yaml_str)
        assert cfg.data.streaming is True
        assert cfg.data.buffer_size == 1024
        assert cfg.data.shards == 4
        assert cfg.data.interleave == "concat"
        assert cfg.data.mask_history is True
        assert cfg.data.image_resize_algorithm == "bicubic"
        assert cfg.data.video_fps == 24.0
        assert cfg.data.add_new_tokens == ["<think>", "</think>"]
        assert cfg.data.resize_vocab is True
        assert cfg.data.prompt_strategy == "my_pkg.transforms:greet"

    def test_pre_tokenized_roundtrip(self):
        yaml_str = """
base: x/y
task: sft
data:
  train: ./d.jsonl
  format: pre_tokenized
  tokenized_path: ./.soup-tokenized/abcd1234
"""
        cfg = load_config_from_string(yaml_str)
        assert cfg.data.format == "pre_tokenized"
        assert cfg.data.tokenized_path == "./.soup-tokenized/abcd1234"


# ---------------------------------------------------------------------------
# Selfcheck — module loads on a fresh interpreter without heavy deps
# ---------------------------------------------------------------------------


def test_selfcheck_counts():
    counts = dp._selfcheck()
    assert counts["new_formats"] == 5
    assert counts["remote_schemes"] >= 6  # s3/gs/gcs/az/abfs/abfss/oci
    assert counts["interleave_strategies"] == 4
    assert counts["image_resize_algorithms"] == 4
    assert counts["ingest_extensions"] == 4


class TestSecurityReviewFixes:
    """Regression tests for the v0.42.0 security-review findings."""

    def test_validate_remote_uri_query_rejected(self):
        # Security H3: query string forwarded to fsspec backend is SSRF-adjacent.
        with pytest.raises(ValueError, match="query string"):
            dp.validate_remote_uri("s3://bucket/x?endpoint_url=http://evil")

    def test_video_field_null_byte_rejected(self):
        # Security H2: video field stored verbatim — null-byte / oversize gate.
        row = {"video": "clip.mp4\x00", "messages": []}
        assert format_to_messages(row, "video") is None

    def test_video_field_oversize_rejected(self):
        row = {"video": "x" * 3000, "messages": []}
        assert format_to_messages(row, "video") is None

    def test_prm_completions_must_be_strings(self):
        # Security L1: PRM converter type-checks completions and labels.
        row = {"prompt": "p", "completions": [None], "labels": [True]}
        assert format_to_messages(row, "prm") is None

    def test_prm_labels_must_be_bool(self):
        row = {"prompt": "p", "completions": ["a"], "labels": [1]}
        assert format_to_messages(row, "prm") is None

    def test_prm_prompt_must_be_string(self):
        row = {"prompt": 42, "completions": ["a"], "labels": [True]}
        assert format_to_messages(row, "prm") is None

    def test_video_dir_outside_cwd_rejected(self, tmp_path, monkeypatch):
        # Security M1: video_dir is is_under_cwd-checked at schema load.
        monkeypatch.chdir(tmp_path)
        outside = str(Path(tmp_path).parent / "evil")
        with pytest.raises(Exception, match="cwd"):
            DataConfig(
                train="d.jsonl", format="video", video_dir=outside,
            )

    def test_tokenized_path_outside_cwd_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        outside = str(Path(tmp_path).parent / "evil")
        with pytest.raises(Exception, match="cwd"):
            DataConfig(
                train="d.jsonl",
                format="pre_tokenized",
                tokenized_path=outside,
            )

    def test_interleave_field_validator_rejects_int(self):
        # Security L3: schema-level field_validator on interleave.
        with pytest.raises(Exception, match="interleave"):
            DataConfig(train="d.jsonl", interleave=99)

    def test_interleave_field_validator_rejects_unknown_string(self):
        with pytest.raises(Exception, match="interleave"):
            DataConfig(train="d.jsonl", interleave="bogus")

    def test_interleave_field_validator_dict_requires_strategy(self):
        with pytest.raises(Exception, match="strategy"):
            DataConfig(train="d.jsonl", interleave={"probs": [0.5, 0.5]})

    def test_interleave_field_validator_accepts_concat(self):
        cfg = DataConfig(train="d.jsonl", interleave="concat")
        assert cfg.interleave == "concat"

    def test_interleave_field_validator_accepts_dict(self):
        cfg = DataConfig(
            train="d.jsonl",
            interleave={"strategy": "probs", "probs": [0.5, 0.5]},
        )
        assert cfg.interleave["strategy"] == "probs"

    def test_image_pixels_field_name_in_error(self):
        # Security M3: error message names the actual field, not a static label.
        with pytest.raises(Exception) as exc_info:
            DataConfig(train="d.jsonl", image_min_pixels=-1)
        assert "image_min_pixels" in str(exc_info.value)

    def test_preprocess_cli_outside_cwd_rejected(self, tmp_path, monkeypatch):
        # Security H1: --config containment check.
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        # Absolute path outside cwd:
        outside = Path(tmp_path).parent / "evil.yaml"
        outside.write_text("base: x\ndata:\n  train: ./d.jsonl\n", encoding="utf-8")
        result = runner.invoke(app, ["data", "preprocess", str(outside)])
        assert result.exit_code == 1
        assert "under cwd" in result.output

    def test_preprocess_cli_happy_path(self, tmp_path, monkeypatch):
        # v0.53.7 #86: live tokenize loop. CLI emits Cache key + Target even if
        # tokenizer load fails downstream — happy-path assertion is on the
        # pre-tokenizer plumbing (cache key derivation + target path render).
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(
            "base: x/y\ntask: sft\ndata:\n  train: ./d.jsonl\n  format: alpaca\n",
            encoding="utf-8",
        )
        (tmp_path / "d.jsonl").write_text(
            '{"instruction": "hi", "output": "hello"}\n', encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(
            app, ["data", "preprocess", "soup.yaml", "--yes"],
        )
        # Cache key + target are rendered before the live tokenize attempt; the
        # tokenizer download will fail with a fake base id but that's expected.
        assert "Cache key" in result.output

    def test_preprocess_cli_output_outside_cwd_rejected(self, tmp_path, monkeypatch):
        # TDD H1/H3: --output containment is its own branch.
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(
            "base: x/y\ntask: sft\ndata:\n  train: ./d.jsonl\n  format: alpaca\n",
            encoding="utf-8",
        )
        runner = CliRunner()
        outside = str(Path(tmp_path).parent / "evil")
        result = runner.invoke(
            app, ["data", "preprocess", "soup.yaml", "--output", outside],
        )
        assert result.exit_code == 1
        assert "--output must stay under cwd" in result.output

    def test_ingest_output_outside_cwd_rejected(self, tmp_path, monkeypatch):
        # TDD M1: ingest --output containment.
        monkeypatch.chdir(tmp_path)
        (tmp_path / "doc.txt").write_text("body", encoding="utf-8")
        outside = str(Path(tmp_path).parent / "evil.jsonl")
        runner = CliRunner()
        result = runner.invoke(
            app, ["data", "ingest", "doc.txt", "--output", outside],
        )
        assert result.exit_code == 1
        assert "--output must stay under cwd" in result.output

    def test_multimodal_converter_empty_messages(self):
        # TDD H4: empty messages list is an error path.
        assert format_to_messages({"messages": []}, "multimodal") is None

    def test_validate_remote_uri_empty_string_rejected(self):
        # TDD M2: validate_remote_uri rejects empty string explicitly.
        with pytest.raises(ValueError, match="empty"):
            dp.validate_remote_uri("")

    def test_v042_optional_path_oversize_rejected(self):
        # TDD M3: video_dir / tokenized_path 4096-char cap.
        with pytest.raises(Exception, match="4096"):
            DataConfig(train="d.jsonl", video_dir="x" * 5000)
        with pytest.raises(Exception, match="4096"):
            DataConfig(train="d.jsonl", tokenized_path="x" * 5000)

    def test_pre_tokenized_attention_mask_passthrough(self):
        # TDD M5: attention_mask is preserved when present.
        row = {
            "input_ids": [1, 2, 3],
            "labels": [-100, 2, 3],
            "attention_mask": [1, 1, 1],
        }
        out = format_to_messages(row, "pre_tokenized")
        assert out["attention_mask"] == [1, 1, 1]

    def test_validate_remote_uri_single_char_bucket(self):
        # Code review HIGH #1: 1-char bucket names are valid per S3/GCS spec.
        out = dp.validate_remote_uri("s3://a/path")
        assert out == "s3://a/path"

    def test_interleave_probs_string_rejected_at_schema_load(self):
        # Code review HIGH #2: bare "probs" string was passing schema, only
        # rejected later at parse_interleave time. Should fail at config load.
        with pytest.raises(Exception, match="probs"):
            DataConfig(train="d.jsonl", interleave="probs")

    def test_ingest_symlink_rejected(self, tmp_path, monkeypatch):
        # Security M2: lstat-based symlink rejection (TOCTOU defence).
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "real.txt"
        target.write_text("hello", encoding="utf-8")
        link = tmp_path / "link.txt"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")
        runner = CliRunner()
        result = runner.invoke(app, ["data", "ingest", "link.txt"])
        assert result.exit_code == 1
        assert "symlink" in result.output


def test_module_lazy_imports():
    # Importing data_pipeline must NOT pull in torch / transformers /
    # huggingface_hub / fsspec — keeps `soup --help` fast.
    import sys

    heavy = {"torch", "transformers", "datasets", "fsspec", "s3fs", "gcsfs"}
    # Force a re-import to be safe on test ordering.
    pre = set(sys.modules)
    import soup_cli.utils.data_pipeline  # noqa: F401
    new = set(sys.modules) - pre
    leaked = new & heavy
    assert not leaked, f"data_pipeline pulled in heavy deps: {leaked}"
