"""``soup rewind`` previews a ``training.curriculum: true`` run (#2972).

With ``curriculum`` on, the SFT wrapper sorts the train rows short to long
before the trainer and the flight recorder see them, so the logged row ids and
the dataset fingerprint are taken over that sorted list. ``load_row_previews``
rebuilt the rows in file order, the fingerprints never matched, and every
curriculum run was told its unchanged dataset "differs from the one this run
trained on", with no previews.

The first case is driven for real: a tiny checkpoint trains on CPU and writes
the log, then the previews are rebuilt from the config the tracker records for
the run. The others are hand-built and need no training: the MLX backend does
not sort, ``curriculum: false`` keeps the file order, and a dataset edited after
a curriculum run is still refused with the mismatch line.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.conftest import strip_ansi

#: Answer lengths in words, deliberately out of file order and all distinct, so
#: the sorted list differs from the file at every position but the shortest.
_WORD_COUNTS = (5, 1, 7, 3, 8, 2, 6, 4)


def _requires_train_extra():
    for mod in ("torch", "transformers", "peft", "trl", "datasets"):
        pytest.importorskip(mod, reason=f"{mod} is only in the [train] extra")


@pytest.fixture(autouse=True)
def _no_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_DISABLED", "true")


# A real, tiny checkpoint on disk. Duplicated from
# test_issue1223_evaluate_val_split.py deliberately, as the suites around it do,
# so this file stays runnable on its own.
def _tiny_llama_dir(root: Path, vocab: int = 64, hidden: int = 64) -> str:
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = root / "model"
    weights.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))

    words = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    for word in ("hello", "world", "hi", "good", "answer", "bad"):
        words[word] = len(words)
    tok = Tokenizer(models.WordLevel(vocab=words, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    ).save_pretrained(str(weights))
    return str(weights)


def _answer(count: int) -> str:
    return " ".join(["hello"] * count)


def _write_rows(path: Path, word_counts) -> None:
    """One chat row per entry, its answer ``count`` words long, in the given order."""
    rows = [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": _answer(count)},
            ]
        }
        for count in word_counts
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _train_curriculum_run(tmp_path, monkeypatch):
    """Train for real; return the tracker's config dict and the rewind log path."""
    import yaml

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.data.loader import load_dataset
    from soup_cli.trainer.sft import SFTTrainerWrapper

    monkeypatch.chdir(tmp_path)
    base = _tiny_llama_dir(tmp_path)
    _write_rows(tmp_path / "train.jsonl", _WORD_COUNTS)
    cfg = load_config_from_string(
        yaml.safe_dump(
            {
                "base": base,
                "task": "sft",
                "backend": "transformers",
                "modality": "text",
                "data": {
                    "train": str(tmp_path / "train.jsonl"),
                    "format": "chatml",
                    "val_split": 0.0,
                    "max_length": 64,
                    "chat_template": "chatml",
                },
                "training": {
                    "batch_size": 2,
                    "gradient_accumulation_steps": 1,
                    "quantization": "none",
                    "epochs": 1,
                    "logging_steps": 1,
                    "save_steps": 1000,
                    "curriculum": True,
                    "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
                },
                "output": str(tmp_path / "out"),
            }
        )
    )
    # What `soup train` hands the wrapper, and what its tracker records.
    wrapper = SFTTrainerWrapper(cfg, device="cpu")
    wrapper.setup(load_dataset(cfg.data))
    wrapper.train()
    recorded = json.loads(json.dumps(cfg.model_dump(), default=str))
    return recorded, Path(wrapper._output_dir) / "rewind.jsonl"


def _read_log(path: Path) -> tuple[str, list[tuple[int, int]]]:
    """The header's fingerprint, and every ``(row, supervised tokens)`` logged."""
    fingerprint = ""
    logged = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("kind") == "header":
            fingerprint = record["dataset_fingerprint"]
        elif record.get("kind") == "batch":
            logged.extend(zip(record["rows"], record["row_tokens"]))
    return fingerprint, logged


class TestRewindPreviewsForARealCurriculumRun:
    def test_an_unchanged_dataset_is_previewed_in_the_order_the_run_trained(
        self, tmp_path, monkeypatch
    ):
        _requires_train_extra()
        from soup_cli.utils.rewind import load_row_previews

        config, log_path = _train_curriculum_run(tmp_path, monkeypatch)
        fingerprint, logged = _read_log(log_path)
        n_rows = len(_WORD_COUNTS)
        assert sorted(row for row, _ in logged) == list(range(n_rows))

        row_ids = [row for row, _ in logged]
        previews = load_row_previews(config, row_ids, expected_fingerprint=fingerprint)

        assert previews is not None, "an unchanged dataset was reported as changed"
        # Row i of a curriculum run is the i-th shortest answer.
        for row in row_ids:
            assert previews[row] == _answer(sorted(_WORD_COUNTS)[row])
        # The recorder's own token counts say the same: they rise with the row id.
        tokens_by_row = dict(logged)
        counts = [tokens_by_row[row] for row in range(n_rows)]
        assert all(a < b for a, b in zip(counts, counts[1:])), counts


# ------------------------------------------------------------ hand-built --


def _hand_built_config(tmp_path: Path, *, curriculum: bool, backend: str) -> dict:
    """The config the tracker records (``json.dumps(model_dump(), default=str)``)."""
    from soup_cli.config.schema import SoupConfig

    data_path = tmp_path / "train.jsonl"
    _write_rows(data_path, _WORD_COUNTS)
    cfg = SoupConfig.model_validate(
        {
            "base": "HuggingFaceTB/SmolLM2-135M",
            "task": "sft",
            "backend": backend,
            "data": {"train": str(data_path), "val_split": 0.0},
            "training": {"curriculum": curriculum},
        }
    )
    return json.loads(json.dumps(cfg.model_dump(), default=str))


def _loaded_rows(config: dict) -> list:
    """The train rows exactly as ``soup train`` loads them, in file order."""
    from soup_cli.config.schema import SoupConfig
    from soup_cli.data.loader import load_dataset

    return list(load_dataset(SoupConfig.model_validate(config).data)["train"])


def _answer_words(row) -> int:
    return len(row["messages"][-1]["content"].split())


@pytest.fixture
def plain_console(monkeypatch):
    """A wide, colourless console, so Rich neither wraps nor colours the output.

    ``commands/rewind.py`` builds its module-level ``Console`` at import, so the
    console itself is replaced for the test (as ``test_rewind_command.py`` does).
    """
    from rich.console import Console

    import soup_cli.commands.rewind as rewind_cmd

    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("CLICOLOR_FORCE", raising=False)
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(
        rewind_cmd,
        "console",
        Console(force_terminal=False, no_color=True, width=10_000, soft_wrap=True),
    )


class TestRewindPreviewsHandBuilt:
    def test_mlx_keeps_the_file_order_with_curriculum_on(self, tmp_path):
        """``trainer/mlx_sft.py`` does not sort: its recorder numbers the file order."""
        from soup_cli.monitoring.rewind_log import dataset_fingerprint
        from soup_cli.utils.rewind import load_row_previews

        config = _hand_built_config(tmp_path, curriculum=True, backend="mlx")
        fingerprint = dataset_fingerprint(_loaded_rows(config))
        row_ids = list(range(len(_WORD_COUNTS)))

        previews = load_row_previews(config, row_ids, expected_fingerprint=fingerprint)

        assert previews is not None, "an unchanged MLX dataset was reported as changed"
        assert [previews[row] for row in row_ids] == [_answer(c) for c in _WORD_COUNTS]

    def test_curriculum_off_is_unchanged(self, tmp_path):
        from soup_cli.monitoring.rewind_log import dataset_fingerprint
        from soup_cli.utils.rewind import load_row_previews

        config = _hand_built_config(tmp_path, curriculum=False, backend="transformers")
        fingerprint = dataset_fingerprint(_loaded_rows(config))
        row_ids = list(range(len(_WORD_COUNTS)))

        previews = load_row_previews(config, row_ids, expected_fingerprint=fingerprint)

        assert previews is not None
        assert [previews[row] for row in row_ids] == [_answer(c) for c in _WORD_COUNTS]

    def test_a_dataset_edited_after_the_run_still_gives_the_mismatch_line(
        self, tmp_path, plain_console, monkeypatch
    ):
        import soup_cli.experiment.tracker as trk
        from soup_cli.cli import app
        from soup_cli.monitoring.rewind_log import RewindLog, dataset_fingerprint

        config = _hand_built_config(tmp_path, curriculum=True, backend="transformers")
        # What the recorder of a curriculum run fingerprints: the rows shortest
        # answer first, ordered here by hand rather than by the code under test.
        trained = sorted(_loaded_rows(config), key=_answer_words)
        log = RewindLog(
            tmp_path / RewindLog.FILENAME,
            backend="transformers",
            task="sft",
            n_rows=len(trained),
            batch_size=2,
            grad_accum=1,
            dataset_fingerprint=dataset_fingerprint(trained),
        )
        for step in range(1, len(trained) // 2 + 1):
            rows = [2 * step - 2, 2 * step - 1]
            log.record_batch(
                step=step, micro=0, rows=rows, row_loss=[1.0, 1.0], row_tokens=[10, 10]
            )
        log.close()

        target = {
            "run_id": "r1",
            "output_dir": str(tmp_path),
            "status": "completed",
            "config_json": json.dumps(config),
        }
        monkeypatch.setattr(trk.ExperimentTracker, "__init__", lambda self, *a, **k: None)
        monkeypatch.setattr(trk.ExperimentTracker, "list_runs", lambda self, limit=50: [target])
        monkeypatch.setattr(trk.ExperimentTracker, "get_run", lambda self, rid: target)
        mismatch = (
            "Preview withheld: the dataset on disk differs from the one this run "
            "trained on (fingerprint mismatch)."
        )

        # Unchanged: the last step's rows 6 and 7 are the two longest answers.
        result = CliRunner().invoke(app, ["rewind", "--step", "4"])
        assert result.exit_code == 0, result.output
        assert _answer(max(_WORD_COUNTS)) in result.output
        assert mismatch not in result.output

        # Edited after the run: the longest answer gains a word. Same row count,
        # same order, different text, so the fingerprint test refuses it.
        _write_rows(
            tmp_path / "train.jsonl",
            [c + 1 if c == max(_WORD_COUNTS) else c for c in _WORD_COUNTS],
        )
        result = CliRunner().invoke(app, ["rewind", "--step", "4"])
        assert result.exit_code == 0, result.output
        assert mismatch in result.output
        assert "hello" not in result.output
        assert "Step 4 rows" in " ".join(strip_ansi(result.output).split())
