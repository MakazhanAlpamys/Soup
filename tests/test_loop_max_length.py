"""The loop writes ``data.max_length`` from the data it built, so ``soup train``'s fit gate fits.

Review 5, F1: the loop's generated recipe left ``data.max_length`` at the schema default (2048).
``soup train`` prices every row of a micro-batch at that length, so 26 rows of about 25 tokens on
Qwen3-0.6B were predicted at 21.7 GB and refused on an 8 GB card (the same run peaks at 1.2 GB).
Now the train stage measures the longest row of the dataset it just built with the base model's
tokenizer (in a short-lived child: transformers imports torch, the daemon must not), rounds it up
and writes it into the recipe; when it cannot measure, the default stays and the note says so.
The gate itself is untouched and ``--allow-oom-attempt`` is never passed.
"""

from __future__ import annotations

import io
import json
import subprocess

import pytest
import yaml

import soup_cli.utils.loop_max_length as lml
import soup_cli.utils.loop_stages as ls
from soup_cli.utils.loop_state import LoopState

#: The real child launcher, taken before the suite's autouse fixture replaces it.
REAL_MEASURE = lml._measure_in_child


class FakeTokenizer:
    """One token per whitespace-separated word, and a chat template of ``<role> text`` lines."""

    def __init__(self, template: bool = True) -> None:
        self.calls = []
        if not template:
            self.apply_chat_template = None  # type: ignore[assignment]

    def __call__(self, text, add_special_tokens=True):
        self.calls.append((text, add_special_tokens))
        return {"input_ids": list(range(len(text.split())))}

    def apply_chat_template(self, messages, tokenize=False):  # noqa: D102
        assert tokenize is False
        return "".join(f"<{m['role']}> {m['content']} <end> " for m in messages)


def _state(**kwargs) -> LoopState:
    base = dict(served_model="org/base", eval_suite="e", baseline="b", base_model="org/base")
    base.update(kwargs)
    return LoopState(**base)


class TestRoundUp:
    @pytest.mark.parametrize(
        ("longest", "expected"),
        [
            (0, 64),       # nothing to measure: the estimate's floor
            (25, 64),      # the review-5 rows
            (48, 64),      # 48 + 16 headroom is exactly one block
            (49, 128),     # one token over: the next block
            (100, 128),
            (300, 320),
            (2032, 2048),  # reaches the ceiling exactly
            (2033, 2048),  # above it: clamped, never above
            (50_000, 2048),
        ],
    )
    def test_round_up_to_a_block_within_the_floor_and_the_ceiling(self, longest, expected):
        assert lml.round_up(longest, ceiling=2048) == expected

    def test_a_ceiling_below_the_block_is_the_ceiling(self):
        assert lml.round_up(70, ceiling=100) == 100

    def test_a_ceiling_below_the_floor_still_gives_the_floor(self):
        # the estimate prices nothing below 64 tokens, so a (degenerate) ceiling cannot lower it
        assert lml.round_up(10, ceiling=32) == lml.MIN_MAX_LENGTH == 64

    def test_the_floor_is_the_estimates_and_the_schemas(self):
        from soup_cli.config.schema import DataConfig
        from soup_cli.utils.hardware_fit import _MIN_SEQ_LEN

        assert lml.MIN_MAX_LENGTH == _MIN_SEQ_LEN == 64
        assert DataConfig.model_fields["max_length"].metadata[0].ge == lml.MIN_MAX_LENGTH

    def test_the_ceiling_is_the_configured_default(self):
        from soup_cli.config.schema import DataConfig

        assert lml.configured_ceiling() == DataConfig.model_fields["max_length"].default == 2048


class TestRowTexts:
    def test_sft_is_the_chat_templated_messages(self):
        row = {"messages": [{"role": "user", "content": "hi there"},
                            {"role": "assistant", "content": "ok"}]}

        texts = lml.row_texts(row, "sft", FakeTokenizer())

        assert texts == ["<user> hi there <end> <assistant> ok <end> "]

    def test_sft_without_a_template_is_plain_role_lines(self):
        row = {"messages": [{"role": "user", "content": "hi"},
                            {"role": "assistant", "content": "ok"}]}

        assert lml.row_texts(row, "sft", FakeTokenizer(template=False)) == [
            "user: hi\nassistant: ok"
        ]

    def test_sft_with_a_template_that_refuses_falls_back_to_plain_lines(self):
        class Refusing(FakeTokenizer):
            def apply_chat_template(self, messages, tokenize=False):
                raise ValueError("roles must alternate")

        row = {"messages": [{"role": "user", "content": "a"}]}

        assert lml.row_texts(row, "sft", Refusing()) == ["user: a"]

    def test_dpo_is_prompt_plus_each_answer(self):
        row = {"prompt": "q", "chosen": "good", "rejected": "bad"}

        assert lml.row_texts(row, "dpo") == ["q\ngood", "q\nbad"]

    def test_kto_is_prompt_plus_completion(self):
        row = {"prompt": "q", "completion": "a", "label": True}

        assert lml.row_texts(row, "kto") == ["q\na"]

    @pytest.mark.parametrize(
        "row",
        [None, [], "text", {}, {"messages": []}, {"messages": "x"}, {"messages": [3]},
         {"prompt": "q"}],
    )
    def test_a_row_without_the_methods_fields_is_ignored(self, row):
        assert lml.row_texts(row, "sft", FakeTokenizer()) == []
        assert lml.row_texts(row, "dpo") == []

    def test_an_unknown_method_measures_nothing(self):
        assert lml.row_texts({"prompt": "q", "chosen": "c"}, "ppo") == []


class TestLongestRow:
    def test_the_longest_row_of_the_rows_is_found(self):
        rows = [
            {"messages": [{"role": "user", "content": "a b"},
                          {"role": "assistant", "content": "c"}]},
            {"messages": [{"role": "user", "content": "a b c d e f"},
                          {"role": "assistant", "content": "g h"}]},
        ]
        tokenizer = FakeTokenizer()

        longest, measured = lml.longest_row_tokens(rows, tokenizer, "sft")

        # "<user> a b c d e f <end> <assistant> g h <end> " = 12 words
        assert (longest, measured) == (12, 2)
        assert all(flag is False for _text, flag in tokenizer.calls)

    def test_dpo_takes_the_longer_of_chosen_and_rejected(self):
        rows = [{"prompt": "a b", "chosen": "c", "rejected": "d e f g"}]

        assert lml.longest_row_tokens(rows, FakeTokenizer(), "dpo") == (6, 1)

    def test_rows_to_ignore_are_not_counted_and_nothing_is_zero(self):
        assert lml.longest_row_tokens([{}, None, "x"], FakeTokenizer(), "sft") == (0, 0)
        assert lml.longest_row_tokens([], FakeTokenizer(), "sft") == (0, 0)


class TestChild:
    def _run(self, tmp_path, rows, loader, method="sft", base="org/base", path=None):
        data = tmp_path / "rows.jsonl"
        data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        out = io.StringIO()
        lml.child_main(
            [base, str(path or data), method], load_tokenizer=loader, out=out,
        )
        return json.loads(out.getvalue().strip().splitlines()[-1])

    def test_it_prints_one_json_line_with_the_longest_row(self, tmp_path):
        rows = [{"messages": [{"role": "user", "content": "a b c"},
                              {"role": "assistant", "content": "d"}]}]
        seen = []

        def loader(base):
            seen.append(base)
            return FakeTokenizer()

        assert self._run(tmp_path, rows, loader) == {"longest": 8, "rows": 1}
        assert seen == ["org/base"]

    def test_a_tokenizer_that_cannot_load_is_an_error_line_not_a_traceback(self, tmp_path):
        def loader(base):
            raise OSError("not found")

        answer = self._run(tmp_path, [{"messages": [{"role": "user", "content": "a"}]}], loader)

        assert answer == {"error": "OSError while loading the tokenizer or tokenising"}

    def test_a_missing_dataset_is_an_error_line(self, tmp_path):
        answer = self._run(tmp_path, [], lambda base: FakeTokenizer(),
                           path=tmp_path / "missing.jsonl")

        assert answer["error"].startswith("the dataset cannot be read")

    def test_a_dataset_with_no_measurable_row_is_an_error_line(self, tmp_path):
        answer = self._run(tmp_path, [{"other": 1}], lambda base: FakeTokenizer())

        assert answer == {"error": "the dataset has no row to measure"}

    def test_a_torn_line_is_skipped_not_fatal(self, tmp_path):
        data = tmp_path / "rows.jsonl"
        data.write_text(
            '{"messages": [{"role": "user", "content": "a b"}]}\n{"half"\n', encoding="utf-8"
        )
        out = io.StringIO()

        lml.child_main(["b", str(data), "sft"], load_tokenizer=lambda b: FakeTokenizer(), out=out)

        assert json.loads(out.getvalue()) == {"longest": 4, "rows": 1}

    def test_wrong_arguments_are_an_error_line(self):
        out = io.StringIO()

        lml.child_main(["only-one"], load_tokenizer=lambda b: FakeTokenizer(), out=out)

        assert "error" in json.loads(out.getvalue())


class TestParent:
    """``_measure_in_child`` against a ``subprocess.run`` stand-in."""

    def _fake_run(self, monkeypatch, *, stdout="", returncode=0, raises=None):
        calls = []

        def run(argv, **kwargs):
            calls.append((argv, kwargs))
            if raises is not None:
                raise raises
            return subprocess.CompletedProcess(argv, returncode, stdout, "")

        monkeypatch.setattr(subprocess, "run", run)
        return calls

    def test_it_runs_the_scrubbed_child_script_without_a_shell(self, monkeypatch):
        calls = self._fake_run(monkeypatch, stdout='{"longest": 40, "rows": 3}\n')

        assert REAL_MEASURE("org/base", "d.jsonl", "sft") == (40, 3)

        argv, kwargs = calls[0]
        assert argv[1] == "-c" and argv[2] == lml.CHILD_SCRIPT
        assert argv[3:] == ["org/base", "d.jsonl", "sft"]
        assert "'', '.'" in lml.CHILD_SCRIPT  # the working directory is dropped from sys.path
        assert kwargs["stdin"] == subprocess.DEVNULL and "shell" not in kwargs
        assert kwargs["timeout"] == lml.CHILD_TIMEOUT_S

    def test_the_last_line_is_the_answer(self, monkeypatch):
        self._fake_run(monkeypatch, stdout='warning: x\n{"longest": 7, "rows": 1}\n\n')

        assert REAL_MEASURE("b", "d", "sft") == (7, 1)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"stdout": '{"error": "OSError while loading the tokenizer"}'},
             "OSError while loading the tokenizer"),
            ({"stdout": ""}, "the tokenizer child printed no answer"),
            ({"stdout": "not json"}, "the tokenizer child printed no answer"),
            ({"stdout": "[1]"}, "the tokenizer child printed no answer"),
            ({"stdout": '{"longest": true, "rows": 1}'}, "printed a bad answer"),
            ({"stdout": '{"longest": -1, "rows": 1}'}, "printed a bad answer"),
            ({"stdout": '{"longest": 5}'}, "printed a bad answer"),
            ({"returncode": 3221225786}, "STATUS_CONTROL_C_EXIT"),
            ({"raises": subprocess.TimeoutExpired("x", 1)}, "took over"),
            ({"raises": FileNotFoundError("python")}, r"could not start \(FileNotFoundError\)"),
        ],
    )
    def test_every_failure_is_a_measure_failed_error_that_says_why(
        self, monkeypatch, kwargs, message
    ):
        self._fake_run(monkeypatch, **kwargs)

        with pytest.raises(lml.MeasureFailedError, match=message):
            REAL_MEASURE("b", "d", "sft")

    def test_the_tokenizer_child_is_started_without_a_console_window(self, monkeypatch):
        import soup_cli.utils.loop_spawn as spawn

        seen = {}

        def run(argv, **kwargs):
            seen.update(kwargs)
            return subprocess.CompletedProcess(argv, 0, '{"longest": 4, "rows": 1}\n', "")

        monkeypatch.setattr(spawn, "child_popen_kwargs", lambda: {"creationflags": 0x08000000})
        monkeypatch.setattr(subprocess, "run", run)

        REAL_MEASURE("b", "d", "sft")

        assert seen["creationflags"] == 0x08000000

    def test_the_real_child_measures_a_real_tokenizer(self, tmp_path):
        """No stand-in: the child script, its sys.path scrub and transformers end to end."""
        pytest.importorskip("transformers")
        pytest.importorskip("tokenizers")
        from tokenizers import Tokenizer, models, pre_tokenizers

        words = ["[UNK]", "user:", "assistant:", "the", "number", "ten", "is", "x", "y", "z"]
        tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        model_dir = tmp_path / "tiny-tokenizer"
        model_dir.mkdir()
        tok.save(str(model_dir / "tokenizer.json"))
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}), encoding="utf-8"
        )
        data = tmp_path / "rows.jsonl"
        data.write_text(
            json.dumps({"messages": [
                {"role": "user", "content": "the number ten"},
                {"role": "assistant", "content": "x y z"},
            ]}) + "\n",
            encoding="utf-8",
        )

        longest, rows = REAL_MEASURE(str(model_dir), str(data), "sft")

        # "user: the number ten\nassistant: x y z": the whitespace pre-tokenizer splits each
        # "role:" into two tokens, so 5 + 5 tokens.
        assert (longest, rows) == (10, 1)


class TestDecision:
    def test_short_rows_get_the_floor_and_the_note_says_how(self):
        decided = lml.decide_max_length(
            "org/base", "d.jsonl", "sft", measure=lambda *_a: (25, 26), ceiling=2048
        )

        assert decided.value == 64
        assert decided.note == (
            "max_length: 64 (longest of 26 rows is 25 tokens, +16, rounded up to a multiple of "
            "64, ceiling 2048)"
        )

    def test_longer_rows_get_the_next_block(self):
        decided = lml.decide_max_length("b", "d", "sft", measure=lambda *_a: (300, 5), ceiling=2048)

        assert decided.value == 320

    def test_a_row_above_the_ceiling_is_the_ceiling_and_the_note_names_it(self):
        decided = lml.decide_max_length(
            "b", "d", "sft", measure=lambda *_a: (3000, 5), ceiling=2048
        )

        assert decided.value == 2048
        assert "above the ceiling 2048" in decided.note

    def test_an_unmeasurable_length_keeps_the_default_and_says_so(self):
        def measure(*_args):
            raise lml.MeasureFailedError("OSError while loading the tokenizer")

        decided = lml.decide_max_length("b", "d", "sft", measure=measure, ceiling=2048)

        assert decided.value is None
        assert decided.note == (
            "max_length: 2048 kept, the longest row could not be measured "
            "(OSError while loading the tokenizer)"
        )

    def test_the_measurement_gets_the_base_the_data_and_the_method(self):
        seen = []

        def measure(base, path, method):
            seen.append((base, path, method))
            return 10, 1

        lml.decide_max_length("org/b", "x.jsonl", "kto", measure=measure, ceiling=2048)

        assert seen == [("org/b", "x.jsonl", "kto")]


class TestTheTrainStageWritesIt:
    @pytest.fixture
    def cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
        for name in ("SOUP_LOOP_TRACE_DIR", "SOUP_LOOP_THUMBS_DB", "SOUP_LOOP_SERVE_ENDPOINT"):
            monkeypatch.delenv(name, raising=False)
        row = {"messages": [{"role": "user", "content": "q"},
                            {"role": "assistant", "content": "a"}]}
        (tmp_path / "sft.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        return tmp_path

    def test_the_rendered_recipe_carries_the_measured_length_and_the_note(
        self, cwd, monkeypatch
    ):
        monkeypatch.setattr(lml, "_measure_in_child", lambda *_a: (25, 26))
        notes: list = []

        text = ls._render_train_yaml(
            _state(train_method="sft"), "sft.jsonl", "out", notes=notes
        )

        config = yaml.safe_load(text)
        assert config["data"] == {
            "train": "sft.jsonl", "format": "chatml", "val_split": 0, "max_length": 64,
        }
        assert any(
            note.startswith("max_length: 64 (longest of 26 rows is 25 tokens") for note in notes
        )

    def test_an_unmeasurable_length_leaves_the_schema_default_and_notes_it(
        self, cwd, monkeypatch
    ):
        def refuse(*_args):
            raise lml.MeasureFailedError("the tokenizer child took over 180 s")

        monkeypatch.setattr(lml, "_measure_in_child", refuse)
        notes: list = []

        config = yaml.safe_load(
            ls._render_train_yaml(_state(train_method="sft"), "sft.jsonl", "out", notes=notes)
        )

        assert "max_length" not in config["data"]
        assert "max_length: 2048 kept, the longest row could not be measured" in " ".join(notes)

    @pytest.mark.parametrize(("method", "fmt"), [("dpo", "dpo"), ("kto", "kto")])
    def test_dpo_and_kto_recipes_carry_it_too(self, cwd, monkeypatch, method, fmt):
        monkeypatch.setattr(lml, "_measure_in_child", lambda *_a: (200, 4))

        config = yaml.safe_load(ls._render_train_yaml(_state(), "sft.jsonl", "out", method))

        assert config["data"]["format"] == fmt and config["data"]["max_length"] == 256

    def test_the_stage_hands_soup_train_that_yaml_and_records_the_value_used(
        self, cwd, monkeypatch
    ):
        monkeypatch.setattr(lml, "_measure_in_child", lambda *_a: (25, 26))
        seen = {}

        def runner(argv, **kwargs):
            config_path = argv[argv.index("--config") + 1]
            with open(config_path, encoding="utf-8") as handle:
                seen["config"] = yaml.safe_load(handle)
            seen["argv"] = list(argv)
            return type("R", (), {"returncode": 0, "stderr": None})()

        monkeypatch.setattr(ls, "_TRAIN_RUNNER", runner)

        out = ls.train_dpo_from_pairs(
            _state(train_method="sft"), {"pairs_path": "sft.jsonl", "pairs_harvested": 1}
        )

        assert seen["config"]["data"]["max_length"] == 64
        assert "max_length: 64 (longest of 26 rows is 25 tokens" in out["notes"]
        # The fit gate is not bypassed: the loop never passes the opt-out.
        assert "--allow-oom-attempt" not in seen["argv"]


class TestTheFitGateAcceptsShortRows:
    """The arithmetic of ``soup train``'s hardware-fit gate for the card of review 5 (no GPU)."""

    #: Qwen3-0.6B's config.json (the numbers that price the run).
    QWEN3_0_6B = {
        "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 1024,
        "intermediate_size": 3072, "num_hidden_layers": 28, "num_attention_heads": 16,
        "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936,
        "tie_word_embeddings": True, "max_position_embeddings": 40960,
    }
    CARD_GB = 8.5  # what the gate measured on the RTX 5070 Laptop: "8.5 GB available"

    @pytest.fixture
    def model_dir(self, tmp_path):
        folder = tmp_path / "qwen3-0.6b"
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps(self.QWEN3_0_6B), encoding="utf-8")
        return folder

    def _report(self, tmp_path, model_dir, monkeypatch, measure):
        from soup_cli.config.schema import SoupConfig
        from soup_cli.utils.hardware_fit import (
            decide_hardware_fit,
            hardware_fit_input_or_reason,
        )

        monkeypatch.chdir(tmp_path)
        (tmp_path / "sft.jsonl").write_text("{}\n", encoding="utf-8")
        monkeypatch.setattr(lml, "_measure_in_child", measure)
        state = _state(
            served_model=str(model_dir), base_model=str(model_dir), train_method="sft",
            sft_recipe={"quantization": "4bit"},
        )
        config = SoupConfig(**yaml.safe_load(
            ls._render_train_yaml(state, "sft.jsonl", "out")
        ))
        inp = hardware_fit_input_or_reason(config, allow_network=False, gpu_memory_gb=8.0)
        assert not hasattr(inp, "reason"), inp
        return config, decide_hardware_fit(inp, available_vram_gb=self.CARD_GB)

    def test_a_recipe_of_short_rows_is_not_refused_on_an_8_gb_card(
        self, tmp_path, model_dir, monkeypatch
    ):
        config, report = self._report(tmp_path, model_dir, monkeypatch, lambda *_a: (25, 26))

        assert config.data.max_length == 64
        assert report.ok, report.peak_vram_gb
        assert report.required_with_margin_gb < self.CARD_GB

    def test_the_same_recipe_left_at_the_default_is_the_one_review_5_saw(
        self, tmp_path, model_dir, monkeypatch
    ):
        def unmeasured(*_args):
            raise lml.MeasureFailedError("no tokenizer")

        config, report = self._report(tmp_path, model_dir, monkeypatch, unmeasured)

        assert config.data.max_length == 2048
        assert not report.ok
        assert report.peak_vram_gb > 15, report.peak_vram_gb  # 21.7 GB in the real run
