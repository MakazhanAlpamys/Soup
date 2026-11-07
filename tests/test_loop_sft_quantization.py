"""``quantization: auto`` in the loop's sft recipe picks ``none`` when the base fits in VRAM.

The recipe constant is final (candidate B in bf16, Soup Zero fit runs, 2026-11-06) and its
``quantization`` is ``auto``. ``auto`` used to write no line, so ``soup train``'s own default
(``4bit``) applied. Now the train stage measures, right before it writes the soup.yaml:

- need = parameter count x 2 bytes x 5; free = free VRAM of the CUDA device, measured in a
  short-lived subprocess so the long-running loop process never holds a CUDA context;
- CUDA available and need <= free: ``quantization: none``; anything else: ``quantization: 4bit``;
- ``4bit`` and ``none`` in the recipe are explicit and never overridden;
- the parameter count is read without loading weights: the safetensors headers of a local model
  directory, else the Hub's header-only metadata (skipped offline), else unknown (so 4bit);
- the decision is a note of the train stage.

Nothing here touches a GPU or the network: the VRAM and parameter-count functions are replaced,
the hub lookup is faked, and the VRAM probe's ``subprocess.run`` is replaced (the one real probe
runs with ``CUDA_VISIBLE_DEVICES=-1``, so it sees no card).
"""

from __future__ import annotations

import ast
import json
import struct
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import soup_cli.utils.loop_stages as ls
from soup_cli.config.schema import SoupConfig
from soup_cli.utils import loop_quantization as lq
from soup_cli.utils import model_shape
from soup_cli.utils.loop_state import LoopState

# The real functions, taken before the suite's autouse stubs replace them for a test.
_REAL_FREE_VRAM = lq.free_vram_bytes
_REAL_HUB_PARAM_COUNT = model_shape._hub_param_count

GIB = 1024**3


def _state(**kwargs) -> LoopState:
    kwargs.setdefault("served_model", "./model")
    kwargs.setdefault("base_model", "org/base")
    kwargs.setdefault("train_method", "sft")
    return LoopState(eval_suite="e", baseline="b", **kwargs)


def _write_safetensors(path: Path, tensors: dict) -> int:
    """A real safetensors file (header + zero data); returns the parameter count."""
    header: dict = {"__metadata__": {"format": "pt"}}
    offset = 0
    total = 0
    for name, shape in tensors.items():
        numel = 1
        for dim in shape:
            numel *= dim
        header[name] = {
            "dtype": "F16", "shape": shape, "data_offsets": [offset, offset + 2 * numel],
        }
        offset += 2 * numel
        total += numel
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))
    return total


@pytest.fixture
def decide(monkeypatch):
    """Patch the two measurements. ``decide(params, free, setting)`` also returns
    ``resolve_sft_quantization``; ``decide.install(params, free)`` only patches.
    ``decide.calls`` counts how often each measurement was asked for."""
    calls = {"params": 0, "free": 0}

    def install(params, free):
        def fake_params(base):
            calls["params"] += 1
            return params

        def fake_free():
            calls["free"] += 1
            return free

        monkeypatch.setattr(lq, "parameter_count", fake_params)
        monkeypatch.setattr(lq, "free_vram_bytes", fake_free)

    def run(params, free, setting="auto"):
        install(params, free)
        return lq.resolve_sft_quantization(setting, "org/base")

    run.install = install
    run.calls = calls
    return run


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


class TestRule:
    def test_the_estimate_is_two_bytes_times_five_per_parameter(self):
        assert lq.estimated_need_bytes(1000) == 10_000

    def test_a_base_that_fits_gets_none(self, decide):
        value, _ = decide(1000, 20_000)

        assert value == "none"

    def test_a_base_that_does_not_fit_gets_4bit(self, decide):
        value, _ = decide(1000, 5_000)

        assert value == "4bit"

    def test_need_equal_to_free_fits(self, decide):
        value, _ = decide(1000, 10_000)

        assert value == "none"

    def test_one_byte_short_does_not_fit(self, decide):
        value, _ = decide(1000, 9_999)

        assert value == "4bit"

    def test_zero_free_memory_gets_4bit(self, decide):
        value, _ = decide(1000, 0)

        assert value == "4bit"

    def test_no_cuda_gets_4bit_however_large_the_free_figure_would_be(self, decide):
        value, note = decide(1000, None)

        assert value == "4bit"
        assert "free VRAM unknown" in note

    def test_an_unknown_parameter_count_gets_4bit_without_asking_the_gpu(self, decide):
        value, note = decide(None, 100 * GIB)

        assert value == "4bit"
        assert note == "sft quantization auto: 4bit (parameter count unknown)"
        assert decide.calls["free"] == 0

    @pytest.mark.parametrize("setting", ["4bit", "none"])
    def test_an_explicit_value_is_never_overridden_or_measured(self, decide, setting):
        # 4bit with a base that fits, none with a base that does not: both stay as written.
        value, note = decide(1000 if setting == "4bit" else 10**12, 20_000, setting)

        assert (value, note) == (setting, None)
        assert decide.calls == {"params": 0, "free": 0}

    def test_an_unknown_setting_is_refused(self, decide):
        with pytest.raises(ValueError, match="quantization"):
            decide(1000, 20_000, "8bit")


class TestNote:
    def test_it_says_what_was_needed_and_what_was_free_when_it_fits(self, decide):
        _, note = decide(500_000_000, 8 * GIB)

        assert note == "sft quantization auto: none (need 4.7 GB, free 8.0 GB)"

    def test_it_says_what_was_needed_and_what_was_free_when_it_does_not(self, decide):
        _, note = decide(5_000_000_000, 2 * GIB)

        assert note == "sft quantization auto: 4bit (need 46.6 GB, free 2.0 GB)"

    def test_it_says_when_the_free_figure_could_not_be_read(self, decide):
        _, note = decide(500_000_000, None)

        assert note == (
            "sft quantization auto: 4bit (free VRAM unknown: no usable CUDA device)"
        )


# ---------------------------------------------------------------------------
# Free VRAM: measured in a short-lived subprocess, never in this process
# ---------------------------------------------------------------------------


@pytest.fixture
def probe(monkeypatch):
    """Replace ``subprocess.run`` for the probe.

    ``probe(stdout, returncode=0, raises=None)`` sets the answer; ``probe.calls`` holds the
    ``(argv, kwargs)`` of every call. No process starts and no GPU is touched.
    """
    calls: list = []
    state: dict = {"stdout": "", "returncode": 0, "raises": None}

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if state["raises"] is not None:
            raise state["raises"]
        return SimpleNamespace(
            returncode=state["returncode"], stdout=state["stdout"], stderr=""
        )

    def answer(stdout="", returncode=0, raises=None):
        state.update(stdout=stdout, returncode=returncode, raises=raises)

    monkeypatch.setattr(lq.subprocess, "run", fake_run)
    answer.calls = calls
    return answer


class TestFreeVram:
    def test_it_is_the_free_figure_the_probe_prints(self, probe):
        probe(f'{{"free": {7 * GIB}, "total": {8 * GIB}}}\n')

        assert _REAL_FREE_VRAM() == 7 * GIB

    def test_zero_free_is_a_figure_not_unknown(self, probe):
        probe('{"free": 0, "total": 5}\n')

        assert _REAL_FREE_VRAM() == 0

    def test_the_last_line_is_the_answer_when_a_library_printed_before_it(self, probe):
        probe('warning: something noisy\n\n{"free": 42, "total": 99}\n')

        assert _REAL_FREE_VRAM() == 42

    def test_a_probe_that_found_no_cuda_is_unknown(self, probe):
        probe('{"free": null}\n')

        assert _REAL_FREE_VRAM() is None

    @pytest.mark.parametrize(
        "raises",
        [
            subprocess.TimeoutExpired(cmd="probe", timeout=60),
            FileNotFoundError("no interpreter"),
            PermissionError("denied"),
            OSError("cannot fork"),
        ],
        ids=["timeout", "missing", "denied", "oserror"],
    )
    def test_a_probe_that_cannot_run_or_times_out_is_unknown(self, probe, raises):
        probe(raises=raises)

        assert _REAL_FREE_VRAM() is None

    def test_a_probe_that_exits_non_zero_is_unknown_whatever_it_printed(self, probe):
        probe('{"free": 7, "total": 9}\n', returncode=1)

        assert _REAL_FREE_VRAM() is None

    @pytest.mark.parametrize(
        "stdout",
        [
            "",
            "\n\n",
            "not json",
            "{",
            "[1, 2]",
            "7",
            '"free"',
            "null",
            "{}",
            '{"total": 9}',
            '{"free": "7"}',
            '{"free": 7.0}',
            '{"free": 1.5}',
            '{"free": -1}',
            '{"free": true}',
            '{"free": false}',
            '{"free": [7]}',
            '{"free": {"a": 1}}',
            '{"free": NaN}',
            '{"free": Infinity}',
        ],
    )
    def test_anything_but_a_non_negative_int_free_figure_is_unknown(self, probe, stdout):
        probe(stdout + "\n")

        assert _REAL_FREE_VRAM() is None

    def test_the_probe_is_an_argv_list_without_a_shell_and_with_the_timeout(self, probe):
        probe('{"free": 7, "total": 9}\n')

        _REAL_FREE_VRAM()

        ((argv, kwargs),) = probe.calls
        assert isinstance(argv, list)
        assert argv == [sys.executable, "-c", lq.PROBE_SCRIPT]
        assert not kwargs.get("shell", False)
        assert kwargs["timeout"] == lq.PROBE_TIMEOUT_S == 60
        assert kwargs["capture_output"] is True
        assert kwargs["stdin"] == subprocess.DEVNULL

    def test_the_probe_script_asks_torch_for_cuda_and_prints_one_json_line(self):
        assert "import torch" in lq.PROBE_SCRIPT
        assert "torch.cuda.is_available()" in lq.PROBE_SCRIPT
        assert "torch.cuda.mem_get_info()" in lq.PROBE_SCRIPT
        assert lq.PROBE_SCRIPT.count("print(") >= 1
        compile(lq.PROBE_SCRIPT, "<probe>", "exec")

    def test_this_process_never_imports_torch_for_a_measurement(self):
        """A fresh interpreter, so a torch loaded by an earlier test cannot hide an import."""
        code = (
            "import subprocess, sys\n"
            "from types import SimpleNamespace\n"
            "subprocess.run = lambda *a, **k: SimpleNamespace(\n"
            "    returncode=0, stdout='{\"free\": 7, \"total\": 9}\\n', stderr='')\n"
            "from soup_cli.utils import loop_quantization as lq\n"
            "assert 'torch' not in sys.modules, 'torch imported with the module'\n"
            "assert lq.free_vram_bytes() == 7\n"
            "assert 'torch' not in sys.modules, 'torch imported by the measurement'\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )

        assert result.returncode == 0, (result.stdout, result.stderr)

    def test_the_real_probe_script_runs_and_sees_no_card_when_none_is_visible(
        self, monkeypatch
    ):
        """The one real subprocess: CUDA_VISIBLE_DEVICES=-1 hides every card from it."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")

        assert _REAL_FREE_VRAM() is None

    def test_a_torch_py_in_the_working_directory_is_never_imported(
        self, tmp_path, monkeypatch
    ):
        (tmp_path / "torch.py").write_text(
            "open('imported-marker', 'w').write('x')\n", "utf-8"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")

        assert _REAL_FREE_VRAM() is None
        assert not (tmp_path / "imported-marker").exists()


# ---------------------------------------------------------------------------
# The parameter count, without weights
# ---------------------------------------------------------------------------


class TestLocalCount:
    def test_it_sums_the_shapes_of_the_headers(self, tmp_path):
        model = tmp_path / "base"
        model.mkdir()
        expected = _write_safetensors(
            model / "model.safetensors", {"a.weight": [4, 8], "b.weight": [3]}
        )

        assert expected == 35
        assert lq.parameter_count(str(model)) == 35

    def test_it_sums_the_shards_of_a_sharded_checkpoint(self, tmp_path):
        model = tmp_path / "base"
        model.mkdir()
        one = _write_safetensors(model / "model-00001-of-00002.safetensors", {"a": [10, 10]})
        two = _write_safetensors(model / "model-00002-of-00002.safetensors", {"b": [5, 5]})

        assert lq.parameter_count(str(model)) == one + two == 125

    def test_a_header_length_is_read_little_endian(self, tmp_path):
        model = tmp_path / "base"
        model.mkdir()
        _write_safetensors(model / "model.safetensors", {"a": [2, 2]})
        raw = (model / "model.safetensors").read_bytes()
        length = struct.unpack("<Q", raw[:8])[0]

        assert 0 < length < 1000
        # Read big-endian, the same eight bytes are an astronomically large header.
        assert int.from_bytes(raw[:8], "big") > 100 * 1024 * 1024
        assert lq.parameter_count(str(model)) == 4

    def test_an_oversize_header_is_refused(self, tmp_path):
        model = tmp_path / "base"
        model.mkdir()
        (model / "model.safetensors").write_bytes(
            struct.pack("<Q", 200 * 1024 * 1024) + b"{" + bytes(64)
        )

        assert lq.parameter_count(str(model)) is None

    def test_a_header_over_the_cap_is_refused_by_its_size_alone(self, tmp_path, monkeypatch):
        """The header is valid JSON; only its length is past the cap, so only the cap says no."""
        model = tmp_path / "base"
        model.mkdir()
        _write_safetensors(model / "model.safetensors", {"a": [2, 2]})
        assert lq.parameter_count(str(model)) == 4  # control: under the real cap it is read

        monkeypatch.setattr("soup_cli.utils.gpu._MAX_ST_HEADER_BYTES", 16)

        assert lq.parameter_count(str(model)) is None

    def test_a_file_shorter_than_its_length_prefix_is_unknown(self, tmp_path):
        model = tmp_path / "base"
        model.mkdir()
        (model / "model.safetensors").write_bytes(b"\x01\x02\x03")

        assert lq.parameter_count(str(model)) is None

    def test_a_header_that_is_not_json_is_unknown(self, tmp_path):
        model = tmp_path / "base"
        model.mkdir()
        (model / "model.safetensors").write_bytes(struct.pack("<Q", 5) + b"not j")

        assert lq.parameter_count(str(model)) is None

    def test_a_directory_without_weights_is_unknown(self, tmp_path):
        model = tmp_path / "base"
        model.mkdir()
        (model / "config.json").write_text("{}", "utf-8")

        assert lq.parameter_count(str(model)) is None

    def test_a_local_directory_is_never_looked_up_on_the_hub(self, tmp_path, monkeypatch):
        model = tmp_path / "base"
        model.mkdir()
        monkeypatch.setattr(model_shape, "_hub_param_count", _REAL_HUB_PARAM_COUNT)
        asked = []
        monkeypatch.setattr(
            "soup_cli.utils.hf.get_hf_api", lambda *a, **k: asked.append("api") or object()
        )

        assert lq.parameter_count(str(model)) is None
        assert asked == []

    @pytest.mark.requires_symlink
    def test_a_symlinked_weights_file_is_refused(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        _write_safetensors(real / "weights.safetensors", {"a": [4, 4]})
        model = tmp_path / "base"
        model.mkdir()
        (model / "model.safetensors").symlink_to(real / "weights.safetensors")

        assert lq.parameter_count(str(model)) is None

    @pytest.mark.requires_symlink
    def test_a_symlinked_model_directory_is_refused(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        _write_safetensors(real / "model.safetensors", {"a": [4, 4]})
        link = tmp_path / "link"
        link.symlink_to(real, target_is_directory=True)

        assert lq.parameter_count(str(link)) is None
        assert lq.parameter_count(str(real)) == 16  # control: the real directory is read


class TestHubCount:
    @pytest.fixture
    def hub(self, monkeypatch):
        """The real hub lookup, with ``get_hf_api`` faked; ``hub.calls`` records the requests."""
        monkeypatch.setattr(model_shape, "_hub_param_count", _REAL_HUB_PARAM_COUNT)
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        calls: list = []

        def install(getter):
            def record(repo, **kwargs):
                calls.append(repo)
                return getter(repo, **kwargs)

            monkeypatch.setattr(
                "soup_cli.utils.hf.get_hf_api",
                lambda *a, **k: SimpleNamespace(get_safetensors_metadata=record),
            )

        install.calls = calls
        return install

    def test_a_repo_id_is_counted_from_the_hub_metadata(self, hub):
        hub(lambda repo, **kw: SimpleNamespace(parameter_count={"BF16": 360_000_000}))

        assert lq.parameter_count("HuggingFaceTB/SmolLM2-360M") == 360_000_000
        assert hub.calls == ["HuggingFaceTB/SmolLM2-360M"]

    def test_the_dtypes_of_a_mixed_checkpoint_are_added_up(self, hub):
        hub(lambda repo, **kw: SimpleNamespace(parameter_count={"BF16": 100, "F32": 20}))

        assert lq.parameter_count("org/base") == 120

    @pytest.mark.parametrize("var", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
    def test_offline_never_asks_the_hub(self, hub, monkeypatch, var):
        hub(lambda repo, **kw: SimpleNamespace(parameter_count={"BF16": 360_000_000}))
        monkeypatch.setenv(var, "1")

        assert lq.parameter_count("org/base") is None
        assert hub.calls == []

    def test_online_the_same_fake_answers_so_the_offline_test_can_fail(self, hub):
        """Control: without the offline variable the fake is reached."""
        hub(lambda repo, **kw: SimpleNamespace(parameter_count={"BF16": 7}))

        assert lq.parameter_count("org/base") == 7

    def test_a_hub_error_is_unknown_and_does_not_leak(self, hub):
        def gated(repo, **kw):
            raise RuntimeError("401 token hf_secret")

        hub(gated)

        assert lq.parameter_count("org/base") is None

    @pytest.mark.parametrize("base", ["../escape", "a b", "", "x" * 400 + "/y/z", "/abs/path"])
    def test_a_string_that_is_not_a_repo_id_is_never_sent_to_the_hub(self, hub, base):
        hub(lambda repo, **kw: SimpleNamespace(parameter_count={"BF16": 7}))

        assert lq.parameter_count(base) is None
        assert hub.calls == []

    def test_a_stalled_lookup_gives_up_at_the_deadline(self, hub, monkeypatch):
        monkeypatch.setattr(model_shape, "HUB_METADATA_DEADLINE_S", 0.2)
        release = threading.Event()

        def stalled(repo, **kw):
            release.wait(10)
            return SimpleNamespace(parameter_count={"BF16": 7})

        hub(stalled)
        try:
            assert lq.parameter_count("org/base") is None
        finally:
            release.set()


# ---------------------------------------------------------------------------
# The rendered config and the train stage
# ---------------------------------------------------------------------------


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("_DEPLOY_POSTER", "_LOAD_POSTER", "_TRACE_DIR_RESOLVER", "_TRAIN_RUNNER"):
        monkeypatch.setattr(ls, name, None)
    return tmp_path


def _render(state: LoopState, notes=None) -> dict:
    return yaml.safe_load(ls._render_train_yaml(state, "d.jsonl", "out", notes=notes))


class TestRender:
    def test_auto_that_fits_renders_none(self, cwd, decide):
        decide.install(500_000_000, 8 * GIB)
        notes: list = []

        rendered = _render(_state(), notes)
        config = SoupConfig(**rendered)

        assert rendered["training"]["quantization"] == "none"
        assert config.training.quantization == "none"
        assert config.training.lora.target_modules == "all-linear"
        assert notes == ["sft quantization auto: none (need 4.7 GB, free 8.0 GB)"]

    def test_auto_that_does_not_fit_renders_4bit(self, cwd, decide):
        decide.install(5_000_000_000, 2 * GIB)
        notes: list = []

        rendered = _render(_state(), notes)
        config = SoupConfig(**rendered)

        assert rendered["training"]["quantization"] == "4bit"
        assert config.training.quantization == "4bit"
        assert notes == ["sft quantization auto: 4bit (need 46.6 GB, free 2.0 GB)"]

    def test_auto_with_an_unknown_size_renders_4bit(self, cwd, decide):
        decide.install(None, 8 * GIB)
        notes: list = []

        rendered = _render(_state(), notes)

        assert rendered["training"]["quantization"] == "4bit"
        assert notes == ["sft quantization auto: 4bit (parameter count unknown)"]

    def test_the_rendered_config_never_says_auto(self, cwd, decide):
        decide.install(500_000_000, 8 * GIB)

        text = ls._render_train_yaml(_state(), "d.jsonl", "out")

        assert "auto" not in yaml.safe_load(text)["training"].values()
        assert yaml.safe_load(text)["training"]["quantization"] in ("none", "4bit")

    @pytest.mark.parametrize("value", ["4bit", "none"])
    def test_an_explicit_value_is_written_as_is_with_no_note(self, cwd, decide, value):
        decide.install(1000 if value == "4bit" else 10**12, 20_000)
        notes: list = []

        rendered = _render(_state(sft_recipe={"quantization": value}), notes)

        assert rendered["training"]["quantization"] == value
        assert notes == []
        assert decide.calls == {"params": 0, "free": 0}

    @pytest.mark.parametrize("method", ["dpo", "kto"])
    def test_dpo_and_kto_do_not_measure(self, cwd, decide, method):
        decide.install(1000, 20_000)
        notes: list = []

        rendered = _render(_state(train_method=method), notes)

        assert "quantization" not in rendered["training"]
        assert notes == []
        assert decide.calls == {"params": 0, "free": 0}

    def test_a_local_model_directory_is_sized_from_its_headers(self, cwd, monkeypatch):
        model = cwd / "base"
        model.mkdir()
        _write_safetensors(model / "model.safetensors", {"a": [100, 100]})  # 10_000 params
        monkeypatch.setattr(lq, "free_vram_bytes", lambda: 100_000)  # need == 100_000: fits
        notes: list = []

        rendered = _render(_state(base_model=str(model)), notes)

        assert rendered["training"]["quantization"] == "none"
        assert notes and notes[0].startswith("sft quantization auto: none (need ")

        monkeypatch.setattr(lq, "free_vram_bytes", lambda: 99_999)  # one byte short
        assert _render(_state(base_model=str(model)))["training"]["quantization"] == "4bit"


class TestTrainStage:
    def _run(self, cwd, monkeypatch, state, returncode=0):
        configs = []

        def trainer(argv, **kwargs):
            with open(argv[argv.index("--config") + 1], encoding="utf-8") as handle:
                configs.append(yaml.safe_load(handle))
            return SimpleNamespace(returncode=returncode)

        monkeypatch.setattr(ls, "_TRAIN_RUNNER", trainer)
        (cwd / "rows.jsonl").write_text('{"messages": []}\n', "utf-8")
        ctx = {"pairs_harvested": 4, "new_pairs": 4, "pairs_path": "rows.jsonl"}
        return ls.train_dpo_from_pairs(state, ctx), configs

    def test_the_decision_is_in_the_notes_and_the_yaml_soup_train_reads(
        self, cwd, monkeypatch, decide
    ):
        decide.install(500_000_000, 8 * GIB)

        out, configs = self._run(cwd, monkeypatch, _state())

        assert out["skipped"] is False, out
        assert out["notes"] == "sft quantization auto: none (need 4.7 GB, free 8.0 GB)"
        (config,) = configs
        assert config["training"]["quantization"] == "none"
        SoupConfig(**config)

    def test_a_failed_run_still_carries_the_note(self, cwd, monkeypatch, decide):
        decide.install(500_000_000, 8 * GIB)

        out, _ = self._run(cwd, monkeypatch, _state(), returncode=1)

        assert out["failed"] is True
        assert out["notes"] == "sft quantization auto: none (need 4.7 GB, free 8.0 GB)"

    def test_the_measurement_comes_after_the_skip_checks(self, cwd, monkeypatch, decide):
        """A skipped iteration measures nothing: no GPU context for a run that never trains."""
        decide.install(500_000_000, 8 * GIB)

        out = ls.train_dpo_from_pairs(_state(), {"pairs_harvested": 0})

        assert out["skipped"] is True
        assert decide.calls == {"params": 0, "free": 0}

    def test_an_explicit_value_leaves_the_notes_alone(self, cwd, monkeypatch, decide):
        decide.install(500_000_000, 8 * GIB)

        out, configs = self._run(
            cwd, monkeypatch, _state(sft_recipe={"quantization": "4bit"})
        )

        assert "notes" not in out
        assert configs[0]["training"]["quantization"] == "4bit"


# ---------------------------------------------------------------------------
# What soup train does with the rendered `none`: a frozen base keeps the checkpoint dtype
# ---------------------------------------------------------------------------


class TestRenderedNoneIsNotFp32:
    """``quantization: none`` from the loop reaches soup train's frozen-base dtype rule.

    The loop's sft config is an all-linear LoRA (r > 0), so ``is_full_finetune`` is false and
    ``SFTTrainerWrapper._resolve_load_dtype`` delegates to ``resolve_frozen_base_load_dtype``:
    ``"auto"``, the checkpoint's own dtype, never the fp32 that ``from_pretrained`` would
    otherwise upcast to.
    """

    @staticmethod
    def _config(decide):
        decide.install(500_000_000, 8 * GIB)
        rendered = yaml.safe_load(ls._render_train_yaml(_state(), "d.jsonl", "out"))
        assert rendered["training"]["quantization"] == "none"
        return SoupConfig(**rendered)

    def test_the_loop_config_is_a_frozen_lora_base(self, decide):
        from soup_cli.trainer.sft import is_full_finetune

        config = self._config(decide)

        assert config.training.lora.target_modules == "all-linear"
        assert config.training.lora.r > 0
        assert is_full_finetune(config.training) is False

    @pytest.mark.parametrize("cuda", [None, False, True])
    def test_the_resolved_dtype_is_auto_never_float32(self, decide, monkeypatch, cuda):
        torch = pytest.importorskip("torch")
        from soup_cli.trainer.sft import SFTTrainerWrapper

        config = self._config(decide)
        device = "cpu"
        if cuda is not None:
            monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
            monkeypatch.setattr("soup_cli.utils.gpu.cuda_supports_bf16", lambda: True)
            device = "cuda"

        resolved = SFTTrainerWrapper(config, device=device)._resolve_load_dtype(config.training)

        assert resolved == "auto"
        assert resolved is not torch.float32

    def test_a_bfloat16_checkpoint_loads_as_bfloat16_with_that_dtype(
        self, decide, tmp_path, monkeypatch
    ):
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")
        from soup_cli.trainer.sft import SFTTrainerWrapper

        checkpoint = tmp_path / "ckpt"
        model_config = transformers.LlamaConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, vocab_size=64, max_position_embeddings=32,
        )
        transformers.LlamaForCausalLM(model_config).to(torch.bfloat16).save_pretrained(
            str(checkpoint)
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        config = self._config(decide)

        resolved = SFTTrainerWrapper(config, device="cpu")._resolve_load_dtype(config.training)
        loaded = transformers.AutoModelForCausalLM.from_pretrained(str(checkpoint), dtype=resolved)

        assert loaded.dtype == torch.bfloat16
        assert loaded.dtype != torch.float32


# ---------------------------------------------------------------------------
# The CLI stays light
# ---------------------------------------------------------------------------


class TestStaysLight:
    def test_the_module_has_no_top_level_torch_import(self):
        with open(lq.__file__, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        top = {
            alias.name.split(".")[0]
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            (node.module or "").split(".")[0]
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
        }

        assert "torch" not in top

    def test_importing_it_and_asking_for_an_explicit_value_loads_no_torch(self):
        code = (
            "import sys\n"
            "from soup_cli.utils import loop_quantization as lq\n"
            "assert lq.resolve_sft_quantization('4bit', 'org/base') == ('4bit', None)\n"
            "assert 'torch' not in sys.modules, 'torch was imported'\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )

        assert result.returncode == 0, (result.stdout, result.stderr)
