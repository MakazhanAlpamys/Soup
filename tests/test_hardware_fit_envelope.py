"""The train pre-flight refuses a peak that does not fit only inside the range it was fitted on.

The VRAM estimate (``utils/vram_estimate.py``) is fitted on dense models of 3B and
below on one 6 GB card (``benchmarks/gate-2049-resident-vram.md``). Outside that range
(a larger model, a mixture-of-experts model, ``medium`` / ``selective`` checkpointing,
a fused-loss flag, ``backend: unsloth``) an estimate that does not fit is printed as a
warning and the run continues; inside it the refusal stays and names
``--allow-oom-attempt``. Outside it a run is still refused when its floor alone, the
weights, the CUDA context and the state of the parameters it trains, is more than the card
holds: those terms are byte counts, not fitted coefficients.

No GPU here: the gate is called with a card size, and the model's ``config.json`` is
replaced by the dict a Hub lookup would return.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import io
import typing

import pytest
import typer
import yaml
from rich.console import Console

from soup_cli.commands import train as train_mod
from soup_cli.commands.train import _hardware_fit_preflight
from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import TEMPLATES, SoupConfig
from soup_cli.utils import hardware_fit, model_shape
from soup_cli.utils.vram_estimate import GB, OVERHEAD_FIXED_BYTES
from tests.conftest import strip_ansi

# Llama-3.1-8B: 8.03B parameters, a 128,256-token vocabulary.
_LLAMA_8B = {
    "model_type": "llama",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "vocab_size": 128256,
}
_LLAMA_8B_PARAMS = 8_030_261_248

# Qwen2.5-0.5B: 0.49B parameters, inside the calibrated range.
_SMALL = {
    "model_type": "qwen2",
    "hidden_size": 896,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "intermediate_size": 4864,
    "vocab_size": 151936,
    "tie_word_embeddings": True,
}
_SMALL_PARAMS = 494_032_768

# Llama-2-7B: 6.74B parameters; a full fine-tune holds 27 GB of fp32 weights and as much
# again in gradients.
_LLAMA_7B = {
    "model_type": "llama",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "intermediate_size": 11008,
    "vocab_size": 32000,
}
_LLAMA_7B_PARAMS = 6_738_415_616

# Qwen3-4B: 4.02B parameters, the next size up from the calibrated range.
_QWEN_4B = {
    "model_type": "qwen3",
    "hidden_size": 2560,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 9728,
    "vocab_size": 151936,
    "tie_word_embeddings": True,
}
_QWEN_4B_PARAMS = 4_022_468_096

# Llama-3.1-70B: 70.55B parameters; its 4-bit weights alone are about 35 GB.
_LLAMA_70B = {
    "model_type": "llama",
    "hidden_size": 8192,
    "num_hidden_layers": 80,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "intermediate_size": 28672,
    "vocab_size": 128256,
}
_LLAMA_70B_PARAMS = 70_553_706_496

# Qwen2.5-14B: 14.77B parameters; its bf16 weights alone are about 30 GB.
_QWEN_14B = {
    "model_type": "qwen2",
    "hidden_size": 5120,
    "num_hidden_layers": 48,
    "num_attention_heads": 40,
    "num_key_value_heads": 8,
    "intermediate_size": 13824,
    "vocab_size": 152064,
}
_QWEN_14B_PARAMS = 14_770_033_664

#: A 6 GB card: too small for either model once the logits are priced.
SIX_GB_CARD = {"memory_total_bytes": int(6e9)}
#: Holds the 8B model's 4-bit weights (7.9 GB) and the CUDA context, but not its peak.
TWELVE_GB_CARD = {"memory_total_bytes": int(12e9)}
BIG_CARD = {"memory_total_bytes": int(10**15)}


def _plain(text: str) -> str:
    """ANSI stripped and whitespace collapsed, as the CLI-output assertions need."""
    return " ".join(strip_ansi(text).split())


@pytest.fixture
def shape_of(monkeypatch):
    """Pick the model the Hub lookup 'finds': ``shape_of(config, params)``."""

    def pick(config: dict, params: int) -> None:
        monkeypatch.setattr(model_shape, "_hub_lookup", lambda repo, **kw: (dict(config), None))
        monkeypatch.setattr(model_shape, "_hub_param_count", lambda repo, **kw: params)

    return pick


@pytest.fixture
def screen(monkeypatch):
    """The pre-flight's console, rendered wide enough that no line wraps."""
    buffer = io.StringIO()
    monkeypatch.setattr(
        train_mod, "console", Console(file=buffer, width=400, force_terminal=False)
    )
    return lambda: _plain(buffer.getvalue())


def _yaml(*, backend: str | None = None, task: str = "sft", **training) -> str:
    document = {
        "base": "someone/model",
        "task": task,
        "data": {"train": "train.jsonl", "max_length": 2048},
        "training": {"quantization": "4bit", "batch_size": 4, **training},
    }
    if backend:
        document["backend"] = backend
    return yaml.safe_dump(document)


def _run(cfg_text: str, *, card=SIX_GB_CARD, allow_oom_attempt=False) -> int | None:
    """Run the pre-flight; the exit code if it refused, else ``None``."""
    try:
        _hardware_fit_preflight(
            load_config_from_string(cfg_text), card, allow_oom_attempt=allow_oom_attempt
        )
    except typer.Exit as exc:
        return exc.exit_code
    return None


OUTSIDE = "outside the calibrated range ("
OVERSTATE = "the estimate may overstate the peak"


class TestInsideTheRange:
    def test_a_dense_model_of_3b_and_below_is_still_refused(self, shape_of, screen):
        shape_of(_SMALL, _SMALL_PARAMS)
        assert _run(_yaml()) == 1
        text = screen()
        assert "Hardware-fit gate" in text
        assert "Predicted peak VRAM" in text

    def test_the_refusal_names_the_override_and_the_range(self, shape_of, screen):
        shape_of(_SMALL, _SMALL_PARAMS)
        _run(_yaml())
        text = screen()
        assert "inside the calibrated range (dense models of 3B and below" in text
        assert "one 6 GB card" in text
        assert "--allow-oom-attempt launches it anyway" in text
        assert OUTSIDE not in text

    def test_full_gradient_checkpointing_is_inside(self, shape_of, screen):
        shape_of(_SMALL, _SMALL_PARAMS)
        assert _run(_yaml(gradient_checkpointing=True)) == 1

    def test_allow_oom_attempt_warns_and_launches_without_the_range_line(
        self, shape_of, screen
    ):
        shape_of(_SMALL, _SMALL_PARAMS)
        assert _run(_yaml(), allow_oom_attempt=True) is None
        text = screen()
        assert "Hardware-fit warning" in text
        assert "--allow-oom-attempt set: launching anyway." in text
        assert OUTSIDE not in text

    def test_a_card_that_fits_is_let_through_without_a_word(self, shape_of, screen):
        shape_of(_SMALL, _SMALL_PARAMS)
        assert _run(_yaml(), card=BIG_CARD) is None
        assert screen() == ""


class TestOutsideTheRange:
    def _assert_warned(self, screen, reason: str) -> None:
        text = screen()
        assert "Hardware-fit warning" in text
        assert "Hardware-fit gate" not in text
        assert "Predicted peak VRAM" in text
        assert OUTSIDE in text
        assert reason in text
        assert OVERSTATE in text
        assert "launching anyway" in text

    def test_a_model_above_3b_is_warned_about_and_launched(self, shape_of, screen):
        shape_of(_LLAMA_8B, _LLAMA_8B_PARAMS)
        assert _run(_yaml(gradient_checkpointing=True), card=TWELVE_GB_CARD) is None
        self._assert_warned(screen, "8.0B parameters")

    def test_the_review_case_batch_4_on_a_24_gib_card_is_not_refused(self, shape_of, screen):
        # Llama-3.1-8B, 4-bit, 2,048 tokens, batch 4, checkpointing: priced at 29 GB.
        shape_of(_LLAMA_8B, _LLAMA_8B_PARAMS)
        card = {"memory_total_bytes": int(24 * 1024**3)}
        assert _run(_yaml(gradient_checkpointing=True), card=card) is None
        self._assert_warned(screen, "8.0B parameters")

    def test_a_mixture_of_experts_model_is_warned_about(self, shape_of, screen):
        shape_of({**_SMALL, "num_local_experts": 8}, _SMALL_PARAMS)
        assert _run(_yaml()) is None
        self._assert_warned(screen, "mixture-of-experts")

    @pytest.mark.parametrize("tier", ["medium", "selective"])
    def test_medium_and_selective_checkpointing_are_warned_about(
        self, shape_of, screen, tier
    ):
        shape_of(_SMALL, _SMALL_PARAMS)
        card = {"memory_total_bytes": int(2e9)}
        assert _run(_yaml(gradient_checkpointing=tier), card=card) is None
        self._assert_warned(screen, f"{tier} gradient checkpointing")

    @pytest.mark.parametrize("flag", ["use_liger", "use_cut_ce"])
    def test_a_fused_loss_flag_is_warned_about(self, shape_of, screen, flag):
        shape_of(_SMALL, _SMALL_PARAMS)
        assert _run(_yaml(**{flag: True})) is None
        self._assert_warned(screen, flag)

    def test_backend_unsloth_is_warned_about(self, shape_of, screen):
        shape_of(_SMALL, _SMALL_PARAMS)
        assert _run(_yaml(backend="unsloth")) is None
        self._assert_warned(screen, "backend unsloth")

    def test_every_reason_is_listed_when_several_apply(self, shape_of, screen):
        shape_of({**_LLAMA_8B, "num_local_experts": 8}, _LLAMA_8B_PARAMS)
        card = TWELVE_GB_CARD
        assert _run(_yaml(use_liger=True, gradient_checkpointing="medium"), card=card) is None
        text = screen()
        for reason in (
            "8.0B parameters",
            "mixture-of-experts",
            "medium gradient checkpointing",
            "use_liger",
        ):
            assert reason in text

    def test_the_warning_is_the_same_with_allow_oom_attempt(self, shape_of, screen):
        shape_of(_LLAMA_8B, _LLAMA_8B_PARAMS)
        assert _run(_yaml(), card=TWELVE_GB_CARD, allow_oom_attempt=True) is None
        self._assert_warned(screen, "8.0B parameters")


_WEIGHTS_DO_NOT_FIT = [
    pytest.param(_LLAMA_70B, _LLAMA_70B_PARAMS, {}, 24e9, id="llama-70b-4bit-on-24-gb"),
    pytest.param(
        _QWEN_14B, _QWEN_14B_PARAMS, {"quantization": "none"}, 8e9, id="qwen-14b-bf16-on-8-gb"
    ),
    pytest.param(_LLAMA_8B, _LLAMA_8B_PARAMS, {}, 4e9, id="llama-8b-4bit-on-4-gb"),
]


class TestWeightsThatDoNotFitTheCard:
    """Outside the range a peak that does not fit is a warning; the weights alone are exact.

    The weights are bytes per parameter times the parameter count and the CUDA context is a
    constant, so "outside the calibrated range" is no reason to start a run whose weights
    and context alone are more than the card holds. (The estimate's `overhead` term is not
    that context: it adds 5% of the whole peak, activations included.)
    """

    @pytest.mark.parametrize(("config", "params", "training", "card_bytes"), _WEIGHTS_DO_NOT_FIT)
    def test_it_is_refused_although_the_run_is_outside_the_range(
        self, shape_of, screen, config, params, training, card_bytes
    ):
        shape_of(config, params)
        card = {"memory_total_bytes": int(card_bytes)}
        assert _run(_yaml(**training), card=card) == 1
        text = screen()
        assert "Hardware-fit gate" in text
        assert OUTSIDE in text
        assert "not an extrapolation" in text
        assert OVERSTATE not in text
        assert "pass --allow-oom-attempt to try anyway" in text
        assert "Warning only" not in text

    @pytest.mark.parametrize(("config", "params", "training", "card_bytes"), _WEIGHTS_DO_NOT_FIT)
    def test_allow_oom_attempt_still_launches_it(
        self, shape_of, screen, config, params, training, card_bytes
    ):
        shape_of(config, params)
        card = {"memory_total_bytes": int(card_bytes)}
        assert _run(_yaml(**training), card=card, allow_oom_attempt=True) is None
        text = screen()
        assert "Hardware-fit warning" in text
        assert "--allow-oom-attempt set: launching anyway." in text
        assert "not an extrapolation" in text
        assert OVERSTATE not in text

    @pytest.mark.parametrize(
        ("factor", "refused"), [(0.99, True), (1.01, False)], ids=["just-below", "just-above"]
    )
    def test_the_line_is_the_floor_and_not_the_whole_peak(
        self, shape_of, screen, factor, refused
    ):
        shape_of(_LLAMA_8B, _LLAMA_8B_PARAMS)
        text = _yaml(gradient_checkpointing=True)
        inp = train_mod._build_hardware_fit_input(load_config_from_string(text))
        parts = hardware_fit.estimate_peak_vram_gb(inp)
        # Weights, the CUDA context, and the adapters with their gradients and optimizer state.
        floor_gb = (
            parts.weights_gb + OVERHEAD_FIXED_BYTES / GB
            + parts.adapters_gb + parts.gradients_gb + parts.optimizer_gb
        )
        # Both cards are below the peak, so only the floor tells them apart; and the floor is
        # more than the weights and the context alone, which the first cases above price.
        assert parts.total_gb * 1.1 > floor_gb * 1.01 + 0.5
        assert floor_gb > parts.weights_gb + OVERHEAD_FIXED_BYTES / GB + 0.1
        card = {"memory_total_bytes": int(floor_gb * factor * 1e9)}
        assert _run(text, card=card) == (1 if refused else None)
        shown = screen()
        assert ("Hardware-fit gate" in shown) is refused
        assert ("Hardware-fit warning" in shown) is not refused
        assert ("not an extrapolation" in shown) is refused
        assert (OVERSTATE in shown) is not refused


_FULL = {"quantization": "none", "lora": {"r": 0}}

#: (model, parameters, training, card in bytes, what the refusal says the floor is made of)
_TRAINED_STATE_DOES_NOT_FIT = [
    pytest.param(
        _LLAMA_7B, _LLAMA_7B_PARAMS, _FULL, 80e9,
        "weights, CUDA context, gradients and optimizer state",
        id="llama-7b-full-adamw-on-80-gb",
    ),
    pytest.param(
        _LLAMA_7B, _LLAMA_7B_PARAMS, _FULL, 40e9,
        "weights, CUDA context, gradients and optimizer state",
        id="llama-7b-full-adamw-on-40-gb",
    ),
    pytest.param(
        _QWEN_4B, _QWEN_4B_PARAMS, _FULL, 24e9,
        "weights, CUDA context, gradients and optimizer state",
        id="qwen-4b-full-adamw-on-24-gb",
    ),
    pytest.param(
        _LLAMA_7B, _LLAMA_7B_PARAMS, {**_FULL, "optimizer": "sgd"}, 40e9,
        "weights, CUDA context and gradients",
        id="llama-7b-full-sgd-on-40-gb",
    ),
    pytest.param(
        _LLAMA_7B, _LLAMA_7B_PARAMS, {**_FULL, "optimizer": "adamw_bnb_8bit"}, 60e9,
        "weights, CUDA context, gradients and optimizer state",
        id="llama-7b-full-8bit-adamw-on-60-gb",
    ),
    # Every AdamW the floor charges in full: the weights and gradients (54.4 GB) fit 80 GB,
    # the 54 GB of AdamW state on top do not.
    pytest.param(
        _LLAMA_7B, _LLAMA_7B_PARAMS, {**_FULL, "optimizer": "adamw_torch_fused"}, 80e9,
        "weights, CUDA context, gradients and optimizer state",
        id="llama-7b-full-fused-adamw-on-80-gb",
    ),
    pytest.param(
        _LLAMA_7B, _LLAMA_7B_PARAMS, {**_FULL, "optimizer": "schedule_free_adamw"}, 80e9,
        "weights, CUDA context, gradients and optimizer state",
        id="llama-7b-full-schedule-free-adamw-on-80-gb",
    ),
    # GaLore replaces the optimizer, so its state is not counted, but the full gradient is.
    pytest.param(
        _LLAMA_7B, _LLAMA_7B_PARAMS,
        {**_FULL, "optimizer": "adamw_torch", "use_galore": True}, 50e9,
        "weights, CUDA context and gradients",
        id="llama-7b-full-galore-on-50-gb",
    ),
    # The DPO trainer does not apply GaLore (``task`` goes to ``_yaml``): it keeps AdamW,
    # whose state is counted.
    pytest.param(
        _LLAMA_7B, _LLAMA_7B_PARAMS, {**_FULL, "task": "dpo", "use_galore": True}, 80e9,
        "weights, CUDA context, gradients and optimizer state",
        id="llama-7b-full-dpo-galore-on-80-gb",
    ),
]

#: Runs whose floor fits although the peak does not: the warning, and the run starts.
_FLOOR_FITS = [
    pytest.param(
        _LLAMA_7B, _LLAMA_7B_PARAMS, {"quantization": "none", "batch_size": 8}, 80e9,
        id="llama-7b-lora-on-80-gb",
    ),
    pytest.param(
        _QWEN_4B, _QWEN_4B_PARAMS, {"quantization": "none"}, 24e9, id="qwen-4b-lora-on-24-gb"
    ),
    # Fused into the backward pass: no full gradient is held, though the estimate charges one.
    pytest.param(
        _LLAMA_7B, _LLAMA_7B_PARAMS, {**_FULL, "optimizer": "lomo"}, 40e9,
        id="llama-7b-full-lomo-on-40-gb",
    ),
    # An optimizer the estimate has no figure for is priced as AdamW: no state term.
    pytest.param(
        _LLAMA_7B, _LLAMA_7B_PARAMS, {**_FULL, "optimizer": "apollo_adamw"}, 60e9,
        id="llama-7b-full-apollo-on-60-gb",
    ),
    # A paged optimizer moves its state to host memory under pressure: no state term.
    pytest.param(
        _LLAMA_7B, _LLAMA_7B_PARAMS, {**_FULL, "optimizer": "paged_adamw_8bit"}, 60e9,
        id="llama-7b-full-paged-8bit-adamw-on-60-gb",
    ),
    # The estimate's figure for these optimizers is more than they hold, so no state term.
    # adalomo: no gradient either, so the weights and the context (27.5 GB) are the floor;
    # with its 2 bytes a parameter of state it was 40.9 GB.
    pytest.param(
        _LLAMA_7B, _LLAMA_7B_PARAMS, {**_FULL, "optimizer": "adalomo"}, 38e9,
        id="llama-7b-full-adalomo-on-38-gb",
    ),
    # Adafactor factors its second moment (the estimate charges 4 bytes a parameter): the
    # floor is 32.7 GB; with the state it was 48.8 GB.
    pytest.param(
        _QWEN_4B, _QWEN_4B_PARAMS, {**_FULL, "optimizer": "adafactor"}, 42.5e9,
        id="qwen-4b-full-adafactor-on-42-gb",
    ),
    # lion_8bit keeps one 8-bit moment (the estimate charges two): 54.4 GB, was 67.9 GB.
    pytest.param(
        _LLAMA_7B, _LLAMA_7B_PARAMS, {**_FULL, "optimizer": "lion_8bit"}, 60e9,
        id="llama-7b-full-lion-8bit-on-60-gb",
    ),
    # GaLore swaps adamw_torch for galore_adamw, whose state is low-rank: 54.4 GB, was 108.3.
    pytest.param(
        _LLAMA_7B, _LLAMA_7B_PARAMS,
        {**_FULL, "optimizer": "adamw_torch", "use_galore": True}, 60e9,
        id="llama-7b-full-galore-adamw-on-60-gb",
    ),
    # Whatever optimizer is set: GaLore replaces it (54.4 GB; with 8-bit AdamW's state 67.9).
    pytest.param(
        _LLAMA_7B, _LLAMA_7B_PARAMS,
        {**_FULL, "optimizer": "adamw_bnb_8bit", "use_galore": True}, 60e9,
        id="llama-7b-full-galore-over-8bit-adamw-on-60-gb",
    ),
    # These train a subset of the parameters but are priced as fully trainable.
    pytest.param(
        _LLAMA_7B, _LLAMA_7B_PARAMS, {"quantization": "none", "lisa_enabled": True}, 40e9,
        id="llama-7b-lisa-on-40-gb",
    ),
    pytest.param(
        _LLAMA_7B, _LLAMA_7B_PARAMS,
        {"quantization": "none", "unfrozen_parameters": ["model.layers.0.*"]}, 40e9,
        id="llama-7b-spectrum-on-40-gb",
    ),
    pytest.param(
        _LLAMA_7B, _LLAMA_7B_PARAMS, {**_FULL, "freeze_ratio": 0.5}, 40e9,
        id="llama-7b-full-freeze-ratio-on-40-gb",
    ),
    pytest.param(
        _LLAMA_7B, _LLAMA_7B_PARAMS, {**_FULL, "freeze_layers": 16}, 40e9,
        id="llama-7b-full-freeze-layers-on-40-gb",
    ),
    pytest.param(
        _LLAMA_7B, _LLAMA_7B_PARAMS,
        {**_FULL, "expand_layers": 2, "freeze_trainable_layers": 2}, 40e9,
        id="llama-7b-llama-pro-on-40-gb",
    ),
]


def _dispatched_wrappers() -> dict:
    """task -> wrapper class, read from the if-chain in ``trainer/dispatch.py`` (as in
    ``test_issue1497_freeze_refused_outside_sft.py``), so a new task is scanned too."""
    from soup_cli.trainer import dispatch

    found = {}
    for node in ast.walk(ast.parse(inspect.getsource(dispatch))):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if not (isinstance(test.left, ast.Name) and test.left.id == "task"):
            continue
        right = test.comparators[0]
        if isinstance(test.ops[0], ast.Eq) and isinstance(right, ast.Constant):
            tasks = [right.value]
        elif isinstance(test.ops[0], ast.In) and isinstance(right, ast.Tuple):
            tasks = [elt.value for elt in right.elts]
        else:
            continue
        imports = [n for n in node.body if isinstance(n, ast.ImportFrom)]
        assert len(imports) == 1, f"expected one import under task={tasks}"
        module = importlib.import_module(imports[0].module)
        for task in tasks:
            found[task] = getattr(module, imports[0].names[0].name)
    return found


def _applies_galore(cls) -> bool:
    """Whether the wrapper, or a soup class it inherits from, builds the GaLore optimizer."""
    for klass in cls.__mro__:
        if not klass.__module__.startswith("soup_cli."):
            continue
        for node in ast.walk(ast.parse(inspect.getsource(inspect.getmodule(klass)))):
            if isinstance(node, ast.ClassDef) and node.name == klass.__name__:
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Name) and sub.id == "get_galore_optimizer_and_params":
                        return True
    return False


class TestTheFloorIncludesWhatTheRunTrains:
    """A full fine-tune holds gradients and optimizer state for every parameter it trains.

    They are bytes per parameter times the parameter count, like the weights, so a run whose
    weights fit the card but whose gradients and optimizer state do not is refused outside
    the calibrated range as well. A LoRA run trains only its adapters, so the same model
    with LoRA starts with the warning. Where the estimate prices more trainable state than
    the run holds (a subset of the parameters, no full gradient, a state paged to the host,
    an optimizer that holds less than its figure, GaLore) the floor leaves that term out
    rather than refuse a run that could fit.
    """

    def test_the_optimizers_whose_state_is_counted_are_the_adamw_ones(self):
        from soup_cli.utils.optimizer_zoo import SUPPORTED_OPTIMIZERS
        from soup_cli.utils.vram_estimate import OPTIMIZER_BYTES_PER_PARAM

        counted = hardware_fit._EXACT_STATE_OPTIMIZERS
        assert counted == {
            "adamw_torch", "adamw_torch_fused", "adamw_bnb_8bit", "schedule_free_adamw",
        }
        # Each is a name the config accepts and the estimate has its own figure for.
        assert counted <= SUPPORTED_OPTIMIZERS
        assert counted <= set(OPTIMIZER_BYTES_PER_PARAM)
        assert counted <= hardware_fit._VALID_OPTIMIZERS

    def test_the_galore_tasks_are_the_ones_whose_trainer_builds_galore(self):
        wrappers = _dispatched_wrappers()
        assert set(wrappers) == set(typing.get_args(SoupConfig.model_fields["task"].annotation))
        applying = {task for task, cls in wrappers.items() if _applies_galore(cls)}
        assert applying == hardware_fit.GALORE_APPLYING_TASKS
        # The scan itself: it sees SFT's call and tts reaching it through the SFT wrapper.
        assert {"sft", "tts", "pretrain"} <= applying
        assert "dpo" not in applying

    @pytest.mark.parametrize(
        ("config", "params", "training", "card_bytes", "terms"), _TRAINED_STATE_DOES_NOT_FIT
    )
    def test_it_is_refused_although_the_weights_fit(
        self, shape_of, screen, config, params, training, card_bytes, terms
    ):
        shape_of(config, params)
        assert _run(_yaml(**training), card={"memory_total_bytes": int(card_bytes)}) == 1
        text = screen()
        assert "Hardware-fit gate" in text
        assert OUTSIDE in text
        assert f"Its {terms} alone take" in text
        assert "not an extrapolation" in text
        assert OVERSTATE not in text
        assert "pass --allow-oom-attempt to try anyway" in text

    @pytest.mark.parametrize(
        ("config", "params", "training", "card_bytes", "terms"), _TRAINED_STATE_DOES_NOT_FIT
    )
    def test_allow_oom_attempt_still_launches_it(
        self, shape_of, screen, config, params, training, card_bytes, terms
    ):
        shape_of(config, params)
        card = {"memory_total_bytes": int(card_bytes)}
        assert _run(_yaml(**training), card=card, allow_oom_attempt=True) is None
        text = screen()
        assert "Hardware-fit warning" in text
        assert "--allow-oom-attempt set: launching anyway." in text
        assert "not an extrapolation" in text

    @pytest.mark.parametrize(("config", "params", "training", "card_bytes"), _FLOOR_FITS)
    def test_it_starts_with_the_warning_when_the_floor_fits(
        self, shape_of, screen, config, params, training, card_bytes
    ):
        shape_of(config, params)
        assert _run(_yaml(**training), card={"memory_total_bytes": int(card_bytes)}) is None
        text = screen()
        assert "Hardware-fit warning" in text
        assert "Hardware-fit gate" not in text
        assert "Predicted peak VRAM" in text
        assert OVERSTATE in text
        assert "Warning only: launching anyway." in text
        assert "not an extrapolation" not in text


class TestTheRangeCheck:
    """``outside_calibrated_range`` is the one place the envelope is decided."""

    def _inp(self, config=_SMALL, params=_SMALL_PARAMS, **fields):
        shape = model_shape.shape_from_config(
            config, params=params, params_source="safetensors"
        )
        base = dict(
            params_b=params / 1e9,
            seq_len=2048,
            batch_size=1,
            optimizer="adamw_torch",
            quant="4bit",
            peft="qlora",
            gradient_checkpointing=False,
            shape=shape,
        )
        return hardware_fit.HardwareFitInput(**{**base, **fields})

    def test_a_small_dense_run_is_inside(self):
        assert hardware_fit.outside_calibrated_range(self._inp()) == ()

    def test_the_bound_is_between_3b_and_the_next_size_up(self):
        three_b = self._inp(params=3_090_000_000, params_b=3.09)
        four_b = self._inp(params=4_020_000_000, params_b=4.02)
        assert hardware_fit.outside_calibrated_range(three_b) == ()
        assert len(hardware_fit.outside_calibrated_range(four_b)) == 1

    def test_full_checkpointing_and_none_are_inside(self):
        for value in (False, True, "full"):
            assert not hardware_fit.outside_calibrated_range(
                self._inp(gradient_checkpointing=value)
            )

    def test_a_shape_with_no_config_is_inside_when_small(self):
        inp = hardware_fit.HardwareFitInput(
            params_b=1.0, seq_len=2048, batch_size=1, optimizer="adamw_torch",
            quant="none", peft="lora", gradient_checkpointing=False,
        )
        assert hardware_fit.outside_calibrated_range(inp) == ()

    def test_the_backend_is_a_reason_only_when_it_is_unsloth(self):
        inp = self._inp()
        assert hardware_fit.outside_calibrated_range(inp, backend="transformers") == ()
        assert hardware_fit.outside_calibrated_range(inp, backend=None) == ()
        assert hardware_fit.outside_calibrated_range(inp, backend="mlx") == ()
        assert hardware_fit.outside_calibrated_range(inp, backend="unsloth") == (
            "backend unsloth",
        )

    def test_the_reasons_read_as_a_phrase(self):
        reasons = hardware_fit.outside_calibrated_range(
            self._inp(config={**_SMALL, "num_experts": 64}, fused_loss=("use_liger",)),
            backend="unsloth",
        )
        assert reasons == (
            "a mixture-of-experts model",
            "fused-loss flag use_liger",
            "backend unsloth",
        )


class TestTheLongcontextTemplate:
    """The shipped template is an 8B model at 131,072 tokens: priced far above any card."""

    def _template(self, *, liger: bool) -> str:
        text = TEMPLATES["longcontext"]
        assert "# use_liger: true" in text
        return text.replace("# use_liger: true", "use_liger: true") if liger else text

    @pytest.mark.parametrize("liger", [True, False])
    def test_it_is_never_refused_on_a_24_gib_card(self, shape_of, screen, liger):
        shape_of(_LLAMA_8B, _LLAMA_8B_PARAMS)
        card = {"memory_total_bytes": int(24 * 1024**3)}
        assert _run(self._template(liger=liger), card=card) is None
        text = screen()
        assert "Hardware-fit warning" in text
        assert "Hardware-fit gate" not in text

    def test_with_the_fused_loss_on_the_warning_names_it(self, shape_of, screen):
        shape_of(_LLAMA_8B, _LLAMA_8B_PARAMS)
        _run(self._template(liger=True), card={"memory_total_bytes": int(80e9)})
        text = screen()
        assert "use_liger" in text
        assert "overstates the peak" in text

    def test_the_header_says_what_the_pre_flight_does_and_how_to_turn_on_the_fused_loss(self):
        header = [
            line for line in TEMPLATES["longcontext"].splitlines() if line.startswith("#")
        ]
        text = _plain(" ".join(header))
        assert "pre-flight" in text
        assert "warning" in text
        assert "use_liger" in text
        assert "soup-cli[liger]" in text
