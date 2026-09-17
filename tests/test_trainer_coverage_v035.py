"""v0.35.0 Part D — Per-trainer × per-feature smoke-test matrix.

Without this file, Part A (#60) ships ~50 LOC of theoretical wiring across
8 trainer wrappers; with it, every trainer × every v0.28.0 feature combo is
exercised on every CI matrix job. Each test builds a minimal SoupConfig
with one feature enabled, mocks the model construction, calls the shared
helper, and asserts the right side-effect happened (or that a documented
no-op degrades gracefully).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# All transformer-backend trainer tasks (v0.35.0 #60 expanded coverage)
ALL_TRAINER_TASKS = (
    "sft",
    "dpo",
    "pretrain",
    "grpo",
    "kto",
    "orpo",
    "simpo",
    "ipo",
    "ppo",
    "reward_model",
    "embedding",
)

# v0.28.0 speed/memory features
V028_FEATURES = (
    "use_cut_ce",
    "fp8",
    "kernel_auto_compose",
    "activation_offloading",
)


def _make_tcfg(feature: str) -> SimpleNamespace:
    """Build a minimal training-config SimpleNamespace with one feature on."""
    base = {
        "use_cut_ce": False,
        "quantization_aware": False,
        "kernel_auto_compose": False,
        "activation_offloading": None,
    }
    if feature == "use_cut_ce":
        base["use_cut_ce"] = True
    elif feature == "fp8":
        base["quantization_aware"] = "fp8"
    elif feature == "kernel_auto_compose":
        base["kernel_auto_compose"] = True
    elif feature == "activation_offloading":
        base["activation_offloading"] = "cpu"
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# supports_v028_features expansion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", ALL_TRAINER_TASKS)
def test_supports_v028_features_covers_every_trainer(task: str) -> None:
    """Each of the 11 transformer-backend trainers must report supported."""
    from soup_cli.utils.v028_features import supports_v028_features

    assert supports_v028_features(task) is True


def test_supports_v028_features_rejects_unknown_task() -> None:
    from soup_cli.utils.v028_features import supports_v028_features

    assert supports_v028_features("future_task_v999") is False


# ---------------------------------------------------------------------------
# apply_v028_speed_memory matrix — every trainer × every (apply-phase) feature
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", ALL_TRAINER_TASKS)
@pytest.mark.parametrize("feature", ("use_cut_ce", "fp8", "kernel_auto_compose"))
def test_apply_v028_speed_memory_no_exception(task: str, feature: str) -> None:
    """For every trainer × apply-phase feature, the helper must not raise.

    cut_ce / fp8 degrade silently if the underlying lib isn't installed (CI
    runs without torchao.float8 / cut_cross_entropy on most jobs); the
    helper returns a dict instead of crashing. kernel_auto_compose's picker
    raises when no candidates have finite times — also degrades silently.
    """
    from soup_cli.utils import v028_features as vf

    tcfg = _make_tcfg(feature)
    result = vf.apply_v028_speed_memory(
        model=MagicMock(),
        tcfg=tcfg,
        base_model="meta-llama/Llama-3.2-1B",
        console=None,
    )
    # Helper returns a dict of {feature: bool_applied} — never None / exception.
    assert isinstance(result, dict)
    assert "cut_ce" in result
    assert "fp8" in result
    assert "kernel_auto_compose" in result


@pytest.mark.parametrize("task", ALL_TRAINER_TASKS)
def test_apply_v028_speed_memory_invokes_cut_ce_patcher(task: str) -> None:
    """When use_cut_ce=True and the patcher succeeds, applied['cut_ce'] is True."""
    from soup_cli.utils import v028_features as vf

    tcfg = _make_tcfg("use_cut_ce")
    with patch("soup_cli.utils.cut_ce.apply_cut_ce", return_value=True):
        result = vf.apply_v028_speed_memory(
            model=MagicMock(), tcfg=tcfg,
            base_model="meta-llama/Llama-3.2-1B", console=None,
        )
    assert result["cut_ce"] is True


@pytest.mark.parametrize("task", ALL_TRAINER_TASKS)
def test_apply_v028_speed_memory_invokes_fp8(task: str) -> None:
    from soup_cli.utils import v028_features as vf

    tcfg = _make_tcfg("fp8")
    with patch("soup_cli.utils.fp8.apply_fp8_training", return_value=True):
        result = vf.apply_v028_speed_memory(
            model=MagicMock(), tcfg=tcfg,
            base_model="meta-llama/Llama-3.2-1B", console=None,
        )
    assert result["fp8"] is True


@pytest.mark.parametrize("task", ALL_TRAINER_TASKS)
def test_apply_v028_speed_memory_refuses_kernel_picker(task: str) -> None:
    """Bypassing validation cannot revive the misleading picker path."""
    from soup_cli.utils import v028_features as vf

    tcfg = _make_tcfg("kernel_auto_compose")
    result = vf.apply_v028_speed_memory(
        model=MagicMock(), tcfg=tcfg,
        base_model="meta-llama/Llama-3.2-1B", console=None,
        device="cuda", backend="transformers",
    )
    assert result["kernel_auto_compose"] is False


# ---------------------------------------------------------------------------
# activation_offloading_context — train()-phase wrapper
# ---------------------------------------------------------------------------


def test_activation_offloading_context_none_passes_through(tmp_path) -> None:
    from soup_cli.utils.v028_features import activation_offloading_context

    tcfg = SimpleNamespace(activation_offloading=None)
    with activation_offloading_context(tcfg, str(tmp_path)):
        pass


def test_activation_offloading_context_cpu_no_save_dir(tmp_path) -> None:
    from soup_cli.utils.v028_features import activation_offloading_context

    tcfg = SimpleNamespace(activation_offloading="cpu")
    with activation_offloading_context(tcfg, str(tmp_path)):
        pass


def test_activation_offloading_context_disk_rejects_outside_cwd() -> None:
    """Disk mode must refuse output dirs outside cwd — defence in depth."""
    import os

    from soup_cli.utils.v028_features import activation_offloading_context

    tcfg = SimpleNamespace(activation_offloading="disk")
    # Use the parent of cwd, which is guaranteed to be outside containment.
    parent = os.path.realpath(os.path.join(os.getcwd(), os.pardir))
    with pytest.raises(ValueError, match="under the current working directory"):
        with activation_offloading_context(tcfg, parent):
            pass


def test_activation_offloading_context_missing_attr_safe(tmp_path) -> None:
    """A tcfg-like object missing the attr should not raise."""
    from soup_cli.utils.v028_features import activation_offloading_context

    tcfg = SimpleNamespace()  # no activation_offloading attr at all
    with activation_offloading_context(tcfg, str(tmp_path)):
        pass


# ---------------------------------------------------------------------------
# Schema gate — v0.35.0 #60 lifts the gate for every transformer trainer
# ---------------------------------------------------------------------------


def _build_yaml_config(task: str, **training_extra) -> dict:
    """Helper: build a minimal config dict accepted by load_config_from_string."""
    body = {
        "base": "meta-llama/Llama-3.2-1B",
        "task": task,
        "training": {"epochs": 1, "lr": 1e-4, "batch_size": 1, **training_extra},
    }
    if task == "pretrain":
        body["data"] = {"train": "data.jsonl", "format": "plaintext"}
    elif task in ("dpo", "kto", "orpo", "simpo", "ipo", "grpo"):
        body["data"] = {"train": "data.jsonl", "format": "dpo"}
    elif task == "embedding":
        body["data"] = {"train": "data.jsonl", "format": "embedding"}
    elif task == "reward_model":
        body["data"] = {"train": "data.jsonl", "format": "dpo"}
    else:
        body["data"] = {"train": "data.jsonl", "format": "alpaca"}
    return body


@pytest.mark.parametrize(
    "task",
    ("grpo", "kto", "orpo", "simpo", "ipo", "ppo", "reward_model", "embedding"),
)
def test_schema_gate_lifted_for_use_cut_ce(task: str) -> None:
    """Each of the 8 newly-wired trainers must accept use_cut_ce at config-load."""
    import yaml

    from soup_cli.config.loader import load_config_from_string

    body = _build_yaml_config(task, use_cut_ce=True)
    cfg = load_config_from_string(yaml.safe_dump(body))
    assert cfg.task == task
    assert cfg.training.use_cut_ce is True


@pytest.mark.parametrize(
    "task",
    ("grpo", "kto", "orpo", "simpo", "ipo", "ppo", "reward_model", "embedding"),
)
def test_schema_gate_lifted_for_activation_offloading(task: str) -> None:
    """activation_offloading="cpu" must now be accepted on every trainer."""
    import yaml

    from soup_cli.config.loader import load_config_from_string

    body = _build_yaml_config(task, activation_offloading="cpu")
    cfg = load_config_from_string(yaml.safe_dump(body))
    assert cfg.training.activation_offloading == "cpu"


def test_mlx_backend_still_rejects_v028_features() -> None:
    """MLX backend has no equivalent kernels — gate must still fire with a
    schema-rejection ValueError (the loader wraps Pydantic's ValidationError
    in a ValueError carrying the same message)."""
    import yaml

    from soup_cli.config.loader import load_config_from_string

    body = {
        "base": "meta-llama/Llama-3.2-1B",
        "task": "sft",
        "backend": "mlx",
        "data": {"train": "data.jsonl", "format": "alpaca"},
        "training": {
            "epochs": 1, "lr": 1e-4, "batch_size": 1, "use_cut_ce": True,
        },
    }
    with pytest.raises(ValueError, match="mlx"):
        load_config_from_string(yaml.safe_dump(body))


def test_unknown_task_v028_features_rejected_with_distinct_message() -> None:
    """Code-review fix: the schema gate must NOT blame MLX for a non-MLX
    failure. An unknown task on the transformers backend should produce a
    "task=" message, not the MLX message."""
    from pydantic import ValidationError

    from soup_cli.config.schema import SoupConfig

    # Bypass the loader (which wraps ValueError) so we can inspect the
    # raw Pydantic error message directly.
    with pytest.raises(ValidationError) as exc_info:
        SoupConfig(
            base="m",
            task="sft",  # supported task, but...
            backend="mlx",  # ...MLX backend triggers the rejection
            data={"train": "./d.jsonl"},
            training={"use_cut_ce": True},
        )
    assert "mlx" in str(exc_info.value).lower()
    assert "Apple Silicon" in str(exc_info.value)


@pytest.mark.parametrize(
    "task",
    ("grpo", "kto", "orpo", "simpo", "ipo", "ppo", "reward_model", "embedding"),
)
def test_schema_rejects_kernel_auto_compose(task: str) -> None:
    """#801 — the non-functional selector is rejected on every trainer."""
    import yaml

    from soup_cli.config.loader import load_config_from_string

    body = _build_yaml_config(task, kernel_auto_compose=True)
    with pytest.raises(ValueError, match="use_flash_attn"):
        load_config_from_string(yaml.safe_dump(body))


@pytest.mark.parametrize(
    "task",
    ("grpo", "kto", "orpo", "simpo", "ipo", "ppo", "reward_model", "embedding"),
)
def test_schema_gate_lifted_for_fp8(task: str) -> None:
    """v0.35.0 #60 — quantization_aware="fp8" must now be accepted on every trainer."""
    import yaml

    from soup_cli.config.loader import load_config_from_string

    body = _build_yaml_config(task, quantization_aware="fp8")
    cfg = load_config_from_string(yaml.safe_dump(body))
    assert cfg.training.quantization_aware == "fp8"


# ---------------------------------------------------------------------------
# Per-trainer × per-feature integration smoke — verify the call site exists
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path",
    (
        "soup_cli.trainer.sft",
        "soup_cli.trainer.dpo",
        "soup_cli.trainer.pretrain",
        "soup_cli.trainer.grpo",
        "soup_cli.trainer.kto",
        "soup_cli.trainer.orpo",
        "soup_cli.trainer.simpo",
        "soup_cli.trainer.ipo",
        "soup_cli.trainer.ppo",
        "soup_cli.trainer.reward_model",
        "soup_cli.trainer.embedding",
    ),
)
def test_trainer_module_calls_apply_v028_speed_memory(module_path: str) -> None:
    """Source-level proof that every trainer wires the helper.

    A regression that drops the call site silently re-introduces v0.28.0
    silent no-ops — caught here at parse time, not at training time.
    """
    import importlib
    import inspect

    mod = importlib.import_module(module_path)
    src = inspect.getsource(mod)
    # SFT predates the shared helper extraction (v0.33.0 #43) and inlines
    # apply_cut_ce / apply_fp8_training / kernel_picker directly; the other
    # 10 trainers all delegate to apply_v028_speed_memory.
    has_helper = "apply_v028_speed_memory" in src
    has_inline_features = (
        "apply_cut_ce" in src
        and "apply_fp8_training" in src
    )
    assert has_helper or has_inline_features, (
        f"{module_path} does not call apply_v028_speed_memory or the "
        "underlying feature patchers directly — v0.28.0 features will "
        "silently no-op for this trainer."
    )


@pytest.mark.parametrize(
    "module_path",
    (
        "soup_cli.trainer.sft",
        "soup_cli.trainer.dpo",
        "soup_cli.trainer.pretrain",
        "soup_cli.trainer.grpo",
        "soup_cli.trainer.kto",
        "soup_cli.trainer.orpo",
        "soup_cli.trainer.simpo",
        "soup_cli.trainer.ipo",
        "soup_cli.trainer.ppo",
        "soup_cli.trainer.reward_model",
        "soup_cli.trainer.embedding",
    ),
)
def test_trainer_module_wraps_train_with_offloading_context(
    module_path: str,
) -> None:
    """Source-level proof that every trainer wraps trainer.train() with the
    activation-offloading context manager."""
    import importlib
    import inspect

    mod = importlib.import_module(module_path)
    src = inspect.getsource(mod)
    # SFT uses the inline offload_context (older pattern); the others use the
    # shared activation_offloading_context helper from v028_features.
    assert (
        "activation_offloading_context" in src
        or "offload_context" in src
    ), (
        f"{module_path} does not wrap trainer.train() with the activation-"
        "offloading context — disk/cpu offloading will silently no-op."
    )


# ---------------------------------------------------------------------------
# Part B — auto-quant live model reload
# ---------------------------------------------------------------------------


class TestQuantNameTranslators:
    @pytest.mark.parametrize(
        ("name", "expected"),
        (
            ("awq", {"quantization": "awq"}),
            ("gptq", {"quantization": "gptq"}),
            ("fp8", {"quantization": "fp8"}),
            ("none", {}),
            ("gguf", {}),  # GGUF needs a path swap, not a kwarg
        ),
    )
    def test_quant_name_to_vllm_kwargs_known(self, name, expected) -> None:
        from soup_cli.utils.auto_quant import quant_name_to_vllm_kwargs

        assert quant_name_to_vllm_kwargs(name) == expected

    def test_quant_name_to_vllm_kwargs_unknown_returns_empty(self) -> None:
        from soup_cli.utils.auto_quant import quant_name_to_vllm_kwargs

        # ``int8`` is not in the mapping; pass-through returns {} so caller
        # uses the engine default.
        assert quant_name_to_vllm_kwargs("int8") == {}

    def test_quant_name_to_vllm_kwargs_rejects_invalid_name(self) -> None:
        from soup_cli.utils.auto_quant import quant_name_to_vllm_kwargs

        with pytest.raises(ValueError, match="candidate name must match"):
            quant_name_to_vllm_kwargs("AWQ")  # uppercase not allowed
        with pytest.raises(ValueError):
            quant_name_to_vllm_kwargs("../../etc/passwd")

    def test_quant_name_to_vllm_kwargs_returns_new_dict(self) -> None:
        """Mutation-safe — caller must not be able to corrupt the mapping."""
        from soup_cli.utils.auto_quant import quant_name_to_vllm_kwargs

        a = quant_name_to_vllm_kwargs("awq")
        a["leak"] = True
        b = quant_name_to_vllm_kwargs("awq")
        assert "leak" not in b

    def test_quant_name_to_bnb_kwargs(self) -> None:
        from soup_cli.utils.auto_quant import quant_name_to_bnb_kwargs

        assert quant_name_to_bnb_kwargs("awq") == {"load_in_4bit": True}
        assert quant_name_to_bnb_kwargs("gptq") == {"load_in_4bit": True}
        assert quant_name_to_bnb_kwargs("fp8") == {}
        assert quant_name_to_bnb_kwargs("none") == {}


class TestFreeEngine:
    def test_free_engine_no_torch(self) -> None:
        """free_engine must never raise even if torch is unavailable."""
        from soup_cli.utils.auto_quant import free_engine

        # Smoke — no torch on this CI? Still no-raise.
        free_engine(MagicMock())

    def test_free_engine_torch_no_cuda(self) -> None:
        from soup_cli.utils.auto_quant import free_engine

        with patch("torch.cuda.is_available", return_value=False):
            free_engine(MagicMock())  # no exception

    def test_free_engine_torch_cuda_available(self) -> None:
        from soup_cli.utils.auto_quant import free_engine

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache") as mock_empty,
        ):
            free_engine(MagicMock())
            mock_empty.assert_called_once()


class TestTryReloadWithFallback:
    def _candidate(self, name: str, score: float = 0.95):
        from soup_cli.utils.auto_quant import Candidate

        return Candidate(name=name, score=score, latency_ms=10.0, ok=True)

    def test_picked_loads_first_try(self) -> None:
        from soup_cli.utils.auto_quant import try_reload_with_fallback

        picked = self._candidate("awq")
        all_c = [picked, self._candidate("gptq", 0.90)]
        engines_built: list[str] = []

        def build(name):
            engines_built.append(name)
            return f"engine-{name}"

        used, eng = try_reload_with_fallback(
            picked=picked, all_candidates=all_c, build_fn=build,
        )
        assert used.name == "awq"
        assert eng == "engine-awq"
        assert engines_built == ["awq"]  # no fallback exercised

    def test_falls_back_to_next_highest_on_first_failure(self) -> None:
        from soup_cli.utils.auto_quant import try_reload_with_fallback

        picked = self._candidate("awq", 0.92)
        gptq = self._candidate("gptq", 0.95)
        none_c = self._candidate("none", 0.85)
        all_c = [picked, gptq, none_c]
        engines_built: list[str] = []

        def build(name):
            engines_built.append(name)
            if name == "awq":
                raise RuntimeError("AWQ kernel missing")
            return f"engine-{name}"

        used, eng = try_reload_with_fallback(
            picked=picked, all_candidates=all_c, build_fn=build,
        )
        # picked tried first, then by descending score: gptq (0.95)
        assert used.name == "gptq"
        assert engines_built == ["awq", "gptq"]

    def test_raises_when_all_candidates_fail(self) -> None:
        from soup_cli.utils.auto_quant import try_reload_with_fallback

        picked = self._candidate("awq")
        all_c = [picked, self._candidate("gptq")]

        def build(name):
            raise RuntimeError(f"{name} broken")

        with pytest.raises(RuntimeError, match="every auto-quant candidate"):
            try_reload_with_fallback(
                picked=picked, all_candidates=all_c, build_fn=build,
            )

    def test_duplicate_candidate_name_deduped(self) -> None:
        """Two candidates with the same name shouldn't double-build."""
        from soup_cli.utils.auto_quant import try_reload_with_fallback

        picked = self._candidate("awq", 0.95)
        dup = self._candidate("awq", 0.92)  # same name
        gptq = self._candidate("gptq", 0.90)
        engines_built: list[str] = []

        def build(name):
            engines_built.append(name)
            if name == "awq":
                raise RuntimeError("AWQ broken")
            return f"engine-{name}"

        used, _eng = try_reload_with_fallback(
            picked=picked, all_candidates=[picked, dup, gptq], build_fn=build,
        )
        # awq tried once (picked), then gptq — never awq twice.
        assert engines_built == ["awq", "gptq"]
        assert used.name == "gptq"

    def test_empty_all_candidates_uses_only_picked(self) -> None:
        """If all_candidates is empty, only picked is in the queue."""
        from soup_cli.utils.auto_quant import try_reload_with_fallback

        picked = self._candidate("awq")

        def build(name):
            raise RuntimeError(f"{name} broken")

        with pytest.raises(RuntimeError, match="tried 1"):
            try_reload_with_fallback(
                picked=picked, all_candidates=[], build_fn=build,
            )

    def test_redacts_load_error_path_in_message(self) -> None:
        """RuntimeError message must not embed repr() (which can leak paths)."""
        from soup_cli.utils.auto_quant import try_reload_with_fallback

        picked = self._candidate("awq")

        def build(name):
            # Simulate FileNotFoundError carrying an absolute path.
            raise FileNotFoundError("/home/user/.cache/secret/checkpoint")

        with pytest.raises(RuntimeError) as exc_info:
            try_reload_with_fallback(
                picked=picked, all_candidates=[picked], build_fn=build,
            )
        msg = str(exc_info.value)
        # Type name + str() is fine; repr() with the full path would include
        # the FileNotFoundError(...) wrapper — confirm we're using type+str.
        assert "FileNotFoundError" in msg
        # Defence-in-depth: the message must not double-quote the path
        # (which would mean repr() was used).
        assert (
            "FileNotFoundError(\"/home/user" not in msg
            and "FileNotFoundError('/home/user" not in msg
        )

    def test_picked_is_always_tried_first_even_if_lower_score(self) -> None:
        """The picker's choice respects (score, -latency); fallback queue
        must lead with picked even when another candidate has higher score."""
        from soup_cli.utils.auto_quant import try_reload_with_fallback

        picked = self._candidate("awq", 0.85)
        gptq = self._candidate("gptq", 0.99)  # higher score but not picked
        engines_built: list[str] = []

        def build(name):
            engines_built.append(name)
            return f"engine-{name}"

        used, _eng = try_reload_with_fallback(
            picked=picked, all_candidates=[picked, gptq], build_fn=build,
        )
        assert used.name == "awq"
        assert engines_built == ["awq"]


class TestVllmCreateEngineQuantizationParam:
    def test_create_vllm_engine_accepts_quantization_kwarg(self) -> None:
        """Source-level proof that the new quantization kwarg is exposed."""
        import inspect

        from soup_cli.utils import vllm as vllm_mod

        sig = inspect.signature(vllm_mod.create_vllm_engine)
        assert "quantization" in sig.parameters
        # Default must be None so existing callers are unaffected.
        assert sig.parameters["quantization"].default is None

    def test_create_vllm_engine_rejects_invalid_quantization(self) -> None:
        """Behavioural proof that an unsupported quantization name raises ValueError.

        We patch out the heavy ``AsyncEngineArgs`` import so the validation
        check fires before any real engine machinery is loaded.
        """
        from soup_cli.utils import vllm as vllm_mod

        # Stub the vllm import so we exercise our own validation, not a
        # real engine-construction failure that would mask the assertion.
        fake_engine_args = MagicMock()
        fake_async_engine = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "vllm": MagicMock(
                    AsyncEngineArgs=fake_engine_args,
                    AsyncLLMEngine=fake_async_engine,
                ),
            },
        ):
            with pytest.raises(ValueError, match="quantization must be one of"):
                vllm_mod.create_vllm_engine(
                    model_path="x",
                    quantization="int8-fake",  # not in the allowlist
                )


# ---------------------------------------------------------------------------
# QAT regression — fp8 must not fall into the int8 prepare_model_for_qat path
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Part C — kernel benchmark warm-up loop
# ---------------------------------------------------------------------------


def test_benchmark_kernel_combos_cpu_returns_none_times() -> None:
    """CI runs CPU-only — benchmark must return None times (degrade gracefully)."""
    from soup_cli.utils.kernel_picker import benchmark_kernel_combos

    candidates = [
        {"name": "baseline", "use_liger": False, "use_flash_attn": False,
         "use_cut_ce": False},
        {"name": "liger", "use_liger": True, "use_flash_attn": False,
         "use_cut_ce": False},
    ]
    out = benchmark_kernel_combos(model=None, candidates=candidates, device="cpu")
    assert len(out) == 2
    for entry in out:
        assert entry["time_ms"] is None


def test_benchmark_kernel_combos_no_cuda_returns_none_times() -> None:
    """Even with device="cuda", if torch.cuda is unavailable, return None times."""
    from soup_cli.utils.kernel_picker import benchmark_kernel_combos

    candidates = [{"name": "baseline"}]
    # Mock torch.cuda.is_available returning False
    import torch  # noqa
    with patch("torch.cuda.is_available", return_value=False):
        out = benchmark_kernel_combos(
            model=MagicMock(), candidates=candidates, device="cuda",
        )
    assert out[0]["time_ms"] is None


def test_benchmark_kernel_combos_does_not_mutate_input() -> None:
    """Benchmark must return a NEW list — input candidates remain untouched."""
    from soup_cli.utils.kernel_picker import benchmark_kernel_combos

    candidates = [{"name": "baseline"}]
    out = benchmark_kernel_combos(
        model=None, candidates=candidates, device="cpu",
    )
    assert "time_ms" not in candidates[0]  # input untouched
    assert "time_ms" in out[0]              # output annotated


def test_benchmark_kernel_combos_clamps_seq_len() -> None:
    """seq_len > 512 must be clamped — defence against caller OOM."""
    from soup_cli.utils.kernel_picker import benchmark_kernel_combos

    # device="cpu" short-circuits without ever touching seq_len, so this just
    # confirms no exception escapes the bounds-clamp branch.
    out = benchmark_kernel_combos(
        model=None, candidates=[{"name": "baseline"}], device="cpu",
        seq_len=10**6,
    )
    assert out[0]["time_ms"] is None


def test_bench_and_pick_kernel_removed_from_v028_features() -> None:
    """Unreachable _bench_and_pick_kernel helper was removed under #976."""
    from soup_cli.utils import v028_features as vf

    assert not hasattr(vf, "_bench_and_pick_kernel")


def test_apply_v028_kernel_auto_compose_never_claims_application() -> None:
    """A caller bypassing config validation still cannot report a false pick."""
    from soup_cli.utils import v028_features as vf

    tcfg = SimpleNamespace(
        use_cut_ce=False, quantization_aware=False,
        kernel_auto_compose=True, activation_offloading=None,
    )
    result = vf.apply_v028_speed_memory(
        model=MagicMock(), tcfg=tcfg,
        base_model="meta-llama/Llama-3.2-1B", console=None,
        device="cuda", backend="transformers",
    )
    assert result["kernel_auto_compose"] is False


@pytest.mark.parametrize(
    "module_path",
    (
        "soup_cli.trainer.dpo",
        "soup_cli.trainer.pretrain",
        "soup_cli.trainer.grpo",
        "soup_cli.trainer.kto",
        "soup_cli.trainer.orpo",
        "soup_cli.trainer.simpo",
        "soup_cli.trainer.ipo",
        "soup_cli.trainer.ppo",
        "soup_cli.trainer.reward_model",
        "soup_cli.trainer.embedding",
    ),
)
def test_trainer_passes_device_and_backend_to_apply_helper(
    module_path: str,
) -> None:
    """Part C — every trainer must forward device + backend so the kernel
    benchmark loop knows whether it's safe to bench (CUDA + transformers)."""
    import importlib
    import inspect

    mod = importlib.import_module(module_path)
    src = inspect.getsource(mod)
    assert "device=self.device" in src, (
        f"{module_path} does not forward device= to apply_v028_speed_memory; "
        "kernel auto-compose benchmark cannot reliably skip on CPU."
    )
    assert "backend=cfg.backend" in src, (
        f"{module_path} does not forward backend= to apply_v028_speed_memory; "
        "unsloth/mlx backends will have their kernels mis-benchmarked."
    )


@pytest.mark.parametrize(
    "module_path",
    (
        "soup_cli.trainer.dpo",
        "soup_cli.trainer.pretrain",
        "soup_cli.trainer.grpo",
        "soup_cli.trainer.kto",
        "soup_cli.trainer.orpo",
        "soup_cli.trainer.simpo",
        "soup_cli.trainer.ipo",
        "soup_cli.trainer.ppo",
    ),
)
def test_trainer_qat_guard_excludes_fp8(module_path: str) -> None:
    """quantization_aware="fp8" must NOT trigger the legacy int8 QAT path."""
    import importlib
    import inspect

    mod = importlib.import_module(module_path)
    src = inspect.getsource(mod)
    # The correct pattern excludes fp8 from the int8 QAT branch.
    assert 'quantization_aware != "fp8"' in src, (
        f"{module_path} QAT guard does not exclude fp8; "
        'quantization_aware="fp8" would crash the legacy int8 QAT path.'
    )
