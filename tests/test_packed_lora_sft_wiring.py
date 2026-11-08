"""``training.packed_lora`` in the SFT trainer's set-up and teardown (CPU, stub engine).

``PackedLoraSetupMixin`` has four steps (the checks before the load, the context around the kbit
preparation, the hook after the LoRA exists, the teardown); each is tested on a host object with a
fake runtime, and the whole is tested through the real ``SFTTrainerWrapper`` (its real
``_finish_setup`` and ``train``) on a tiny CPU model whose "packed" engine computes the stock
forward, so that the trainer's logged columns, the rewind log and the weights can be compared with
a run that has no hook. The kernels themselves run in ``test_packed_lora_e2e_gpu.py``.
"""

from __future__ import annotations

import json
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("trl")

from soup_cli.config.schema import SoupConfig  # noqa: E402
from soup_cli.kernels.packed_lora import budget, loader  # noqa: E402
from soup_cli.kernels.packed_lora import hook as hook_module  # noqa: E402
from soup_cli.kernels.packed_lora import preflight as checks  # noqa: E402
from soup_cli.kernels.packed_lora.hook import PackedLoraHook, step_from_logits  # noqa: E402
from soup_cli.trainer import packed_lora_setup  # noqa: E402
from soup_cli.trainer.packed_lora_setup import PackedLoraSetupMixin  # noqa: E402
from soup_cli.utils.stream_loss_chunk import HOOK_ATTR  # noqa: E402

DTYPE = torch.bfloat16


def soup(packed=None, **training):
    return SoupConfig(
        base="Qwen/Qwen2.5-1.5B",
        data={"train": "tests/fixtures/sample_train.jsonl", "max_length": 512},
        training={"quantization": "4bit", "batch_size": 4, "lora": {"dropout": 0.0},
                  "packed_lora": {"enabled": True} if packed is None else packed, **training},
    )


def tiny_model(seed=3):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    return Qwen2ForCausalLM(Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        tie_word_embeddings=False))


class FakeRun:
    """A ``PackedRun`` that does not run kernels; records what it was asked."""

    instances = []
    fail_with = None

    def __init__(self, policy, log=None):
        self.policy, self.log = policy, log
        self.device, self.dtype = torch.device("cpu"), DTYPE
        self.closed, self.calls = False, []
        self.steps_packed, self.steps_stock, self.stock_reasons = 0, 0, {}
        FakeRun.instances.append(self)

    def prepare(self, model, dtype, tokens, park=True):
        self.calls.append(("prepare", model, dtype, tokens, park))
        if FakeRun.fail_with is not None:
            raise FakeRun.fail_with

    def close(self):
        self.closed = True
        self.calls.append(("close",))

    def restore_tables(self):
        pass

    def park_tables(self):
        pass

    def note_stock(self, reason):
        self.steps_stock += 1
        self.stock_reasons[reason] = self.stock_reasons.get(reason, 0) + 1


class UnsupportedError(Exception):
    pass


@pytest.fixture
def runtime(monkeypatch):
    FakeRun.instances, FakeRun.fail_with = [], None
    kept = []

    class PackedModel:
        UnsupportedError = UnsupportedError

        @staticmethod
        def keep_tables_in_compute_dtype(model):
            kept.append(model)
            import contextlib

            return contextlib.nullcontext("kept")

    fake = types.SimpleNamespace(PackedRun=FakeRun, packed_model=PackedModel, kept=kept)
    monkeypatch.setattr(loader, "load", lambda: fake)
    monkeypatch.setattr(packed_lora_setup, "compute_dtype_of", lambda causal: DTYPE)
    return fake


class Host(PackedLoraSetupMixin):
    _trust_remote_code = False

    def __init__(self, model=None):
        self.model = model
        self.trainer = None


@pytest.fixture
def says(monkeypatch):
    lines = []
    monkeypatch.setattr(packed_lora_setup.console, "print", lambda text, *a, **k: lines.append(
        str(text)))
    return lines


def preflight_says(monkeypatch, problems=(), notes=(), fit_problems=()):
    monkeypatch.setattr(checks, "read_model_config", lambda base, trust=False: object())
    monkeypatch.setattr(checks, "read_machine", lambda: object())
    seen = []

    def fake(config, model_config, machine, sharding=None):
        seen.append((config, model_config, machine, sharding))
        return types.SimpleNamespace(
            problems=list(problems), notes=list(notes), fit_problems=list(fit_problems))

    monkeypatch.setattr(checks, "preflight", fake)
    return seen


# --- helpers ---------------------------------------------------------------------------------


class TestHelpers:
    def test_enabled(self):
        assert packed_lora_setup.packed_lora_enabled(soup().training) is True
        assert packed_lora_setup.packed_lora_enabled(
            soup({"enabled": False}).training) is False
        assert packed_lora_setup.packed_lora_enabled(
            types.SimpleNamespace(packed_lora=None)) is False
        assert packed_lora_setup.packed_lora_enabled(types.SimpleNamespace()) is False

    def test_the_compute_dtype_is_the_4bit_layers(self):
        weight = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)
        weight.quant_state = types.SimpleNamespace(dtype=torch.float16)
        base = types.SimpleNamespace(weight=weight)
        layer = types.SimpleNamespace(self_attn=types.SimpleNamespace(
            q_proj=types.SimpleNamespace(base_layer=base)))
        model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[layer]))
        assert packed_lora_setup.compute_dtype_of(model) is torch.float16

    def test_a_layer_that_is_not_4bit_has_none(self):
        plain = torch.nn.Linear(2, 2)
        layer = types.SimpleNamespace(self_attn=types.SimpleNamespace(q_proj=plain))
        model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[layer]))
        assert packed_lora_setup.compute_dtype_of(model) is None


# --- 1. before the load ----------------------------------------------------------------------


class TestBegin:
    def test_a_run_without_the_kernels_is_not_looked_at(self, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("the preflight ran for a run that did not ask for the kernels")

        monkeypatch.setattr(checks, "preflight", explode)
        host = Host()
        host._begin_packed_lora(soup({"enabled": False}), soup({"enabled": False}).training)
        assert host._packed_lora_wanted is False

    def test_a_run_that_passes_is_wanted_and_checked_with_the_machine_and_the_config(
            self, monkeypatch, says):
        seen = preflight_says(monkeypatch, notes=["the model's config could not be read"])
        cfg = soup()
        host = Host()
        host._begin_packed_lora(cfg, cfg.training)
        assert host._packed_lora_wanted is True
        assert seen[0][0] is cfg
        assert any("could not be read" in line for line in says)

    @pytest.mark.parametrize("attribute,named", [("fsdp_config", "FSDP"),
                                                 ("deepspeed_config", "DeepSpeed")])
    def test_a_sharded_run_is_told_to_the_checks(self, monkeypatch, attribute, named):
        seen = preflight_says(monkeypatch)
        cfg = soup()
        host = Host()
        setattr(host, attribute, {"some": "setting"})
        host._begin_packed_lora(cfg, cfg.training)
        assert seen[0][3] == named

    def test_a_run_with_no_sharding_tells_the_checks_none(self, monkeypatch):
        seen = preflight_says(monkeypatch)
        cfg = soup()
        Host()._begin_packed_lora(cfg, cfg.training)
        assert seen[0][3] is None

    def test_memory_that_is_not_free_is_a_warning_here_and_the_run_goes_on(
            self, monkeypatch, says):
        """`soup train` refuses it (or `--allow-oom-attempt` goes on) before this point."""
        preflight_says(monkeypatch, fit_problems=[
            "training.packed_lora needs about 9.0 GiB of pinned host memory"])
        cfg = soup()
        host = Host()
        host._begin_packed_lora(cfg, cfg.training)
        assert host._packed_lora_wanted is True
        assert any("pinned host memory" in line and "Packed LoRA kernels" in line
                   for line in says)

    def test_the_gradient_checkpointing_line_is_said_once_when_it_is_set(
            self, monkeypatch, says):
        preflight_says(monkeypatch)
        cfg = soup(gradient_checkpointing=True)
        Host()._begin_packed_lora(cfg, cfg.training)
        lines = [line for line in says if "gradient checkpointing" in line]
        assert len(lines) == 1 and "stock steps only" in lines[0]

    def test_no_checkpointing_line_when_it_is_off(self, monkeypatch, says):
        preflight_says(monkeypatch)
        cfg = soup(gradient_checkpointing=False)
        Host()._begin_packed_lora(cfg, cfg.training)
        assert not [line for line in says if "gradient checkpointing" in line]

    def test_a_refusal_stops_with_every_reason_and_the_way_out(self, monkeypatch):
        preflight_says(monkeypatch, problems=[
            "training.packed_lora runs on Linux (and WSL2) only; this is Windows.",
            "training.packed_lora needs the triton package."])
        cfg = soup()
        host = Host()
        with pytest.raises(ValueError) as excinfo:
            host._begin_packed_lora(cfg, cfg.training)
        message = str(excinfo.value)
        assert "runs on Linux" in message and "needs the triton package" in message
        assert "on_unsupported: stock" in message
        assert host._packed_lora_wanted is False

    def test_under_on_unsupported_stock_the_run_goes_on_and_says_why(self, monkeypatch, says):
        preflight_says(monkeypatch, problems=["training.packed_lora needs a CUDA GPU."])
        cfg = soup({"enabled": True, "on_unsupported": "stock"})
        host = Host()
        host._begin_packed_lora(cfg, cfg.training)
        assert host._packed_lora_wanted is False
        text = " ".join(says)
        assert "needs a CUDA GPU" in text and "stock step" in text

    def test_a_reason_cannot_restyle_the_terminal(self, monkeypatch, says):
        preflight_says(monkeypatch, problems=["training.packed_lora [bold red]x[/] \x1b[31m"])
        cfg = soup({"enabled": True, "on_unsupported": "stock"})
        Host()._begin_packed_lora(cfg, cfg.training)
        assert "\x1b" not in " ".join(says) and "\\[bold red]" in " ".join(says)


class TestSay:
    def test_a_line_from_the_run_cannot_restyle_the_terminal(self, says):
        Host()._packed_lora_say("plan [bold red]x[/] \x1b[31m done")
        assert says == ["[dim]\\[packed_lora] plan \\[bold red]x\\[/] [31m done[/]"]


# --- 2. around the kbit preparation ----------------------------------------------------------


class TestKbitContext:
    def test_a_run_not_using_the_kernels_gets_a_do_nothing_context(self, runtime):
        host = Host(model="m")
        with host._packed_lora_kbit_context() as value:
            assert value is None
        assert runtime.kept == []

    def test_a_wanted_run_keeps_the_tables_in_the_compute_dtype(self, runtime):
        host = Host(model="the model")
        host._packed_lora_wanted = True
        with host._packed_lora_kbit_context() as value:
            assert value == "kept"
        assert runtime.kept == ["the model"]

    def test_without_triton_the_install_step_reports_it_not_this_one(self, monkeypatch):
        def unavailable():
            raise loader.KernelsUnavailableError("no triton")

        monkeypatch.setattr(loader, "load", unavailable)
        host = Host(model="m")
        host._packed_lora_wanted = True
        with host._packed_lora_kbit_context() as value:
            assert value is None


# --- 3. after the LoRA exists ----------------------------------------------------------------


class TestInstall:
    def _host(self, wanted=True):
        host = Host(model=tiny_model())
        host._packed_lora_wanted = wanted
        return host

    def test_a_run_that_is_not_wanted_installs_nothing(self, runtime):
        host = self._host(wanted=False)
        host._install_packed_lora(soup(), soup().training)
        assert FakeRun.instances == [] and host._packed_lora_hook is None
        assert "forward" not in host.model.__dict__

    def test_the_run_is_prepared_for_the_batch_and_the_hook_goes_on_the_causal_lm(
            self, runtime):
        host = self._host()
        cfg = soup(batch_size=4)
        host._install_packed_lora(cfg, cfg.training)
        run = FakeRun.instances[0]
        assert run.calls[0][0] == "prepare"
        _, model, dtype, tokens, park = run.calls[0]
        assert model is host.model and dtype is DTYPE
        assert tokens == 4 * 512  # micro-batch x data.max_length
        assert park is False  # the tables are parked at the first step
        assert run.policy.gpu_budget_bytes is None and run.policy.on_unsupported == "error"
        assert host.model.forward == host._packed_lora_hook._forward
        assert host._packed_lora_run is run

    def test_the_policy_is_the_blocks(self, runtime):
        host = self._host()
        cfg = soup({"enabled": True, "gpu_budget_mib": 7000, "activations": "offload",
                    "int8_backward": "off", "int8_rounding": "nearest"})
        host._install_packed_lora(cfg, cfg.training)
        policy = FakeRun.instances[0].policy
        assert (policy.gpu_budget_bytes, policy.activations) == (7000 << 20, "offload")
        assert (policy.int8_backward, policy.int8_rounding) == ("off", "nearest")

    def test_the_run_logs_through_the_packed_lora_prefix(self, runtime, says):
        host = self._host()
        host._install_packed_lora(soup(), soup().training)
        FakeRun.instances[0].log("plan: weights resident")
        assert says == ["[dim]\\[packed_lora] plan: weights resident[/]"]

    def test_a_model_the_kernels_do_not_take_stops_with_the_reason(self, runtime):
        FakeRun.fail_with = UnsupportedError("a LoRA on lm_head")
        host = self._host()
        with pytest.raises(ValueError) as excinfo:
            host._install_packed_lora(soup(), soup().training)
        assert "a LoRA on lm_head" in str(excinfo.value)
        assert "on_unsupported: stock" in str(excinfo.value)
        assert FakeRun.instances[0].closed  # nothing is left pinned
        assert host._packed_lora_hook is None and "forward" not in host.model.__dict__

    def test_a_budget_below_the_floor_stops_with_the_minimum(self, runtime):
        FakeRun.fail_with = budget.BudgetTooSmallError("the minimum is 5000 MiB", 5000)
        host = self._host()
        with pytest.raises(ValueError, match="the minimum is 5000 MiB"):
            host._install_packed_lora(soup(), soup().training)
        assert FakeRun.instances[0].closed

    def test_under_on_unsupported_stock_the_model_trains_as_it_is(self, runtime, says):
        FakeRun.fail_with = UnsupportedError("sliding-window attention")
        host = self._host()
        cfg = soup({"enabled": True, "on_unsupported": "stock"})
        host._install_packed_lora(cfg, cfg.training)
        assert host._packed_lora_hook is None and host._packed_lora_run is None
        assert "forward" not in host.model.__dict__
        assert FakeRun.instances[0].closed
        assert "sliding-window attention" in " ".join(says)

    def test_a_stock_run_with_a_table_in_the_compute_dtype_still_gets_the_embedding_fixed(
            self, runtime, says):
        """The kbit context left the embedding in the compute dtype; the stock forward needs
        the fp32 embedding it would have had."""
        FakeRun.fail_with = UnsupportedError("odd head layout")
        host = self._host()
        host.model._packed_lora_fp32_embeds = True
        cfg = soup({"enabled": True, "on_unsupported": "stock"})
        host._install_packed_lora(cfg, cfg.training)
        assert isinstance(host._packed_lora_hook, PackedLoraHook)
        assert host._packed_lora_hook._run is None
        assert host.model.forward == host._packed_lora_hook._forward

    def test_an_unexpected_failure_closes_the_run_and_propagates(self, runtime):
        FakeRun.fail_with = RuntimeError("CUDA out of memory")
        host = self._host()
        with pytest.raises(RuntimeError, match="out of memory"):
            host._install_packed_lora(soup(), soup().training)
        assert FakeRun.instances[0].closed

    def test_without_triton_the_reason_is_the_import(self, monkeypatch):
        def unavailable():
            raise loader.KernelsUnavailableError("the triton package is not installed")

        monkeypatch.setattr(loader, "load", unavailable)
        host = self._host()
        with pytest.raises(ValueError, match="the triton package is not installed"):
            host._install_packed_lora(soup(), soup().training)

    def test_the_compute_dtype_falls_back_to_soups_when_the_layer_has_none(
            self, runtime, monkeypatch):
        monkeypatch.setattr(packed_lora_setup, "compute_dtype_of", lambda causal: None)
        import soup_cli.utils.gpu as gpu

        monkeypatch.setattr(gpu, "get_compute_dtype", lambda: torch.float16)
        host = self._host()
        host._install_packed_lora(soup(), soup().training)
        assert FakeRun.instances[0].calls[0][2] is torch.float16

    def test_under_peft_the_hook_sits_on_the_causal_lm(self, runtime):
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(tiny_model(), LoraConfig(r=4, target_modules=["q_proj"]))
        host = Host(model=model)
        host._packed_lora_wanted = True
        host._install_packed_lora(soup(), soup().training)
        causal = model.get_base_model()
        assert causal.forward == host._packed_lora_hook._forward
        assert "forward" not in model.__dict__


# --- 4. the end ------------------------------------------------------------------------------


class TestClose:
    def _installed(self, runtime):
        host = Host(model=tiny_model())
        host._packed_lora_wanted = True
        host._install_packed_lora(soup(), soup().training)
        host.trainer = types.SimpleNamespace()
        setattr(host.trainer, HOOK_ATTR, host._packed_lora_hook)
        return host

    def test_the_hook_comes_off_the_run_is_closed_and_the_trainer_forgets_it(
            self, runtime, says):
        host = self._installed(runtime)
        run = host._packed_lora_run
        host._close_packed_lora()
        assert "forward" not in host.model.__dict__
        assert run.closed
        assert getattr(host.trainer, HOOK_ATTR) is None
        assert host._packed_lora_hook is None and host._packed_lora_run is None

    def test_the_steps_are_summarised(self, runtime, says):
        host = self._installed(runtime)
        host._packed_lora_run.steps_packed = 17
        host._packed_lora_run.note_stock("a batch that is not right-padded")
        host._packed_lora_run.note_stock("a batch that is not right-padded")
        host._close_packed_lora()
        assert "[dim]\\[packed_lora] steps: 17 packed, 2 stock (2x a batch that is not " \
               "right-padded)[/]" in says

    def test_a_run_with_no_stock_step_has_no_parentheses(self, runtime, says):
        host = self._installed(runtime)
        host._packed_lora_run.steps_packed = 3
        host._close_packed_lora()
        assert "[dim]\\[packed_lora] steps: 3 packed, 0 stock[/]" in says

    def test_it_is_safe_twice(self, runtime, says):
        host = self._installed(runtime)
        host._close_packed_lora()
        host._close_packed_lora()
        assert len([line for line in says if "steps:" in line]) == 1

    def test_another_hook_on_the_trainer_is_left_alone(self, runtime):
        host = self._installed(runtime)
        other = object()
        setattr(host.trainer, HOOK_ATTR, other)
        host._close_packed_lora()
        assert getattr(host.trainer, HOOK_ATTR) is other

    def test_a_host_that_never_installed_has_nothing_to_close(self, runtime):
        Host()._close_packed_lora()

    def test_a_stock_only_hook_comes_off_too(self, runtime):
        model = tiny_model()
        host = Host(model=model)
        host._packed_lora_hook = PackedLoraHook(model, None)
        host._close_packed_lora()
        assert "forward" not in model.__dict__


# --- the real trainer -----------------------------------------------------------------------


class EngineOnTheStockForward:
    """A ``PackedRun`` whose "kernels" are the stock forward: the loss and the per-token terms the
    packed step returns, computed from the model's own logits. It makes the hook, the trainer
    class, the rewind recorder and the teardown run for real, and what they log can be compared
    with a run that has no hook."""

    def __init__(self, model):
        self.model = model
        self.device, self.dtype = torch.device("cpu"), DTYPE
        self.closed = False
        self.steps_packed = self.steps_stock = 0
        self.stock_reasons = {}
        self.calls = []

    def forward(self, input_ids, attention_mask, labels, n_items):
        self.calls.append(n_items)
        kwargs = {} if n_items is None else {"num_items_in_batch": n_items}
        outputs = type(self.model).forward(
            self.model, input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            **kwargs)
        step = step_from_logits(outputs.logits.detach(), labels)
        real = attention_mask.bool()
        target = torch.nn.functional.pad(labels, (0, 1), value=-100)[:, 1:]
        hit = ((outputs.logits.detach().argmax(-1) == target) & (target != -100)).to(torch.int32)
        self.steps_packed += 1
        return outputs.loss, {
            "entropy": step.token_entropy[real], "token_loss": step.token_loss,
            "hit": hit[real], "labelled": step.labelled,
        }

    def note_stock(self, reason):
        self.steps_stock += 1
        self.stock_reasons[reason] = self.stock_reasons.get(reason, 0) + 1

    def restore_tables(self):
        pass

    def park_tables(self):
        pass

    def close(self):
        self.closed = True


def _wrapper(tmp_path, monkeypatch, *, packed, right_padded=True, finish_fails=False,
             **training):
    """A real ``SFTTrainerWrapper`` on a tiny Qwen2 + LoRA on CPU. Its ``_setup_transformers`` is
    replaced by a stand-in that installs the hook the way ``_install_packed_lora`` does (a 4-bit
    model cannot be built on CPU); everything after it is the real ``setup`` and ``train``."""
    import yaml
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoTokenizer

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from tests.test_v07204 import _randomise_lora_b, _sft_rows, _write_tiny_tokenizer

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    tokenizer_dir = tmp_path / "tok"
    tokenizer_dir.mkdir(exist_ok=True)
    _write_tiny_tokenizer(str(tokenizer_dir))
    tcfg = {"batch_size": 2, "gradient_accumulation_steps": 1, "quantization": "none",
            "epochs": 1, "logging_steps": 1, "save_steps": 1000,
            "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]}, **training}
    cfg = load_config_from_string(yaml.safe_dump({
        "base": str(tokenizer_dir), "task": "sft", "backend": "transformers",
        "modality": "text",
        "data": {"train": "train.jsonl", "max_length": 64, "chat_template": "chatml"},
        "training": tcfg, "output": str(tmp_path / "out")}))
    wrapper = SFTTrainerWrapper(cfg, device="cpu")
    engine = {}

    def setup_transformers(self, cfg, tcfg):
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        torch.manual_seed(1234)
        model = tiny_model(seed=1234)
        self.model = get_peft_model(model, LoraConfig(
            r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM))
        _randomise_lora_b(self.model)
        if packed:
            run = EngineOnTheStockForward(model)
            if not right_padded:
                run.forward = lambda *args, **kwargs: None
            engine["run"] = run
            self._packed_lora_run = run
            self._packed_lora_hook = PackedLoraHook(model, run)

    monkeypatch.setattr(SFTTrainerWrapper, "_setup_transformers", setup_transformers)
    monkeypatch.setattr(hook_module, "_autocast_dtype", lambda: DTYPE)
    if finish_fails:
        def boom(self, *args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(SFTTrainerWrapper, "_finish_setup", boom)
        with pytest.raises(RuntimeError, match="boom"):
            wrapper.setup({"train": _sft_rows(8)})
        return wrapper, engine
    wrapper.setup({"train": _sft_rows(8)})
    wrapper.trainer.args.max_steps = 2
    return wrapper, engine


def _train(wrapper):
    grads = []
    from transformers import TrainerCallback

    class Grads(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            grads.append([p.grad.detach().clone() for p in model.parameters() if p.requires_grad])

    wrapper.trainer.add_callback(Grads())
    wrapper.train()
    logs = [entry for entry in wrapper.trainer.state.log_history if "loss" in entry]
    rewind = tmp_rewind(wrapper)
    weights = {name: param.detach().clone() for name, param in wrapper.model.named_parameters()
               if param.requires_grad}
    return {"logs": logs, "grads": grads, "rewind": rewind, "weights": weights}


def tmp_rewind(wrapper):
    path = wrapper._output_dir + "/rewind.jsonl"
    import os

    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class TestThroughTheRealTrainer:
    @pytest.fixture
    def arms(self, tmp_path, monkeypatch, capsys):
        plain, _ = _wrapper(tmp_path / "plain", monkeypatch, packed=False)
        plain_result = _train(plain)
        plain_result["class"] = type(plain.trainer).__name__
        packed, engine = _wrapper(tmp_path / "packed", monkeypatch, packed=True)
        packed_result = _train(packed)
        packed_result.update(engine=engine["run"], wrapper=packed,
                             **{"class": type(packed.trainer).__name__})
        return plain_result, packed_result

    def test_the_trainer_class_is_the_one_for_a_model_with_no_logits(self, arms):
        plain, packed = arms
        assert "ChunkedLoss" not in plain["class"]
        assert "ChunkedLoss" in packed["class"]

    def test_the_engine_ran_every_step_and_nothing_fell_back(self, arms):
        _, packed = arms
        assert packed["engine"].steps_packed == 2
        assert packed["engine"].steps_stock == 0

    def test_the_loss_the_columns_and_the_weights_agree_with_a_run_with_no_hook(self, arms):
        plain, packed = arms
        assert len(plain["logs"]) == len(packed["logs"]) == 2
        for want, got in zip(plain["logs"], packed["logs"]):
            assert got.keys() == want.keys()
            assert {"entropy", "num_tokens", "mean_token_accuracy"} <= got.keys()
            assert got["num_tokens"] == want["num_tokens"] > 0
            for column in ("loss", "grad_norm", "entropy", "mean_token_accuracy"):
                assert got[column] == pytest.approx(want[column], rel=1e-5, abs=1e-7), column
        for name, want in plain["weights"].items():
            assert torch.allclose(packed["weights"][name], want, rtol=1e-4, atol=1e-6), name

    def test_the_rewind_log_records_the_same_rows(self, arms):
        plain, packed = arms
        want = [e for e in plain["rewind"] if e["kind"] == "batch"]
        got = [e for e in packed["rewind"] if e["kind"] == "batch"]
        assert len(want) == len(got) == 2
        for a, b in zip(want, got):
            assert b["rows"] == a["rows"] and b["row_tokens"] == a["row_tokens"]
            assert b["row_loss"] == pytest.approx(a["row_loss"], rel=1e-5)

    def test_training_ending_takes_the_hook_off_and_closes_the_run(self, arms, capsys):
        _, packed = arms
        wrapper = packed["wrapper"]
        assert packed["engine"].closed
        assert "forward" not in wrapper.model.get_base_model().__dict__
        assert getattr(wrapper.trainer, HOOK_ATTR) is None
        assert wrapper._packed_lora_hook is None

    def test_the_summary_line_is_printed_once(self, tmp_path, monkeypatch, says):
        wrapper, _ = _wrapper(tmp_path, monkeypatch, packed=True)
        wrapper.train()
        assert [line for line in says if "steps: 2 packed, 0 stock" in line]

    def test_a_batch_the_engine_cannot_take_trains_stock_and_is_counted(
            self, tmp_path, monkeypatch, says):
        stock, _ = _wrapper(tmp_path / "stock", monkeypatch, packed=False)
        want = _train(stock)
        wrapper, engine = _wrapper(tmp_path / "packed", monkeypatch, packed=True,
                                   right_padded=False)
        got = _train(wrapper)
        assert engine["run"].steps_packed == 0 and engine["run"].steps_stock == 2
        assert engine["run"].stock_reasons == {"a batch that is not right-padded": 2}
        assert [line for line in says if "steps: 0 packed, 2 stock" in line]
        # the stock steps are the plain trainer's, and log TRL's columns from their logits
        for a, b in zip(want["logs"], got["logs"]):
            assert b["num_tokens"] == a["num_tokens"] > 0
            for column in ("loss", "entropy", "mean_token_accuracy"):
                assert b[column] == pytest.approx(a[column], rel=1e-5, abs=1e-7), column

    def test_an_evaluation_is_a_stock_forward_with_the_columns_logged(
            self, tmp_path, monkeypatch):
        wrapper, engine = _wrapper(tmp_path, monkeypatch, packed=True)
        wrapper.trainer.train()
        metrics = wrapper.trainer.evaluate(eval_dataset=wrapper.trainer.train_dataset)
        assert metrics["eval_loss"] > 0
        logged = next(e for e in wrapper.trainer.state.log_history if "eval_loss" in e)
        assert {"eval_entropy", "eval_mean_token_accuracy", "eval_num_tokens"} <= logged.keys()
        assert engine["run"].steps_packed == 2  # evaluation took no packed step
        wrapper._close_packed_lora()

    def test_a_failure_after_the_hook_is_installed_still_gives_the_memory_back(
            self, tmp_path, monkeypatch, says):
        wrapper, engine = _wrapper(tmp_path, monkeypatch, packed=True, finish_fails=True)
        assert engine["run"].closed
        assert "forward" not in wrapper.model.get_base_model().__dict__
        assert wrapper._packed_lora_run is None
        assert [line for line in says if "steps: 0 packed, 0 stock" in line]


class TestTheCallSitesInSetupTransformers:
    """``_setup_transformers`` needs a CUDA 4-bit model, so what CPU can pin is the ORDER of its
    steps (the real run is ``test_packed_lora_e2e_gpu.py``): the checks before anything is
    downloaded, the context around the kbit preparation, the hook after the LoRA exists."""

    @pytest.fixture(scope="class")
    def source(self):
        import inspect

        from soup_cli.trainer.sft import SFTTrainerWrapper

        return inspect.getsource(SFTTrainerWrapper._setup_transformers)

    def test_the_checks_come_before_the_tokenizer_and_the_model_are_loaded(self, source):
        begin = source.index("self._begin_packed_lora(cfg, tcfg)")
        assert begin < source.index("AutoTokenizer.from_pretrained")
        assert begin < source.index("AutoModelForCausalLM.from_pretrained")

    def test_the_kbit_preparation_runs_inside_the_context(self, source):
        context = source.index("with self._packed_lora_kbit_context():")
        prepare = source.index("self.model = prepare_model_for_kbit_training(")
        assert 0 < prepare - context < 120  # the next statement

    def test_the_hook_goes_on_after_the_lora_and_every_patch_of_it(self, source):
        install = source.index("self._install_packed_lora(cfg, tcfg)")
        assert install > source.index("self.model = get_peft_model(self.model, lora_config)")
        assert install > source.index("self._apply_quantization_aware(tcfg)")
        assert source.rstrip().endswith("self._install_packed_lora(cfg, tcfg)")

    def test_the_trainer_wrapper_is_the_mixins_host(self):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        assert issubclass(SFTTrainerWrapper, PackedLoraSetupMixin)
