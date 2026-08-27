"""#377 — LISA ``lisa_train_embeddings``: make the memory half of LISA reachable.

At 8B, embeddings + LM head + final norm are 70.7% of everything LISA trains,
and they are trainable **every** interval, so ``lisa_num_layers`` controls only
~30% of the cost. This adds ``training.lisa_train_embeddings`` (default ``true``
= today's behaviour). Setting it ``false`` freezes the always-on group so only
the sampled decoder layers are trainable.

The correctness assertion here is the **trainable-parameter count**, not "the
flag was accepted" (per the issue's acceptance criteria). VRAM / held-out-loss
measurement is a separate, hardware-gated follow-up and is out of scope for
these unit tests.
"""

import pytest


def _fake_lm(num_layers=6):
    """Tiny module shaped like a decoder LM: embed + N layers + norm + head.

    Mirrors ``tests/test_v07134.py::_fake_lm`` so the parameter names hit the
    same ``_ALWAYS_ON`` / ``_LAYER_RE`` matchers the callback uses.
    """
    import torch.nn as nn

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(4, 4)

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(10, 4)
            self.model.layers = nn.ModuleList([Layer() for _ in range(num_layers)])
            self.model.norm = nn.LayerNorm(4)
            self.lm_head = nn.Linear(4, 10)

    return LM()


class _State:
    def __init__(self, global_step):
        self.global_step = global_step


class _FakeOpt:
    def __init__(self):
        self.state = {}


def _always_on_requires_grad(model):
    """Set of requires_grad flags across the embed / head / norm parameters."""
    from soup_cli.utils.lisa import _is_always_on

    return {p.requires_grad for name, p in model.named_parameters() if _is_always_on(name)}


def _trainable_numel(model):
    return sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)


def _sampled_layer_numel(model):
    """numel of the decoder-layer params that are currently trainable."""
    import re

    pat = re.compile(r"(?:layers|h)\.(\d+)\.")
    return sum(
        p.numel()
        for name, p in model.named_parameters()
        if pat.search(name) and p.requires_grad
    )


# ---------------------------------------------------------------------------
# LisaPolicy
# ---------------------------------------------------------------------------
class TestLisaPolicyTrainEmbeddings:
    def test_defaults_to_true(self):
        from soup_cli.utils.lisa import LisaPolicy

        assert LisaPolicy(num_layers=2, interval_steps=20).train_embeddings is True

    def test_accepts_false(self):
        from soup_cli.utils.lisa import LisaPolicy

        p = LisaPolicy(num_layers=2, interval_steps=20, train_embeddings=False)
        assert p.train_embeddings is False

    def test_non_bool_rejected(self):
        from soup_cli.utils.lisa import LisaPolicy

        with pytest.raises(TypeError, match="train_embeddings"):
            LisaPolicy(num_layers=2, interval_steps=20, train_embeddings=1)


# ---------------------------------------------------------------------------
# Callback freeze behaviour (CPU, no hardware)
# ---------------------------------------------------------------------------
class TestAlwaysOnFreezing:
    def test_default_keeps_always_on_trainable(self):
        # Control: today's behaviour is unchanged when the flag is left default.
        from soup_cli.utils.lisa import LisaCallback, LisaPolicy

        model = _fake_lm(6)
        cb = LisaCallback(LisaPolicy(num_layers=2, interval_steps=20, seed=0))
        cb.on_train_begin(None, _State(0), None, model=model)
        assert _always_on_requires_grad(model) == {True}

    def test_false_freezes_always_on(self):
        from soup_cli.utils.lisa import LisaCallback, LisaPolicy

        model = _fake_lm(6)
        cb = LisaCallback(
            LisaPolicy(num_layers=2, interval_steps=20, seed=0, train_embeddings=False)
        )
        cb.on_train_begin(None, _State(0), None, model=model)
        # Every embed / head / norm parameter is frozen ...
        assert _always_on_requires_grad(model) == {False}

    def test_false_still_trains_the_sampled_decoder_layers(self):
        import re

        from soup_cli.utils.lisa import LisaCallback, LisaPolicy

        model = _fake_lm(6)
        cb = LisaCallback(
            LisaPolicy(num_layers=2, interval_steps=20, seed=0, train_embeddings=False)
        )
        cb.on_train_begin(None, _State(0), None, model=model)
        pat = re.compile(r"(?:layers|h)\.(\d+)\.")
        trainable_layers = {
            int(pat.search(name).group(1))
            for name, p in model.named_parameters()
            if pat.search(name) and p.requires_grad
        }
        assert len(trainable_layers) == 2

    def test_trainable_param_count_is_exactly_the_sampled_layers(self):
        # The acceptance criterion: the count is the assertion. With the
        # always-on group frozen, total trainable == the sampled decoder
        # layers' parameters and nothing else.
        from soup_cli.utils.lisa import LisaCallback, LisaPolicy

        model = _fake_lm(6)
        cb = LisaCallback(
            LisaPolicy(num_layers=2, interval_steps=20, seed=0, train_embeddings=False)
        )
        cb.on_train_begin(None, _State(0), None, model=model)
        assert _trainable_numel(model) == _sampled_layer_numel(model)
        assert _trainable_numel(model) > 0  # the sampled layers are non-empty

    def test_freeze_survives_a_resample(self):
        from soup_cli.utils.lisa import LisaCallback, LisaPolicy

        model = _fake_lm(6)
        cb = LisaCallback(
            LisaPolicy(num_layers=2, interval_steps=10, seed=0, train_embeddings=False)
        )
        cb.on_train_begin(None, _State(0), None, model=model)
        cb.on_step_end(None, _State(10), None, model=model, optimizer=_FakeOpt())
        assert _always_on_requires_grad(model) == {False}


# ---------------------------------------------------------------------------
# Schema surface + footgun
# ---------------------------------------------------------------------------
_LISA_BASE = """
base: HuggingFaceTB/SmolLM2-135M
task: sft
backend: transformers
modality: text
data:
  train: data.jsonl
  format: chatml
training:
  quantization: none
  lisa_enabled: true
  lisa_num_layers: 4
  lisa_interval_steps: 25
"""


def _load(yaml_str):
    from soup_cli.config.loader import load_config_from_string

    return load_config_from_string(yaml_str)


class TestSchema:
    def test_field_defaults_true(self):
        cfg = _load(_LISA_BASE)
        assert cfg.training.lisa_train_embeddings is True

    def test_accepts_false_when_lisa_enabled(self):
        cfg = _load(_LISA_BASE.rstrip("\n") + "\n  lisa_train_embeddings: false\n")
        assert cfg.training.lisa_train_embeddings is False

    def test_footgun_false_while_lisa_disabled_rejected(self):
        yaml_str = (
            "base: HuggingFaceTB/SmolLM2-135M\n"
            "task: sft\n"
            "backend: transformers\n"
            "modality: text\n"
            "data:\n  train: data.jsonl\n  format: chatml\n"
            "training:\n"
            "  quantization: none\n"
            "  lisa_enabled: false\n"
            "  lisa_train_embeddings: false\n"
        )
        with pytest.raises(Exception, match="lisa_enabled"):
            _load(yaml_str)


# ---------------------------------------------------------------------------
# VRAM pre-flight estimator: the documented, deliberately-conservative contract
# ---------------------------------------------------------------------------
_FIT_LISA_BASE = """
base: meta-llama/Llama-2-7b-hf
task: sft
backend: transformers
modality: text
data:
  train: train.jsonl
  max_length: 2048
training:
  batch_size: 1
  quantization: none
  lisa_enabled: true
  lisa_num_layers: 2
  lisa_interval_steps: 20
"""


class TestPreflightEstimatorContract:
    """``_build_hardware_fit_input`` classifies LISA as ``peft='full'``
    regardless of ``lisa_train_embeddings`` (#471 routes ``lisa_enabled`` to
    full-FT so the estimator never *under*-predicts). Freezing the always-on
    group lowers real VRAM, but the analytical estimator is not yet credited
    with that saving — crediting it needs a measured constant on GPU hardware
    (the same methodology #327 uses), so the conservative bound stands and a
    frozen-embeddings run that would fit may still be refused by pre-flight
    (bypass with ``--allow-oom-attempt``). This test pins that contract so the
    limitation is a conscious decision, not a silent regression (#377).
    """

    def test_lisa_is_classified_full_regardless_of_train_embeddings(self):
        from soup_cli.commands.train import _build_hardware_fit_input

        on = _build_hardware_fit_input(_load(_FIT_LISA_BASE))
        off = _build_hardware_fit_input(
            _load(_FIT_LISA_BASE.rstrip("\n") + "\n  lisa_train_embeddings: false\n")
        )
        assert on is not None and off is not None
        # Same conservative classification both ways — never under-predicts.
        assert on.peft == "full"
        assert off.peft == "full"

    def test_frozen_embeddings_is_not_under_predicted(self):
        # Belt-and-braces on the invariant that matters: the frozen-embeddings
        # estimate is >= the trainable-everything estimate (it is in fact equal
        # today), so pre-flight can only ever be conservative, never optimistic.
        from soup_cli.commands.train import _build_hardware_fit_input
        from soup_cli.utils.hardware_fit import estimate_peak_vram_gb

        on = estimate_peak_vram_gb(_build_hardware_fit_input(_load(_FIT_LISA_BASE)))
        off = estimate_peak_vram_gb(
            _build_hardware_fit_input(
                _load(_FIT_LISA_BASE.rstrip("\n") + "\n  lisa_train_embeddings: false\n")
            )
        )
        assert off.total_gb >= on.total_gb


# ---------------------------------------------------------------------------
# Wiring: attach_lisa_callback threads the flag into the policy
# ---------------------------------------------------------------------------
class TestWiring:
    def test_attach_threads_train_embeddings_into_policy(self):
        from soup_cli.utils.peft_wiring import attach_lisa_callback

        captured = []

        class _Trainer:
            def add_callback(self, cb):
                captured.append(cb)

        class _TCfg:
            lisa_enabled = True
            lisa_num_layers = 2
            lisa_interval_steps = 20
            lisa_reset_optimizer = True
            lisa_train_embeddings = False

        assert attach_lisa_callback(_Trainer(), _TCfg()) is True
        assert captured, "expected a LisaCallback to be attached"
        assert captured[0].policy.train_embeddings is False
