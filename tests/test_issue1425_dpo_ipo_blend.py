"""#1425 — a preference blend whose primary loss is ``dpo`` or ``ipo`` trains.

``training.preference_loss_weights`` builds the trainer of the loss with the
largest weight and blends the others into its ``compute_loss``. For ``dpo`` and
``ipo`` that trainer is trl's ``DPOTrainer``, which computes the policy and the
reference log-probs inline and keeps only their means, so a blend naming either
loss stopped at the first step ("cannot compute dpo, simpo").

The capture added for it reads the policy sums from the logits of the trainer's
own forward pass and the reference sums from the trainer's
``compute_ref_log_probs``. Every test below that says "trl's own" compares
against ``type(trainer).compute_loss``, the method under Soup's wrappers, or
against log-probs rebuilt here from a separate forward pass, so none of them
reads the capture to check the capture.

Still refused, and pinned in ``tests/test_v05311.py``: a ``dpo`` / ``ipo`` term
under a ``simpo`` / ``orpo`` primary, whose trainer runs no reference.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

WORDS = [
    "<pad>", "<s>", "</s>", "<unk>", "what", "is", "one", "plus", "?",
    "two", "three", "hello", "there", "hi", "no",
]
# Each prompt ends in "?" so that prompt + response tokenizes to the prompt's
# tokens followed by the response's; the responses have four different lengths.
ROWS = [
    {"prompt": "what is one plus one ?", "chosen": "two", "rejected": "one plus one is three"},
    {"prompt": "hello there ?", "chosen": "hi there hello", "rejected": "no no"},
]
DPO_BETA = 0.3
IPO_TAU = 0.2
SIMPO_GAMMA = 1.5
ORPO_BETA = 0.4
FIELDS = (
    f"  dpo_beta: {DPO_BETA}\n  ipo_tau: {IPO_TAU}\n"
    f"  simpo_gamma: {SIMPO_GAMMA}\n  orpo_beta: {ORPO_BETA}\n"
)


def _tiny_model_dir(tmp_path):
    """A random-init 2-layer Llama plus a WordLevel tokenizer (the #1425 repro)."""
    try:
        from soup_cli.trainer._trl_compat import resolve_trl_symbol

        resolve_trl_symbol("DPOConfig")
    except Exception as exc:  # noqa: BLE001 — a broken trl is not a failure
        pytest.skip(f"this environment cannot import a trl DPO config: {exc}")

    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    base = tmp_path / "tiny"
    raw = Tokenizer(
        models.WordLevel(vocab={w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"),
    )
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw, bos_token="<s>", eos_token="</s>",
        pad_token="<pad>", unk_token="<unk>",
    )
    torch.manual_seed(0)
    LlamaForCausalLM(LlamaConfig(
        vocab_size=len(WORDS), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )).save_pretrained(base)
    tok.save_pretrained(base)
    return base


def _build(tmp_path, monkeypatch, weights, fields=FIELDS, name="run"):
    """A real ``PreferenceTrainerWrapper`` for ``weights``, set up on ROWS.

    A fresh LoRA adapter starts with ``B = 0``, which makes the policy equal to
    the reference and every log-ratio zero. ``B`` is given seeded random values
    so the reference terms are not trivially constant.
    """
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.preference import PreferenceTrainerWrapper

    sub = tmp_path / name
    sub.mkdir()
    monkeypatch.chdir(sub)
    base = _tiny_model_dir(sub)
    cfg = load_config_from_string(
        f"base: {base.as_posix()}\ntask: preference\n"
        "data: {train: ./unused.jsonl, format: dpo, max_length: 64}\n"
        "training:\n  epochs: 1\n  batch_size: 2\n  quantization: none\n"
        f"  preference_loss_weights: {weights}\n{fields}"
        "  lora: {r: 4, alpha: 8, dropout: 0.0}\noutput: ./out\n"
    )
    wrapper = PreferenceTrainerWrapper(cfg, device="cpu")
    wrapper.setup({"train": ROWS})
    trainer = wrapper.trainer
    torch.manual_seed(1)
    for param_name, param in trainer.model.named_parameters():
        if "lora_B" in param_name and param.requires_grad:
            torch.nn.init.normal_(param, std=0.5)
    return wrapper, trainer


def _batch(trainer):
    return trainer._prepare_inputs(next(iter(trainer.get_train_dataloader())))


def _trl_loss(trainer, batch):
    """trl's own loss: the class method, under both of Soup's wrappers."""
    return type(trainer).compute_loss(trainer, trainer.model, batch)


def _sums(model, batch):
    """Per-sequence response log-prob sums and lengths, from a separate forward.

    Written the plain way (a full log-softmax, then pick the label's entry) and
    not the way the capture does it.
    """
    logits = model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    ).logits
    logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
    picked = logp.gather(-1, batch["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
    mask = batch["completion_mask"][:, 1:]
    sums = (picked * mask).sum(dim=1)
    chosen, rejected = sums.chunk(2)
    chosen_len, rejected_len = mask.sum(dim=1).chunk(2)
    return chosen, rejected, chosen_len, rejected_len


def _reference_sums(trainer, batch):
    """The same sums with the reference weights: trl's ``ref`` adapter."""
    from trl.trainer.utils import use_adapter

    with torch.no_grad(), use_adapter(trainer.model, adapter_name="ref"):
        chosen, rejected, _, _ = _sums(trainer.model, batch)
    return chosen, rejected


def _terms(trainer, batch):
    """Every blend term, rebuilt from separate forwards with the config's fields."""
    from soup_cli.utils.preference_combine import (
        DEFAULT_BETA,
        compute_orpo_term,
        compute_simpo_term,
    )

    chosen, rejected, chosen_len, rejected_len = _sums(trainer.model, batch)
    ref_chosen, ref_rejected = _reference_sums(trainer, batch)
    chosen_ratio = chosen - ref_chosen
    rejected_ratio = rejected - ref_rejected
    dpo = -torch.nn.functional.logsigmoid(DPO_BETA * (chosen_ratio - rejected_ratio)).mean()
    ipo = (
        (chosen_ratio / chosen_len - rejected_ratio / rejected_len - 1 / (2 * IPO_TAU)) ** 2
    ).mean()
    return {
        "dpo": dpo,
        "ipo": ipo,
        "simpo": compute_simpo_term(
            chosen, rejected, DEFAULT_BETA, SIMPO_GAMMA, chosen_len, rejected_len
        ),
        "orpo": compute_orpo_term(chosen, rejected, ORPO_BETA, chosen_len, rejected_len),
    }


def _weighted(terms, weights):
    return sum(weight * float(terms[name]) for name, weight in weights.items())


class TestTheFixtureIsNotTrivial:
    """What the other tests rely on, checked once."""

    def test_the_batch_is_what_the_tests_assume(self, tmp_path, monkeypatch):
        _, trainer = _build(tmp_path, monkeypatch, "{dpo: 0.7, simpo: 0.3}")
        batch = _batch(trainer)

        assert type(trainer).__name__ == "DPOTrainer"
        assert {"input_ids", "attention_mask", "completion_mask"} <= set(batch)
        assert "ref" in trainer.model.peft_config
        _, _, chosen_len, rejected_len = _sums(trainer.model, batch)
        # four different response lengths, so a missing or swapped length shows
        assert len(set(chosen_len.tolist() + rejected_len.tolist())) == 4
        chosen, rejected, _, _ = _sums(trainer.model, batch)
        ref_chosen, ref_rejected = _reference_sums(trainer, batch)
        assert not torch.allclose(chosen, ref_chosen)
        assert not torch.allclose(rejected, ref_rejected)

    def test_the_rebuilt_terms_are_trls_own_losses(self, tmp_path, monkeypatch):
        """The yardstick itself: the rebuilt ``dpo`` and ``ipo`` terms are the
        numbers trl's trainer returns for ``task: dpo`` and ``task: ipo``."""
        _, dpo_trainer = _build(tmp_path, monkeypatch, "{dpo: 0.7, simpo: 0.3}", name="dpo")
        batch = _batch(dpo_trainer)
        assert dpo_trainer.loss_types == ["sigmoid"]
        assert float(_trl_loss(dpo_trainer, batch)) == pytest.approx(
            float(_terms(dpo_trainer, batch)["dpo"]), rel=1e-5
        )

        _, ipo_trainer = _build(tmp_path, monkeypatch, "{ipo: 0.7, simpo: 0.3}", name="ipo")
        batch = _batch(ipo_trainer)
        assert ipo_trainer.loss_types == ["ipo"]
        assert float(_trl_loss(ipo_trainer, batch)) == pytest.approx(
            float(_terms(ipo_trainer, batch)["ipo"]), rel=1e-5
        )


class TestABlendWithADpoOrIpoPrimary:
    @pytest.mark.parametrize(
        "weights",
        [
            {"dpo": 0.7, "simpo": 0.3},  # the issue's own blend
            {"dpo": 0.6, "orpo": 0.4},
            {"dpo": 0.6, "ipo": 0.4},
            {"ipo": 0.7, "simpo": 0.3},
            {"ipo": 0.6, "dpo": 0.4},
            {"ipo": 0.4, "dpo": 0.3, "simpo": 0.2, "orpo": 0.1},
        ],
    )
    def test_the_loss_is_the_weighted_sum_of_the_terms(self, tmp_path, monkeypatch, weights):
        yaml = "{" + ", ".join(f"{k}: {v}" for k, v in weights.items()) + "}"
        _, trainer = _build(tmp_path, monkeypatch, yaml)
        batch = _batch(trainer)

        blended = float(trainer.compute_loss(trainer.model, batch))

        terms = _terms(trainer, batch)
        assert blended == pytest.approx(_weighted(terms, weights), rel=1e-5)
        # and it is not the primary loss alone, at its weight or at full weight
        primary = max(weights, key=weights.get)
        own = float(_trl_loss(trainer, batch))
        assert blended != pytest.approx(own, rel=1e-3)
        assert blended != pytest.approx(weights[primary] * own, rel=1e-3)

    def test_the_issues_blend_is_trls_dpo_loss_plus_the_simpo_term(self, tmp_path, monkeypatch):
        """The acceptance line of #1425, with trl's own number for the DPO part."""
        _, trainer = _build(tmp_path, monkeypatch, "{dpo: 0.7, simpo: 0.3}")
        batch = _batch(trainer)

        blended = float(trainer.compute_loss(trainer.model, batch))

        simpo = float(_terms(trainer, batch)["simpo"])
        assert blended == pytest.approx(
            0.7 * float(_trl_loss(trainer, batch)) + 0.3 * simpo, rel=1e-5
        )

    def test_the_ipo_term_is_trls_ipo_loss_also_as_the_secondary(self, tmp_path, monkeypatch):
        """Under a ``dpo`` primary the trainer's loss type is ``sigmoid``, so
        nothing in trl computes IPO here; the term must still be the
        length-normalised loss ``task: ipo`` trains, not the summed form."""
        from soup_cli.utils.preference_combine import compute_ipo_term

        _, trainer = _build(tmp_path, monkeypatch, "{dpo: 0.6, ipo: 0.4}")
        batch = _batch(trainer)

        blended = float(trainer.compute_loss(trainer.model, batch))

        terms = _terms(trainer, batch)
        chosen, rejected, _, _ = _sums(trainer.model, batch)
        ref_chosen, ref_rejected = _reference_sums(trainer, batch)
        summed_ipo = float(compute_ipo_term(chosen, rejected, ref_chosen, ref_rejected, IPO_TAU))
        assert summed_ipo != pytest.approx(float(terms["ipo"]), rel=1e-2)
        dpo = float(terms["dpo"])
        assert blended == pytest.approx(0.6 * dpo + 0.4 * float(terms["ipo"]), rel=1e-5)
        assert blended != pytest.approx(0.6 * dpo + 0.4 * summed_ipo, rel=1e-3)

    def test_the_same_weights_under_either_primary_are_the_same_objective(
        self, tmp_path, monkeypatch
    ):
        """An even split takes the first key as the primary, so the YAML order
        picks the trainer's loss type. The blend must not depend on it."""
        _, first = _build(tmp_path, monkeypatch, "{dpo: 0.5, ipo: 0.5}", name="dpo-first")
        _, second = _build(tmp_path, monkeypatch, "{ipo: 0.5, dpo: 0.5}", name="ipo-first")
        assert first.loss_types == ["sigmoid"] and second.loss_types == ["ipo"]

        one = float(first.compute_loss(first.model, _batch(first)))
        two = float(second.compute_loss(second.model, _batch(second)))

        assert one == pytest.approx(two, rel=1e-6)

    @pytest.mark.parametrize(
        ("weights", "field", "low", "high"),
        [
            ("{dpo: 0.7, simpo: 0.3}", "simpo_gamma", 0.5, 3.0),
            ("{dpo: 0.7, orpo: 0.3}", "orpo_beta", 0.1, 0.9),
            ("{dpo: 0.7, ipo: 0.3}", "ipo_tau", 0.1, 0.9),
            ("{ipo: 0.7, dpo: 0.3}", "dpo_beta", 0.1, 0.9),
        ],
    )
    def test_changing_only_the_secondary_loss_changes_the_loss(
        self, tmp_path, monkeypatch, weights, field, low, high
    ):
        """#1425's second acceptance line: the secondary term is in the loss,
        and it reads its own config field, not the primary trainer's ``beta``."""
        losses, own = [], []
        for value in (low, high):
            fields = "".join(
                line + "\n" for line in FIELDS.splitlines() if not line.strip().startswith(field)
            ) + f"  {field}: {value}\n"
            _, trainer = _build(tmp_path, monkeypatch, weights, fields=fields, name=f"v{value}")
            batch = _batch(trainer)
            losses.append(float(trainer.compute_loss(trainer.model, batch)))
            own.append(float(_trl_loss(trainer, batch)))

        assert own[0] == pytest.approx(own[1], rel=1e-6)  # the primary did not move
        # The log-ratios of this tiny model are small, so `dpo_beta` moves the
        # DPO term by about 1%; float32 noise is three orders below the bound.
        assert losses[0] != pytest.approx(losses[1], rel=1e-4)

    def test_the_gradient_reaches_the_adapter_and_not_the_reference(self, tmp_path, monkeypatch):
        _, trainer = _build(tmp_path, monkeypatch, "{dpo: 0.7, simpo: 0.3}")
        batch = _batch(trainer)

        trainer.compute_loss(trainer.model, batch).backward()

        grads = {
            name: param.grad
            for name, param in trainer.model.named_parameters()
            if "lora_" in name
        }
        trained = [name for name in grads if ".default" in name]
        frozen = [name for name in grads if ".ref" in name]
        assert trained and frozen
        assert all(grads[name] is not None for name in trained)
        assert sum(float(grads[name].abs().sum()) for name in trained) > 0
        assert all(grads[name] is None for name in frozen)
        captured = trainer._soup_policy_logps
        assert captured["policy_chosen_logps"].requires_grad
        assert not captured["ref_chosen_logps"].requires_grad
        assert not captured["ref_rejected_logps"].requires_grad

    def test_the_gradient_is_the_weighted_sum_of_the_terms_gradients(self, tmp_path, monkeypatch):
        """What the optimizer gets: 0.7 of the gradient of trl's DPO loss plus
        0.3 of the SimPO term's, and not 0.7 of the DPO gradient alone."""
        _, trainer = _build(tmp_path, monkeypatch, "{dpo: 0.7, simpo: 0.3}")
        batch = _batch(trainer)
        params = [
            param
            for name, param in trainer.model.named_parameters()
            if ".default" in name and "lora_" in name
        ]

        def grad(loss):
            return torch.cat([g.flatten() for g in torch.autograd.grad(loss, params)])

        blend = grad(trainer.compute_loss(trainer.model, batch))
        own = grad(_trl_loss(trainer, batch))
        simpo = grad(_terms(trainer, batch)["simpo"])

        assert float(simpo.abs().max()) > 0
        scale = float(blend.abs().max())
        assert torch.allclose(blend, 0.7 * own + 0.3 * simpo, atol=1e-5 * scale, rtol=1e-4)
        assert not torch.allclose(blend, 0.7 * own, atol=1e-3 * scale, rtol=1e-3)

    def test_the_policy_is_run_once(self, tmp_path, monkeypatch):
        """The policy sums come from the trainer's own forward pass. The
        reference is run once more than in a single-loss run (trl's own, then
        ``compute_ref_log_probs``); a third forward would be a regression."""
        _, trainer = _build(tmp_path, monkeypatch, "{dpo: 0.7, simpo: 0.3}")
        batch = _batch(trainer)
        calls = []
        handle = trainer.model.register_forward_pre_hook(
            lambda *_: calls.append(torch.is_grad_enabled())
        )
        try:
            trainer.compute_loss(trainer.model, batch)
        finally:
            handle.remove()

        assert calls.count(True) == 1
        assert calls.count(False) == 2

    def test_precomputed_reference_log_probs_are_read_from_the_batch(self, tmp_path, monkeypatch):
        """With trl's ``precompute_ref_log_probs`` the batch carries the
        reference sums; the capture must use them and not run the reference."""
        _, trainer = _build(tmp_path, monkeypatch, "{dpo: 0.7, simpo: 0.3}")
        batch = dict(_batch(trainer))
        chosen, rejected, _, _ = _sums(trainer.model, batch)
        # a reference that equals the policy: every log-ratio is zero
        batch["ref_chosen_logps"] = chosen.detach()
        batch["ref_rejected_logps"] = rejected.detach()
        monkeypatch.setattr(trainer, "precompute_ref_logps", True)

        def _not_called(inputs):
            raise AssertionError("compute_ref_log_probs ran although the batch has the sums")

        monkeypatch.setattr(trainer, "compute_ref_log_probs", _not_called)

        blended = float(trainer.compute_loss(trainer.model, batch))

        simpo = float(_terms(trainer, batch)["simpo"])
        assert blended == pytest.approx(0.7 * math.log(2.0) + 0.3 * simpo, rel=1e-5)

    def test_truncated_inputs_are_measured_the_way_trl_measures_them(self, tmp_path, monkeypatch):
        """trl cuts the batch to ``max_length`` inside ``compute_loss``. The
        capture has to cut the same way, or its sums cover tokens the trainer's
        logits do not have."""
        _, trainer = _build(tmp_path, monkeypatch, "{dpo: 0.7, simpo: 0.3}")
        batch = _batch(trainer)
        width = batch["input_ids"].shape[1]
        monkeypatch.setattr(trainer.args, "max_length", width - 2)
        cut = {key: value[:, : width - 2] for key, value in batch.items()}

        blended = float(trainer.compute_loss(trainer.model, batch))

        assert float(_trl_loss(trainer, batch)) == pytest.approx(
            float(_terms(trainer, cut)["dpo"]), rel=1e-5
        )
        assert blended == pytest.approx(
            _weighted(_terms(trainer, cut), {"dpo": 0.7, "simpo": 0.3}), rel=1e-5
        )
        assert blended != pytest.approx(
            _weighted(_terms(trainer, batch), {"dpo": 0.7, "simpo": 0.3}), rel=1e-3
        )

    def test_evaluation_reports_the_blend(self, tmp_path, monkeypatch):
        """trl's DPO ``prediction_step`` goes through ``compute_loss``, so the
        eval loss of a dpo / ipo primary is the blend's value too."""
        _, trainer = _build(tmp_path, monkeypatch, "{dpo: 0.7, simpo: 0.3}")
        raw = next(iter(trainer.get_train_dataloader()))

        loss, _, _ = trainer.prediction_step(trainer.model, raw, prediction_loss_only=True)

        batch = _batch(trainer)
        assert float(loss) == pytest.approx(
            _weighted(_terms(trainer, batch), {"dpo": 0.7, "simpo": 0.3}), rel=1e-5
        )

    def test_return_outputs_is_passed_through(self, tmp_path, monkeypatch):
        _, trainer = _build(tmp_path, monkeypatch, "{dpo: 0.7, simpo: 0.3}")
        batch = _batch(trainer)

        loss, outputs = trainer.compute_loss(trainer.model, batch, return_outputs=True)

        assert outputs.logits.shape[:2] == batch["input_ids"].shape
        assert float(loss) == pytest.approx(float(trainer.compute_loss(trainer.model, batch)))

    @pytest.mark.parametrize("weights", ["{dpo: 0.7, simpo: 0.3}", "{ipo: 0.6, dpo: 0.4}"])
    def test_a_run_completes_and_logs_the_blend(self, tmp_path, monkeypatch, weights):
        """On main this stops at step 0 with "cannot compute"."""
        wrapper, trainer = _build(tmp_path, monkeypatch, weights)
        parsed = {
            key.strip(): float(value)
            for key, value in (item.split(":") for item in weights.strip("{}").split(","))
        }
        expected = _weighted(_terms(trainer, _batch(trainer)), parsed)

        result = wrapper.train()

        # one batch, one step: the logged loss is that step's blend
        assert result["initial_loss"] == pytest.approx(expected, rel=1e-4)
        assert trainer.state.global_step == 1


class TestWhatStillRefuses:
    @pytest.mark.parametrize(
        ("weights", "missing", "computed"),
        [
            ("{simpo: 0.6, dpo: 0.4}", "dpo", "simpo"),
            ("{orpo: 0.6, ipo: 0.4}", "ipo", "orpo"),
            ("{simpo: 0.5, dpo: 0.3, ipo: 0.2}", "dpo, ipo", "simpo"),
        ],
    )
    def test_a_reference_term_under_a_reference_free_primary_says_why(
        self, tmp_path, monkeypatch, weights, missing, computed
    ):
        """The simpo / orpo trainers run no reference, so a dpo / ipo term has
        nothing to compare with. The message used to blame missing log-probs,
        which is not what is missing here, and gave no way forward."""
        wrapper, _ = _build(tmp_path, monkeypatch, weights)

        with pytest.raises(ValueError, match="cannot compute") as excinfo:
            wrapper.train()

        message = str(excinfo.value)
        assert f"cannot compute {missing} on this step (computed: {computed})" in message
        assert "frozen reference" in message
        assert "Give dpo or ipo the largest weight" in message
        assert "no per-sequence log-probs" not in message

    @pytest.mark.parametrize(
        "fields",
        ["  dpo_beta_schedule: linear\n  dpo_beta_end: 0.5\n", "  dpo_ref_regen_epochs: 1\n"],
    )
    def test_the_dpo_variants_are_refused_with_a_blend(self, fields):
        """The blend's ``dpo`` term reads ``training.dpo_beta`` once, from the
        config. A scheduled beta would move the trainer's and not the term's,
        so the schema refusing the pair at load is what keeps that honest."""
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="DPO variants"):
            load_config_from_string(
                "base: ./tiny\ntask: preference\n"
                "data: {train: ./unused.jsonl, format: dpo, max_length: 64}\n"
                "training:\n  preference_loss_weights: {dpo: 0.7, simpo: 0.3}\n"
                f"{fields}  lora: {{r: 4, alpha: 8}}\noutput: ./out\n"
            )

    _LIGER = (
        "base: ./tiny\ntask: {task}\n"
        "data: {{train: ./unused.jsonl, format: dpo, max_length: 64}}\n"
        "training:\n{loss}  use_liger: true\n  lora: {{r: 4, alpha: 8}}\noutput: ./out\n"
    )
    _WEIGHTS = "  preference_loss_weights: {dpo: 0.7, simpo: 0.3}\n"

    def test_liger_keeps_refusing_a_blend_at_load_with_its_reason(self):
        """Out of scope here: the switch never reaches a preference trainer,
        and the schema refuses it for the task and names the tasks that take
        it. For a blend it also says why the blend could not use it."""
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="use_liger=true requires task") as excinfo:
            load_config_from_string(self._LIGER.format(task="preference", loss=self._WEIGHTS))
        assert (
            "got task='preference'. A preference_loss_weights blend could not use it "
            "either: trl's Liger loss returns no logits for the blend to read."
        ) in str(excinfo.value)

    @pytest.mark.parametrize(
        ("task", "loss"),
        [
            ("preference", "  preference_loss: dpo\n"),
            # Weights under another task are no blend; their own check says so
            # once the switch is off.
            ("dpo", _WEIGHTS),
        ],
    )
    def test_liger_without_a_blend_keeps_the_task_message_alone(self, task, loss):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="use_liger=true requires task") as excinfo:
            load_config_from_string(self._LIGER.format(task=task, loss=loss))
        assert str(excinfo.value).endswith(f"got task={task!r}")

    def test_bco_keeps_refusing_a_paired_blend_with_its_reason(self, tmp_path, monkeypatch):
        """Out of scope here, and unchanged: bco reads one response per row
        with a label, not a pair, and the setup says so before anything loads."""
        with pytest.raises(ValueError, match="cannot mix 'bco' with paired losses"):
            wrapper, _ = _build(tmp_path, monkeypatch, "{dpo: 0.7, bco: 0.3}")
            wrapper.train()

    def test_no_log_probs_at_all_keeps_its_own_reason(self):
        from soup_cli.utils.preference_combine import attach_weighted_preference_combine

        class _Bare:
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                return torch.tensor(2.0)

        trainer = _Bare()
        attach_weighted_preference_combine(trainer, {"simpo": 0.7, "orpo": 0.3})

        with pytest.raises(ValueError, match="no per-sequence log-probs") as excinfo:
            trainer.compute_loss(model=None, inputs={})
        assert "frozen reference" not in str(excinfo.value)


class TestTheCapture:
    def _dpo_like(self, without=(), **attrs):
        """The surface of trl's ``DPOTrainer`` the capture looks for, less the
        methods named in ``without``."""

        def _truncate_inputs(self, input_ids, attention_mask, completion_mask):
            raise AssertionError  # pragma: no cover — not reached

        def compute_ref_log_probs(self, inputs):
            raise AssertionError  # pragma: no cover — not reached

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            return torch.tensor(2.0)

        methods = {
            method.__name__: method
            for method in (_truncate_inputs, compute_ref_log_probs, compute_loss)
            if method.__name__ not in without
        }
        trainer = type("_Trainer", (), methods)()
        for key, value in attrs.items():
            setattr(trainer, key, value)
        return trainer

    def _hand_made(self, truncates=True):
        """A stand-in ``DPOTrainer`` with numbers small enough to check by hand:
        two pairs, three positions, a two-word vocabulary with equal logits, so
        every token has log-prob ``-log 2``. ``truncates=False`` is a trl that
        has renamed its private ``_truncate_inputs``."""
        from types import SimpleNamespace

        logits = torch.zeros(4, 3, 2, requires_grad=True)

        class _Trainer:
            truncated = 0

            def _cut(self, input_ids, attention_mask, completion_mask):
                self.truncated += 1
                return input_ids, attention_mask, completion_mask

            if truncates:
                _truncate_inputs = _cut

            def compute_ref_log_probs(self, inputs):
                return torch.tensor([-1.0, -2.0]), torch.tensor([-3.0, -4.0])

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                loss = logits.sum() * 0 + 5.0
                outputs = SimpleNamespace(logits=logits)
                return (loss, outputs) if return_outputs else loss

        inputs = {
            "input_ids": torch.zeros(4, 3, dtype=torch.long),
            "attention_mask": torch.ones(4, 3, dtype=torch.long),
            # rows: chosen, chosen, rejected, rejected. The first position is
            # never predicted, so only the last two columns can count.
            "completion_mask": torch.tensor([[1, 1, 1], [1, 0, 1], [0, 0, 0], [1, 1, 0]]),
        }
        return _Trainer(), inputs

    def test_the_sums_and_lengths_on_a_hand_made_batch(self):
        from soup_cli.utils.preference_combine import attach_reference_logp_capture

        trainer, inputs = self._hand_made()
        assert attach_reference_logp_capture(trainer) is True

        trainer.compute_loss(None, inputs)

        got = trainer._soup_policy_logps
        log2 = math.log(2.0)
        assert got["chosen_lens"].tolist() == [2, 1]
        assert got["rejected_lens"].tolist() == [0, 1]
        assert got["policy_chosen_logps"].tolist() == pytest.approx([-2 * log2, -log2])
        assert got["policy_rejected_logps"].tolist() == pytest.approx([0.0, -log2])
        assert got["ref_chosen_logps"].tolist() == [-1.0, -2.0]
        assert got["ref_rejected_logps"].tolist() == [-3.0, -4.0]
        assert trainer.truncated == 1

    @pytest.mark.parametrize("return_outputs", [False, True])
    def test_the_capture_alone_returns_what_the_caller_asked_for(self, return_outputs):
        """Without the blend on top, the Trainer calls this directly: a tuple
        where it expects the loss would break the training step."""
        from soup_cli.utils.preference_combine import attach_reference_logp_capture

        trainer, inputs = self._hand_made()
        attach_reference_logp_capture(trainer)

        result = trainer.compute_loss(None, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = result
            assert outputs.logits.shape == (4, 3, 2)
        else:
            loss = result
        assert torch.is_tensor(loss) and float(loss) == 5.0

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16]
    )
    def test_the_per_token_log_probs_are_log_softmax_at_the_label(self, dtype):
        """Both forms (subtracting the logsumexp, and the plain log-softmax
        used for the half types) against the textbook expression."""
        from soup_cli.utils.preference_combine import _selected_log_probs

        torch.manual_seed(3)
        logits = (torch.randn(4, 5, 7) * 3).to(dtype)
        labels = torch.randint(0, 7, (4, 5))

        got = _selected_log_probs(logits, labels)

        want = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        assert got.shape == (4, 5) and got.dtype == dtype
        if dtype in (torch.bfloat16, torch.float16):
            assert torch.equal(got, want)
        else:
            assert torch.allclose(got, want, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize(
        ("dtype", "keeps_a_log_softmax"),
        [
            (torch.float32, False),
            (torch.float64, False),
            (torch.bfloat16, True),
            (torch.float16, True),
        ],
    )
    def test_full_precision_keeps_no_log_softmax_in_the_graph(self, dtype, keeps_a_log_softmax):
        """The values are the same either way in float32; the memory is not.
        A ``log_softmax`` node keeps a second ``[seq, vocab]`` tensor per row
        for the backward pass, next to the logits trl's own loss already
        holds. ``logit - logsumexp`` keeps only a ``[seq]`` one."""
        from soup_cli.utils.preference_combine import _selected_log_probs

        logits = torch.randn(2, 3, 5).to(dtype).requires_grad_()
        out = _selected_log_probs(logits, torch.zeros(2, 3, dtype=torch.long))

        seen, stack = set(), [out.grad_fn]
        while stack:
            node = stack.pop()
            if node is None or node in seen:
                continue
            seen.add(node)
            stack.extend(parent for parent, _ in node.next_functions)
        names = {type(node).__name__ for node in seen}
        assert any(name.startswith("LogSoftmax") for name in names) is keeps_a_log_softmax, names
        assert any(name.startswith("Logsumexp") for name in names) is not keeps_a_log_softmax

    def test_float16_does_not_take_the_subtracting_form(self):
        """In float16 ``logit - logsumexp`` rounds differently from the
        log-softmax trl uses for that type, so the two forms are not
        interchangeable there."""
        from soup_cli.utils.preference_combine import _selected_log_probs

        torch.manual_seed(3)
        logits = (torch.randn(4, 16, 50) * 3).to(torch.float16)
        labels = torch.randint(0, 50, (4, 16))
        subtracted = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(
            logits, dim=-1
        )
        want = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        assert not torch.equal(subtracted, want)  # the premise

        assert torch.equal(_selected_log_probs(logits, labels), want)

    def test_a_trainer_without_a_reference_gets_no_capture(self):
        """CPO and ORPO have no ``compute_ref_log_probs``; their ``compute_loss``
        must be left alone (their capture is on ``concatenated_forward``)."""
        from soup_cli.utils.preference_combine import attach_reference_logp_capture

        class _Cpo:
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                return torch.tensor(2.0)

        trainer = _Cpo()
        before = trainer.compute_loss

        assert attach_reference_logp_capture(trainer) is False
        assert trainer.compute_loss == before

    @pytest.mark.parametrize(
        "missing", ["compute_loss", "compute_ref_log_probs", "_truncate_inputs"]
    )
    def test_a_dpo_trainer_without_a_method_the_capture_calls_gets_none(self, missing):
        """``_truncate_inputs`` is private to trl, and any ``trl<1.0`` is
        allowed: a release that renames it must find ``compute_loss`` left
        alone, as a trainer with no reference does."""
        from soup_cli.utils.preference_combine import attach_reference_logp_capture

        trainer = self._dpo_like(without=(missing,))
        before = getattr(trainer, "compute_loss", None)

        assert attach_reference_logp_capture(trainer) is False
        assert getattr(trainer, "compute_loss", None) == before

    def test_a_blend_on_a_trl_without_truncate_inputs_stops_with_the_reason(self):
        """What the user sees on such a trl: the blend's own refusal at the
        first step, not an ``AttributeError`` from inside the capture."""
        from soup_cli.utils.preference_combine import (
            attach_policy_logp_capture,
            attach_reference_logp_capture,
            attach_weighted_preference_combine,
        )

        trainer, inputs = self._hand_made(truncates=False)
        attach_policy_logp_capture(trainer)
        attach_reference_logp_capture(trainer)
        attach_weighted_preference_combine(trainer, {"dpo": 0.7, "simpo": 0.3})

        with pytest.raises(ValueError, match="no per-sequence log-probs") as excinfo:
            trainer.compute_loss(None, inputs)
        assert "cannot compute dpo, simpo on this step" in str(excinfo.value)
        assert trainer.truncated == 0

    def test_the_liger_path_gets_no_capture(self):
        """trl's Liger DPO loss returns no logits and raises on
        ``return_outputs=True``; wrapping it would break a run that works."""
        from soup_cli.utils.preference_combine import attach_reference_logp_capture

        trainer = self._dpo_like(use_liger_kernel=True)
        before = trainer.compute_loss

        assert attach_reference_logp_capture(trainer) is False
        assert trainer.compute_loss == before

    def test_attaching_twice_is_a_no_op(self):
        from soup_cli.utils.preference_combine import attach_reference_logp_capture

        trainer = self._dpo_like()
        assert attach_reference_logp_capture(trainer) is True
        first = trainer.compute_loss
        assert attach_reference_logp_capture(trainer) is True
        assert trainer.compute_loss is first

    def test_only_the_dpo_trainers_get_it_on_a_real_setup(self, tmp_path, monkeypatch):
        """A simpo primary keeps trl's own ``compute_loss`` under the blend."""
        _, trainer = _build(tmp_path, monkeypatch, "{simpo: 0.7, orpo: 0.3}")

        assert type(trainer).__name__ == "CPOTrainer"
        assert not hasattr(trainer, "compute_ref_log_probs")
        trainer.compute_loss(trainer.model, _batch(trainer))
        assert "ref_chosen_logps" not in trainer._soup_policy_logps


class TestTheIpoKernel:
    def _args(self):
        pol_chosen = torch.tensor([-4.0, -9.0])
        pol_rejected = torch.tensor([-6.0, -5.0])
        ref_chosen = torch.tensor([-5.0, -8.0])
        ref_rejected = torch.tensor([-5.5, -7.0])
        return pol_chosen, pol_rejected, ref_chosen, ref_rejected

    def test_with_lengths_each_side_is_divided_by_its_own_length(self):
        from soup_cli.utils.preference_combine import compute_ipo_term

        loss = compute_ipo_term(
            *self._args(), 0.25,
            chosen_lens=torch.tensor([2, 4]), rejected_lens=torch.tensor([5, 1]),
        )

        # chosen ratios (1, -1) / (2, 4); rejected ratios (-0.5, 2) / (5, 1); target 2
        first = (1 / 2 - (-0.5 / 5) - 2.0) ** 2
        second = (-1 / 4 - 2 / 1 - 2.0) ** 2
        assert float(loss) == pytest.approx((first + second) / 2, rel=1e-6)

    def test_without_lengths_the_summed_form_is_unchanged(self):
        from soup_cli.utils.preference_combine import compute_ipo_term

        loss = compute_ipo_term(*self._args(), 0.25)

        first = (1 - (-0.5) - 2.0) ** 2
        second = (-1 - 2 - 2.0) ** 2
        assert float(loss) == pytest.approx((first + second) / 2, rel=1e-6)

    @pytest.mark.parametrize("missing", ["chosen_lens", "rejected_lens"])
    def test_one_length_alone_is_not_used(self, missing):
        from soup_cli.utils.preference_combine import compute_ipo_term

        lens = {"chosen_lens": torch.tensor([2, 4]), "rejected_lens": torch.tensor([5, 1])}
        lens[missing] = None

        assert float(compute_ipo_term(*self._args(), 0.25, **lens)) == pytest.approx(
            float(compute_ipo_term(*self._args(), 0.25)), rel=1e-6
        )

    def test_an_empty_response_does_not_divide_by_zero(self):
        from soup_cli.utils.preference_combine import compute_ipo_term

        loss = compute_ipo_term(
            *self._args(), 0.25,
            chosen_lens=torch.tensor([0, 4]), rejected_lens=torch.tensor([5, 0]),
        )

        first = (1 / 1 - (-0.5 / 5) - 2.0) ** 2
        second = (-1 / 4 - 2 / 1 - 2.0) ** 2
        assert float(loss) == pytest.approx((first + second) / 2, rel=1e-6)

    def test_it_still_needs_a_reference_and_a_positive_beta(self):
        from soup_cli.utils.preference_combine import compute_ipo_term

        pol_chosen, pol_rejected, ref_chosen, ref_rejected = self._args()
        with pytest.raises(ValueError, match="reference"):
            compute_ipo_term(pol_chosen, pol_rejected, None, ref_rejected, 0.25)
        with pytest.raises(ValueError, match="beta"):
            compute_ipo_term(pol_chosen, pol_rejected, ref_chosen, ref_rejected, 0.0)
