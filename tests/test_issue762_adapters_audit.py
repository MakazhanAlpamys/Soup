"""Issue #762 — `soup adapters audit`: did the run do what the config asked?

Five merged fixes (#683, #684, #685, #686, #749) taught the MLX path to record
what it actually did into `adapter_config.json`, because each existed as a
defect where a setting was accepted and dropped without a word. The record is
the remedy. Nothing read it back, so a user asking "did my run do what I
asked?" had to open two files and compare by eye — which is how #683 and #686
survived as long as they did.

The load-bearing design decision, and the one these tests exist to defend: a
setting the record cannot speak to is reported **unknown**, never as agreeing.
A false clean bill is worse than no audit at all, because it converts "I do
not know" into "I checked" — the exact substitution this command exists to
undo.
"""

from __future__ import annotations

import json
import pathlib

import pytest

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# A realistic MLX record, in the shape mlx_sft.py writes after #683/#684/#686/#749.
# --------------------------------------------------------------------------
def _mlx_record(**overrides):
    record = {
        "fine_tune_type": "lora",
        "model": "mlx-community/Llama-3.1-8B-Instruct-4bit",
        "batch_size": 4,
        "iters": 48,
        "learning_rate": 1e-4,
        "max_seq_length": 512,
        "optimizer": "AdamW",
        "scheduler": "cosine",
        "warmup_updates": 3,
        "total_updates": 12,
        "weight_decay": 0.01,
        "peak_lr": 1e-4,
        "max_grad_norm": 1.0,
        "grad_checkpoint": False,
        "grad_accumulation_steps": 4,
        "mask_prompt": False,
        "response_token_mask": True,
        "train_on_responses_only": True,
        # The real shape, taken from a live Metal run: MLX writes `scale`
        # (= alpha / rank) and no `alpha` key at all. An earlier fixture
        # invented `alpha` here, so lora.alpha passed in tests and read
        # `unknown` on every real adapter.
        "lora_parameters": {"rank": 8, "scale": 2.0, "dropout": 0.0},
    }
    record.update(overrides)
    return record


def _config(**overrides):
    training = {
        "epochs": 1,
        "lr": 1e-4,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "optimizer": "adamw_torch",
        "scheduler": "cosine",
        "warmup_ratio": 0.25,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
    }
    # #763 review: the command now loads through the schema, and
    # `training.lora` is `Field(default_factory=LoraConfig)` -- so it is always
    # materialised, with r=64, even for a config that never mentions LoRA.
    # This fixture used to rely on the key being ABSENT; stating it explicitly
    # is both what a real config does and what makes "a clean run" mean
    # anything. The record says rank 8 / scale 2.0 -> alpha 16.
    training.setdefault("lora", {"r": 8, "alpha": 16})
    data = {"train": "./d.jsonl", "format": "chatml", "train_on_responses_only": True}
    training.update(overrides.pop("training", {}))
    data.update(overrides.pop("data", {}))
    return {
        "base": "mlx-community/Llama-3.1-8B-Instruct-4bit",
        "task": "sft",
        "backend": "mlx",
        "training": training,
        "data": data,
        **overrides,
    }


def _runner():
    """A ``CliRunner`` with the terminal width pinned in the environment.

    Rich wraps console output to the reported width, so any assertion spanning
    a space is width-dependent: ``"transformers path" in res.output`` holds at
    COLUMNS=35 and fails at COLUMNS=20, where the reason wraps between the two
    words. Worse, the negative control in the same class ("a full MLX record
    prints no such excuse") would then pass vacuously -- testing the terminal
    rather than the code.

    Not sufficient on its own here; see :func:`_pin_console`.
    """
    from typer.testing import CliRunner

    return CliRunner(env={"COLUMNS": "200"})


def _pin_console(monkeypatch):
    """Replace the command module's Console with one of a fixed width.

    ``commands/adapters.py`` builds ``console`` at **import** time, and that
    object keeps the width it was constructed with -- measured: with
    ``COLUMNS=20`` in the environment at import it still reports width 20 from
    inside ``CliRunner(env={"COLUMNS": "200"})``. So the env pin alone never
    reaches it, and the wrapped-assertion trap stays open.

    Same remedy the #756 fix used for ``doctor``
    (``test_issue755_backend_support_registry.py``), which is where this class
    of failure was found the first time.
    """
    from rich.console import Console

    import soup_cli.commands.adapters as adapters_module

    monkeypatch.setattr(adapters_module, "console", Console(width=200))


class TestAgreementAndDivergence:
    def test_a_matching_run_reports_no_divergence(self):
        from soup_cli.utils.adapter_audit import audit_adapter

        result = audit_adapter(_config(), _mlx_record())
        diverged = [r for r in result.rows if r.status == "diverged"]
        assert diverged == [], f"unexpected divergences: {[r.setting for r in diverged]}"
        assert result.exit_code == 0

    def test_a_changed_optimizer_is_reported_with_both_values(self):
        """The #686 shape: the plan said one thing, the optimizer was another."""
        from soup_cli.utils.adapter_audit import audit_adapter

        result = audit_adapter(_config(), _mlx_record(optimizer="SGD"))
        row = next(r for r in result.rows if r.setting == "optimizer")
        assert row.status == "diverged"
        assert "adamw_torch" in str(row.asked) and "SGD" in str(row.ran)
        assert result.exit_code != 0

    def test_a_scheduler_that_collapsed_is_reported(self):
        from soup_cli.utils.adapter_audit import audit_adapter

        result = audit_adapter(_config(), _mlx_record(scheduler="constant"))
        row = next(r for r in result.rows if r.setting == "scheduler")
        assert row.status == "diverged"

    def test_masking_that_did_not_happen_is_reported(self):
        """#683: `train_on_responses_only: true` reaching nothing.

        Kept, but it is the weak version: this fixture flips the request key
        too, which no writer ever does. The real record shape — request `true`
        beside both effect keys `false` — is covered by
        `TestMaskingIsAuditedByEffectNotByRequest`, which is what the #763
        review caught this one failing to assert.
        """
        from soup_cli.utils.adapter_audit import audit_adapter

        result = audit_adapter(
            _config(), _mlx_record(response_token_mask=False, train_on_responses_only=False)
        )
        assert any(
            r.status == "diverged" and "responses_only" in r.setting for r in result.rows
        )


class TestWarmupRoundingIsTheMotivatingCase:
    """`warmup_ratio: 0.03` over a short run rounds to zero warmup updates.

    #686 prints a warning at the time; if nobody was watching the terminal
    there is no way to learn it afterwards. The record says `warmup_updates: 0`,
    so an audit can find it after the fact — which is the whole argument for
    this command existing.
    """

    def test_a_warmup_that_rounded_away_is_reported(self):
        from soup_cli.utils.adapter_audit import audit_adapter

        result = audit_adapter(
            _config(training={"warmup_ratio": 0.03}),
            _mlx_record(warmup_updates=0, total_updates=12),
        )
        row = next(r for r in result.rows if r.setting == "warmup_ratio")
        assert row.status == "diverged"
        assert "0" in str(row.ran)
        assert row.detail, "a rounding divergence must explain itself"

    def test_a_warmup_that_survived_rounding_agrees(self):
        """Control: 0.25 of 12 updates is 3, which is what the record says."""
        from soup_cli.utils.adapter_audit import audit_adapter

        result = audit_adapter(
            _config(training={"warmup_ratio": 0.25}),
            _mlx_record(warmup_updates=3, total_updates=12),
        )
        row = next(r for r in result.rows if r.setting == "warmup_ratio")
        assert row.status == "ok", row.detail


class TestUnknownIsNeverAgreement:
    """The load-bearing rule. A false clean bill converts "I do not know" into
    "I checked", which is the substitution this command exists to undo."""

    def test_a_setting_absent_from_the_record_is_unknown_not_ok(self):
        from soup_cli.utils.adapter_audit import audit_adapter

        record = _mlx_record()
        del record["max_grad_norm"]
        result = audit_adapter(_config(), record)

        row = next(r for r in result.rows if r.setting == "max_grad_norm")
        assert row.status == "unknown", (
            "a setting the record cannot speak to must never read as agreeing"
        )

    def test_unknown_rows_do_not_fail_the_command(self):
        """Unknown is not a divergence — an older adapter predates the keys and
        that is not the user's fault. It must be visible, not fatal."""
        from soup_cli.utils.adapter_audit import audit_adapter

        record = _mlx_record()
        for key in ("max_grad_norm", "optimizer", "scheduler"):
            del record[key]
        result = audit_adapter(_config(), record)

        assert all(r.status != "diverged" for r in result.rows)
        assert result.exit_code == 0
        assert result.unknown_count == 3

    def test_a_peft_record_reports_what_it_cannot_check(self):
        """The transformers path writes PEFT's own adapter_config.json, which
        carries LoRA shape and nothing about the schedule. The command must say
        so rather than implying a clean bill it cannot give."""
        from soup_cli.utils.adapter_audit import audit_adapter

        peft_record = {
            "peft_type": "LORA",
            "base_model_name_or_path": "meta-llama/Llama-3.1-8B",
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
        }
        result = audit_adapter(_config(backend="transformers"), peft_record)

        assert result.unknown_count > 0
        assert any("optimizer" == r.setting and r.status == "unknown" for r in result.rows)
        assert result.record_kind == "peft"

    def test_the_mlx_record_is_recognised_as_such(self):
        from soup_cli.utils.adapter_audit import audit_adapter

        assert audit_adapter(_config(), _mlx_record()).record_kind == "mlx"


class TestLoraShapeIsCheckedOnBothRecordKinds:
    def test_a_changed_rank_is_reported_on_a_peft_record(self):
        from soup_cli.utils.adapter_audit import audit_adapter

        cfg = _config(backend="transformers")
        cfg["training"]["lora"] = {"r": 16, "alpha": 32}
        result = audit_adapter(cfg, {"peft_type": "LORA", "r": 8, "lora_alpha": 32})

        row = next(r for r in result.rows if r.setting == "lora.r")
        assert row.status == "diverged"
        assert "16" in str(row.asked) and "8" in str(row.ran)

    def test_a_matching_rank_agrees_on_an_mlx_record(self):
        """Control, and the shapes differ: MLX nests under `lora_parameters`
        with the key `rank`, PEFT uses a flat `r`."""
        from soup_cli.utils.adapter_audit import audit_adapter

        cfg = _config()
        cfg["training"]["lora"] = {"r": 8, "alpha": 16}
        row = next(
            r for r in audit_adapter(cfg, _mlx_record()).rows if r.setting == "lora.r"
        )
        assert row.status == "ok"


class TestTheCommand:
    """#763 review: `audit` now enforces `enforce_under_cwd_and_no_symlink` on
    both paths, matching `adapters scan` (:779) and `merge` (:1484), so these
    invoke the CLI from inside tmp_path rather than pointing at it."""

    @pytest.fixture(autouse=True)
    def _run_from_tmp(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _pin_console(monkeypatch)

    def _write(self, tmp_path, record, config):
        import yaml

        (tmp_path / "adapter_config.json").write_text(json.dumps(record))
        cfg = tmp_path / "soup.yaml"
        cfg.write_text(yaml.safe_dump(config))
        return cfg

    def test_a_clean_run_exits_zero(self, tmp_path):

        from soup_cli.commands.adapters import app

        cfg = self._write(tmp_path, _mlx_record(), _config())
        res = _runner().invoke(app, ["audit", ".", "--config", cfg.name])
        assert res.exit_code == 0, res.output

    def test_a_divergent_run_exits_non_zero(self, tmp_path):
        """So it composes into CI and `soup ship` rather than only being read."""

        from soup_cli.commands.adapters import app

        cfg = self._write(tmp_path, _mlx_record(optimizer="SGD"), _config())
        res = _runner().invoke(app, ["audit", ".", "--config", cfg.name])
        assert res.exit_code != 0
        assert "optimizer" in res.output

    def test_json_output_is_machine_readable(self, tmp_path):

        from soup_cli.commands.adapters import app

        cfg = self._write(tmp_path, _mlx_record(optimizer="SGD"), _config())
        res = _runner().invoke(app, ["audit", ".", "--config", cfg.name, "--json"])
        payload = json.loads(res.output)
        assert payload["diverged_count"] == 1
        assert any(r["setting"] == "optimizer" for r in payload["rows"])

    def test_a_missing_adapter_config_fails_clearly(self, tmp_path):
        import yaml

        from soup_cli.commands.adapters import app

        cfg = tmp_path / "soup.yaml"
        cfg.write_text(yaml.safe_dump(_config()))
        res = _runner().invoke(app, ["audit", ".", "--config", cfg.name])
        assert res.exit_code != 0
        assert "adapter_config.json" in res.output


class TestLoraAlphaComesFromWhicheverShapeTheWriterChose:
    """Found by a live Metal run, not by a fixture.

    MLX writes `lora_parameters: {"rank": 4, "scale": 2.0, ...}` and **no
    `alpha` key**; `scale` is `alpha / rank`. An earlier version of this file
    invented an `alpha` key in its fixture, so the test passed while
    `lora.alpha` read `unknown` on every real MLX adapter. The fixture was
    asserting my assumption rather than the format.
    """

    def test_mlx_scale_is_converted_back_to_alpha(self):
        from soup_cli.utils.adapter_audit import audit_adapter

        cfg = _config()
        cfg["training"]["lora"] = {"r": 4, "alpha": 8}
        record = _mlx_record(lora_parameters={"rank": 4, "scale": 2.0, "dropout": 0.05})

        row = next(r for r in audit_adapter(cfg, record).rows if r.setting == "lora.alpha")
        assert row.status == "ok", (
            f"scale 2.0 x rank 4 is alpha 8; got {row.ran!r} ({row.status})"
        )

    def test_a_wrong_alpha_still_diverges_through_the_conversion(self):
        """Control: the conversion must not launder a real mismatch into `ok`."""
        from soup_cli.utils.adapter_audit import audit_adapter

        cfg = _config()
        cfg["training"]["lora"] = {"r": 4, "alpha": 32}
        record = _mlx_record(lora_parameters={"rank": 4, "scale": 2.0})

        row = next(r for r in audit_adapter(cfg, record).rows if r.setting == "lora.alpha")
        assert row.status == "diverged"
        assert "8" in str(row.ran)

    def test_peft_lora_alpha_is_read_directly(self):
        """The other writer stores alpha itself, with no conversion."""
        from soup_cli.utils.adapter_audit import audit_adapter

        cfg = _config(backend="transformers")
        cfg["training"]["lora"] = {"r": 8, "alpha": 16}
        row = next(
            r for r in audit_adapter(cfg, {"peft_type": "LORA", "r": 8, "lora_alpha": 16}).rows
            if r.setting == "lora.alpha"
        )
        assert row.status == "ok"

    def test_a_record_with_neither_shape_is_unknown_not_ok(self):
        from soup_cli.utils.adapter_audit import audit_adapter

        cfg = _config()
        cfg["training"]["lora"] = {"r": 4, "alpha": 8}
        record = _mlx_record(lora_parameters={"rank": 4})

        row = next(r for r in audit_adapter(cfg, record).rows if r.setting == "lora.alpha")
        assert row.status == "unknown"


# --------------------------------------------------------------------------
# #763 review — the audit must not carry its own idea of a default
# --------------------------------------------------------------------------

class TestAuditDefaultsMatchTheSchema:
    """Three audit fallbacks disagreed with ``config/schema.py``.

    A config that simply omitted `warmup_ratio` was reported ``ok`` against
    0.0 when the schema asks for 0.03 — a false clean bill on the motivating
    example of #762 — and `weight_decay` / `gradient_accumulation_steps`
    reported DIVERGED on a conformant run, which is worse than noise once this
    is wired into CI.

    ``adapter_audit`` is deliberately stdlib-only, so it cannot import the
    schema to find out. This pins the correspondence from the outside instead,
    mechanically, so the next default that moves fails here rather than in a
    user's audit.
    """

    def _audit_fallbacks(self):
        """Every ``mapping.get("name", default)`` literal in the audit module."""
        import ast
        import pathlib

        source = pathlib.Path(
            __file__
        ).resolve().parents[1] / "src/soup_cli/utils/adapter_audit.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        found = {}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant)
            ):
                try:
                    found[node.args[0].value] = ast.literal_eval(node.args[1])
                except ValueError:
                    continue
        return found

    def test_every_fallback_matches_the_schema_default(self):
        from soup_cli.config.schema import DataConfig, TrainingConfig

        mismatches = []
        checked = 0
        for name, fallback in self._audit_fallbacks().items():
            field = TrainingConfig.model_fields.get(name) or DataConfig.model_fields.get(
                name
            )
            if field is None:
                continue  # a record key, not a config field
            checked += 1
            if fallback != field.default:
                mismatches.append(
                    f"adapter_audit falls back to {name}={fallback!r}, "
                    f"schema default is {field.default!r}"
                )
        assert checked >= 5, f"only {checked} fallbacks resolved; the walk is broken"
        assert mismatches == [], "\n".join(mismatches)

    def test_the_check_can_actually_fail(self):
        """It finds zero mismatches today, so prove it still fires."""
        from soup_cli.config.schema import TrainingConfig

        field = TrainingConfig.model_fields["warmup_ratio"]
        assert field.default != 0.0, (
            "this check is vacuous if the schema default is the old audit literal"
        )

    def test_an_omitted_setting_is_audited_against_the_schema_value(self, tmp_path):
        """End to end: the shape that produced the false clean bill."""
        from soup_cli.config.loader import load_config_from_string

        train = tmp_path / "t.jsonl"
        train.write_text('{"instruction": "a", "output": "b"}\n', encoding="utf-8")
        config = load_config_from_string(
            f"base: m\ntask: sft\ndata: {{train: {train}, format: alpaca}}\n"
            f"training: {{epochs: 1}}\noutput: {tmp_path / 'o'}\n"
        )
        asked = config.model_dump()["training"]
        assert asked["warmup_ratio"] == 0.03
        assert asked["weight_decay"] == 0.01
        assert asked["gradient_accumulation_steps"] == 4


class TestSchedulerIsComparedCaseInsensitively:
    """#763 review: `mlx_optim` stores `str(scheduler).strip().lower()`.

    `scheduler: Cosine` in a config would report DIVERGED against a record
    saying `cosine`. `_audit_optimizer` already normalised; `_cmp` did not.
    """

    @pytest.mark.parametrize("written", ["cosine", "Cosine", "COSINE", "  cosine  "])
    def test_case_and_padding_do_not_diverge(self, written):
        from soup_cli.utils.adapter_audit import audit_adapter

        cfg = _config(training={"scheduler": written})
        row = next(
            r for r in audit_adapter(cfg, _mlx_record()).rows if r.setting == "scheduler"
        )
        assert row.status == "ok", f"{written!r} reported {row.status}"

    def test_a_genuinely_different_scheduler_still_diverges(self):
        """Control — normalising must not swallow a real mismatch."""
        from soup_cli.utils.adapter_audit import audit_adapter

        cfg = _config(training={"scheduler": "linear"})
        row = next(
            r for r in audit_adapter(cfg, _mlx_record()).rows if r.setting == "scheduler"
        )
        assert row.status == "diverged"

    def test_the_displayed_value_is_what_the_user_wrote(self):
        """Normalisation is for the comparison, not for the report."""
        from soup_cli.utils.adapter_audit import audit_adapter

        cfg = _config(training={"scheduler": "Cosine"})
        row = next(
            r for r in audit_adapter(cfg, _mlx_record()).rows if r.setting == "scheduler"
        )
        assert row.asked == "Cosine"


class TestTheAliasTableMatchesTheRealOne:
    """#763 review nit: `_MLX_OPTIMIZER_ALIASES` is a hand-copy of
    `mlx_optim._OPTIMIZER_MAP`, identical today with nothing pinning it.
    `mlx_optim` sets the precedent with its own weight-decay table test."""

    def test_every_alias_matches_mlx_optim(self):
        from soup_cli.trainer.mlx_optim import _OPTIMIZER_MAP
        from soup_cli.utils.adapter_audit import _MLX_OPTIMIZER_ALIASES

        assert _MLX_OPTIMIZER_ALIASES == _OPTIMIZER_MAP, (
            "the audit's copy has drifted from mlx_optim's table"
        )


class TestMaskingIsAuditedByEffectNotByRequest:
    """#763 review, blocking: the record's `train_on_responses_only` is the
    **request** written back, not the effect.

    `mlx_sft.py:742-744` writes three keys. `train_on_responses_only` is
    `responses_only` — what the config asked for, echoed. The effect is
    `mask_prompt` (upstream's single masked prefix) and `response_token_mask`
    (Soup's per-token mask), and `plan_response_masking` decides which, if
    either, a given dataset shape can have. For plain-text rows it returns
    neither and warns once, so the record a real run leaves behind says
    `train_on_responses_only: true` beside two false effect keys.

    Auditing the echo is auditing the question. It reported `ok` / exit 0 on
    #683 itself — the headline case this command exists for.

    Every record below is built by calling the real planner on a real row
    shape, not by hand-editing booleans: the previous test passed only because
    its fixture flipped the request key too, which no writer ever does.
    """

    @staticmethod
    def _record_a_run_would_write(sample, responses_only=True, **overrides):
        """The three masking keys `mlx_sft.py` writes, from the real planner."""
        from soup_cli.trainer.mlx_masking import plan_response_masking

        plan = plan_response_masking(responses_only, sample)
        return _mlx_record(
            mask_prompt=plan.mask_prompt,
            response_token_mask=plan.token_mask,
            train_on_responses_only=responses_only,
            **overrides,
        )

    def test_plain_text_rows_that_masked_nothing_are_not_a_clean_bill(self):
        """#683 exactly: asked for response-only, got a warning and full-sequence
        training. The request key says `true` on both sides."""
        from soup_cli.utils.adapter_audit import audit_adapter

        record = self._record_a_run_would_write({"text": "a plain row"})
        assert record["train_on_responses_only"] is True, "the echo the old row read"
        assert record["mask_prompt"] is False and record["response_token_mask"] is False

        result = audit_adapter(_config(), record)
        row = next(r for r in result.rows if r.setting == "data.train_on_responses_only")
        assert row.status == "diverged", f"reported {row.status}: {row.detail}"
        assert result.exit_code != 0

    def test_soups_per_token_mask_counts_as_masking(self):
        """Chat rows — the mask really happened, through `response_token_mask`."""
        from soup_cli.utils.adapter_audit import audit_adapter

        record = self._record_a_run_would_write(
            {"messages": [{"role": "user", "content": "q"}]}
        )
        assert record["response_token_mask"] is True
        row = next(
            r
            for r in audit_adapter(_config(), record).rows
            if r.setting == "data.train_on_responses_only"
        )
        assert row.status == "ok"

    def test_upstreams_prefix_mask_also_counts_as_masking(self):
        """prompt/completion rows — `mask_prompt` is the whole prompt here, so
        upstream's flag is exactly right and the audit must not call it a miss."""
        from soup_cli.utils.adapter_audit import audit_adapter

        record = self._record_a_run_would_write({"prompt": "q", "completion": "a"})
        assert record["mask_prompt"] is True and record["response_token_mask"] is False
        row = next(
            r
            for r in audit_adapter(_config(), record).rows
            if r.setting == "data.train_on_responses_only"
        )
        assert row.status == "ok"

    def test_masking_switched_off_and_not_done_agrees(self):
        """Control: the audit must not simply always report divergence."""
        from soup_cli.utils.adapter_audit import audit_adapter

        record = self._record_a_run_would_write(
            {"messages": [{"role": "user", "content": "q"}]}, responses_only=False
        )
        cfg = _config(data={"train_on_responses_only": False})
        row = next(
            r
            for r in audit_adapter(cfg, record).rows
            if r.setting == "data.train_on_responses_only"
        )
        assert row.status == "ok"

    def test_masking_switched_off_but_performed_anyway_diverges(self):
        """The other direction: a mask the config did not ask for."""
        from soup_cli.utils.adapter_audit import audit_adapter

        record = self._record_a_run_would_write(
            {"messages": [{"role": "user", "content": "q"}]}, responses_only=True
        )
        cfg = _config(data={"train_on_responses_only": False})
        row = next(
            r
            for r in audit_adapter(cfg, record).rows
            if r.setting == "data.train_on_responses_only"
        )
        assert row.status == "diverged"

    def test_a_record_predating_the_effect_keys_is_unknown_with_a_reason(self):
        """An adapter written before #683 carries neither effect key. That is
        `unknown` — not `ok` off the echo, and not a failure either."""
        from soup_cli.utils.adapter_audit import audit_adapter

        record = _mlx_record()
        del record["mask_prompt"]
        del record["response_token_mask"]

        result = audit_adapter(_config(), record)
        row = next(r for r in result.rows if r.setting == "data.train_on_responses_only")
        assert row.status == "unknown", f"reported {row.status}"
        assert row.detail, "an unknown row has to say why"
        assert result.exit_code == 0

    def test_the_echo_alone_never_produces_agreement(self):
        """The mutation this class exists to kill, stated directly: reading
        `record["train_on_responses_only"]` agrees with the config on every
        record a real run writes, including the ones where nothing happened."""
        from soup_cli.utils.adapter_audit import audit_adapter

        record = self._record_a_run_would_write({"text": "a plain row"})
        echoed = record["train_on_responses_only"]
        asked = _config()["data"]["train_on_responses_only"]
        assert echoed == asked, "the echo agrees, which is the trap"

        row = next(
            r
            for r in audit_adapter(_config(), record).rows
            if r.setting == "data.train_on_responses_only"
        )
        assert row.status != "ok"


class TestEverySettingCanActuallyDiverge:
    """#763 review, finding 2: four rows were compared against nothing.

    Mutating `weight_decay`, `max_grad_norm`, `gradient_accumulation_steps` or
    `gradient_checkpointing` to compare the config against *itself* left the
    whole suite green. Those are exactly the settings #684, #685 and #749 exist
    for — the audit claimed to check them while checking nothing.

    One case per audited setting, each divergent in that setting alone, so the
    self-comparison mutation dies wherever it is applied.
    """

    #: (setting, record overrides that make *only* that setting disagree).
    #: `_config()` asks warmup_ratio 0.25 over 12 updates -> 3 warmup updates,
    #: gradient_checkpointing is unset so the schema default False applies.
    DIVERGENT = [
        ("optimizer", {"optimizer": "SGD"}),
        ("scheduler", {"scheduler": "linear"}),
        ("warmup_ratio", {"warmup_updates": 7}),
        ("weight_decay", {"weight_decay": 0.5}),
        ("max_grad_norm", {"max_grad_norm": 0.25}),
        ("gradient_accumulation_steps", {"grad_accumulation_steps": 16}),
        ("gradient_checkpointing", {"grad_checkpoint": True}),
        (
            "data.train_on_responses_only",
            {"mask_prompt": False, "response_token_mask": False},
        ),
        # MLX nests LoRA shape and stores `scale` (= alpha / rank), so each of
        # these is chosen to move one side only: rank 32 x scale 0.5 is still
        # alpha 16, and rank 8 x scale 8.0 is still rank 8.
        ("lora.r", {"lora_parameters": {"rank": 32, "scale": 0.5, "dropout": 0.0}}),
        ("lora.alpha", {"lora_parameters": {"rank": 8, "scale": 8.0, "dropout": 0.0}}),
    ]

    @pytest.mark.parametrize(
        "setting,overrides", DIVERGENT, ids=[s for s, _ in DIVERGENT]
    )
    def test_the_setting_is_reported_diverged(self, setting, overrides):
        from soup_cli.utils.adapter_audit import audit_adapter

        result = audit_adapter(_config(), _mlx_record(**overrides))
        row = next(r for r in result.rows if r.setting == setting)
        assert row.status == "diverged", (
            f"{setting} disagrees with the record and was reported {row.status}"
        )
        assert result.exit_code != 0

    def test_the_control_agrees_on_every_one_of_them(self):
        """Without this, a mutation making every row DIVERGED passes the ten
        cases above. Same fixture, unmutated: all ten must read `ok`."""
        from soup_cli.utils.adapter_audit import audit_adapter

        result = audit_adapter(_config(), _mlx_record())
        statuses = {r.setting: r.status for r in result.rows}
        for setting, _ in self.DIVERGENT:
            assert statuses.get(setting) == "ok", (
                f"{setting} reported {statuses.get(setting)} on a conformant run"
            )
        assert result.exit_code == 0

    def test_every_audited_setting_is_covered_by_a_case(self):
        """The list above is hand-written; a new audited setting must not be
        able to arrive uncovered."""
        from soup_cli.utils.adapter_audit import audit_adapter

        audited = {r.setting for r in audit_adapter(_config(), _mlx_record()).rows}
        covered = {s for s, _ in self.DIVERGENT}
        assert audited == covered, f"not covered: {sorted(audited - covered)}"


class TestWarmupRowsTheRecordCannotSpeakTo:
    """#763 review, finding 2: `_audit_warmup`'s missing-key branch returning
    OK survived — no test deleted `warmup_updates` or `total_updates`. It needs
    both, so each has to be deleted on its own."""

    @pytest.mark.parametrize("missing", ["warmup_updates", "total_updates"])
    def test_a_missing_key_makes_warmup_unknown_not_ok(self, missing):
        from soup_cli.utils.adapter_audit import audit_adapter

        record = _mlx_record()
        del record[missing]
        result = audit_adapter(_config(), record)

        row = next(r for r in result.rows if r.setting == "warmup_ratio")
        assert row.status == "unknown", f"without {missing} the row read {row.status}"
        assert result.exit_code == 0, "unknown is visible, not fatal"


class TestABooleanIsNotTheNumberItEqualsInPython:
    """#763 review, finding 4: `True == 1` and `False == 0`, so a record
    carrying `max_grad_norm: true` was reported as agreeing with `1.0`.

    Only a malformed record does this — but a clean bill for a malformed
    record is the same failure mode as a clean bill for a missing key, which
    is what the rest of this module refuses.
    """

    @pytest.mark.parametrize(
        "setting,asked,overrides",
        [
            ("max_grad_norm", {"max_grad_norm": 1.0}, {"max_grad_norm": True}),
            ("weight_decay", {"weight_decay": 0.0}, {"weight_decay": False}),
            (
                "gradient_accumulation_steps",
                {"gradient_accumulation_steps": 1},
                {"grad_accumulation_steps": True},
            ),
            ("lora.r", {"lora": {"r": 1, "alpha": 16}}, {"r": True}),
        ],
    )
    def test_a_bool_in_the_record_never_agrees_with_a_number(
        self, setting, asked, overrides
    ):
        from soup_cli.utils.adapter_audit import audit_adapter

        result = audit_adapter(_config(training=asked), _mlx_record(**overrides))
        row = next(r for r in result.rows if r.setting == setting)
        assert row.status == "diverged", (
            f"{setting}: a bool in the record read {row.status} against a number"
        )

    def test_a_number_in_the_record_never_agrees_with_a_bool(self):
        """The other direction: `gradient_checkpointing` is asked as a bool."""
        from soup_cli.utils.adapter_audit import audit_adapter

        result = audit_adapter(
            _config(training={"gradient_checkpointing": False}),
            _mlx_record(grad_checkpoint=0),
        )
        row = next(r for r in result.rows if r.setting == "gradient_checkpointing")
        assert row.status == "diverged"

    def test_matching_bools_still_agree(self):
        """Control — the type check must not make every boolean row diverge."""
        from soup_cli.utils.adapter_audit import audit_adapter

        result = audit_adapter(
            _config(training={"gradient_checkpointing": True}),
            _mlx_record(grad_checkpoint=True),
        )
        row = next(r for r in result.rows if r.setting == "gradient_checkpointing")
        assert row.status == "ok"


class TestPathsAreRefusedBeforeTheyAreFollowed:
    """#763 review, finding 4: containment ran on the *resolved* path and
    after the existence check.

    `Path(adapter).resolve()` follows the symlink first, so `os.lstat` in
    `enforce_under_cwd_and_no_symlink` saw the target and the no-symlink half
    never fired for the adapter directory. And because the existence check came
    first, an out-of-cwd path was reported as "No adapter_config.json in: ..."
    — a missing-file message for a refused path.
    """

    @pytest.fixture(autouse=True)
    def _run_from_tmp(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _pin_console(monkeypatch)

    def _config_file(self, tmp_path):
        import yaml

        cfg = tmp_path / "soup.yaml"
        cfg.write_text(yaml.safe_dump(_config()))
        return cfg

    def test_a_path_outside_cwd_is_refused_not_reported_missing(self, tmp_path):

        from soup_cli.commands.adapters import app

        cfg = self._config_file(tmp_path)
        outside = tmp_path.parent  # exists, and is not under cwd
        res = _runner().invoke(app, ["audit", str(outside), "--config", cfg.name])

        assert res.exit_code != 0
        assert "Path refused" in res.output, res.output
        assert "No adapter_config.json" not in res.output

    def test_a_symlinked_adapter_directory_is_refused(self, tmp_path):

        from soup_cli.commands.adapters import app

        real = tmp_path / "real"
        real.mkdir()
        (real / "adapter_config.json").write_text(json.dumps(_mlx_record()))
        link = tmp_path / "link"
        try:
            link.symlink_to(real, target_is_directory=True)
        except (OSError, NotImplementedError):  # unprivileged Windows
            pytest.skip("symlink creation not permitted on this platform")

        cfg = self._config_file(tmp_path)
        res = _runner().invoke(app, ["audit", "link", "--config", cfg.name])

        assert res.exit_code != 0, res.output
        assert "Path refused" in res.output and "symlink" in res.output, res.output

    def test_the_real_directory_behind_it_still_audits(self, tmp_path):
        """Control — the refusal is about the symlink, not about the contents."""

        from soup_cli.commands.adapters import app

        real = tmp_path / "real"
        real.mkdir()
        (real / "adapter_config.json").write_text(json.dumps(_mlx_record()))
        cfg = self._config_file(tmp_path)
        res = _runner().invoke(app, ["audit", "real", "--config", cfg.name])

        assert res.exit_code == 0, res.output


class TestTheCommandLoadsThroughTheSchema:
    """#763 review, finding 2: reverting the command to `yaml.safe_load`
    survived, because every audit fallback now equals its schema default.

    The surviving difference is `training.lora`, which is
    `Field(default_factory=LoraConfig)` — the schema materialises it at r=64 /
    alpha=16 for a config that never mentions LoRA, and a raw mapping does not.
    That is not an artefact: an MLX run on a config with no `lora:` block
    really does train at r=64, so those rows belong in the audit.
    """

    @pytest.fixture(autouse=True)
    def _run_from_tmp(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _pin_console(monkeypatch)

    def test_a_config_with_no_lora_block_is_still_audited_against_r_64(self, tmp_path):
        import yaml

        from soup_cli.commands.adapters import app

        no_lora = {
            "base": "mlx-community/Llama-3.1-8B-Instruct-4bit",
            "task": "sft",
            "backend": "mlx",
            "training": {"epochs": 1, "warmup_ratio": 0.25},
            "data": {"train": "./d.jsonl", "format": "chatml"},
        }
        assert "lora" not in no_lora["training"], "the whole point of the fixture"

        (tmp_path / "adapter_config.json").write_text(json.dumps(_mlx_record()))
        cfg = tmp_path / "soup.yaml"
        cfg.write_text(yaml.safe_dump(no_lora))

        res = _runner().invoke(app, ["audit", ".", "--config", cfg.name, "--json"])
        payload = json.loads(res.stdout)
        rows = {r["setting"]: r for r in payload["rows"]}

        # Under `yaml.safe_load` there is no `lora` key, so no lora row exists
        # at all and this KeyError is the failure the mutation would produce.
        assert rows["lora.r"]["asked"] == 64, "the schema default did not reach the audit"
        assert rows["lora.r"]["ran"] == 8, "the record says rank 8"
        assert rows["lora.r"]["status"] == "diverged"
        assert res.exit_code != 0


class TestTheJsonFlagEmitsOnlyJson:
    """#763 review, finding 3: a regression from the schema-loading fix.

    `load_config_from_string` reports unknown keys through a module-level
    `rich` Console, which writes to **stdout** — so `--json` emitted four lines
    of warning before the payload and `json.load` failed on it. That breaks
    acceptance criterion 6.

    This runs a real process rather than `CliRunner`: whether stdout and stderr
    are separable in-process depends on the click version (`mix_stderr` was
    removed in 8.2), and the claim under test is precisely that they are
    separate. Marked `integration` for that reason, against the module's
    `unit` default.
    """

    @staticmethod
    def _run(cwd, *args):
        import os
        import subprocess
        import sys

        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "COLUMNS": "200"}
        src = str(pathlib.Path(__file__).resolve().parents[1] / "src")
        env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(
            [sys.executable, "-m", "soup_cli", "adapters", "audit", *args],
            cwd=str(cwd), capture_output=True, text=True, timeout=120,
            env=env, encoding="utf-8", errors="replace",
        )

    def _write(self, tmp_path, config):
        import yaml

        (tmp_path / "adapter_config.json").write_text(json.dumps(_mlx_record()))
        (tmp_path / "soup.yaml").write_text(yaml.safe_dump(config))

    @pytest.mark.integration
    def test_an_unknown_config_key_does_not_break_the_payload(self, tmp_path):
        config = _config()
        config["data"]["trian_on_responses_only"] = True  # a typo, warned about
        self._write(tmp_path, config)

        proc = self._run(tmp_path, ".", "--config", "soup.yaml", "--json")

        payload = json.loads(proc.stdout)  # the assertion: stdout is pure JSON
        assert payload["record_kind"] == "mlx"
        assert "unknown config key" in proc.stderr, (
            "the warning must still be shown, on stderr: " + proc.stderr
        )

    @pytest.mark.integration
    def test_the_warning_is_not_silently_dropped_in_table_mode(self, tmp_path):
        """Routing it to stderr must not amount to hiding it."""
        config = _config()
        config["data"]["trian_on_responses_only"] = True
        self._write(tmp_path, config)

        proc = self._run(tmp_path, ".", "--config", "soup.yaml")

        assert "unknown config key" in proc.stdout + proc.stderr


class TestTheReportSaysWhatItCouldNotCheck:
    """#763 review, finding 2: `unknown_reason` returning `None` unconditionally
    survived — acceptance criterion 4 (say *why* a PEFT record cannot be
    checked) was asserted on the helper, never on what the command prints.
    """

    @pytest.fixture(autouse=True)
    def _run_from_tmp(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _pin_console(monkeypatch)

    PEFT_RECORD = {
        "peft_type": "LORA",
        "base_model_name_or_path": "meta-llama/Llama-3.1-8B",
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"],
    }

    def _write(self, tmp_path, record):
        import yaml

        (tmp_path / "adapter_config.json").write_text(json.dumps(record))
        (tmp_path / "soup.yaml").write_text(yaml.safe_dump(_config()))

    def test_the_command_prints_why_a_peft_record_cannot_be_audited(self, tmp_path):

        from soup_cli.commands.adapters import app

        self._write(tmp_path, self.PEFT_RECORD)
        res = _runner().invoke(app, ["audit", ".", "--config", "soup.yaml"])

        assert res.exit_code == 0, "unknown is not a failure"
        assert "transformers path" in res.output, res.output
        assert "PEFT" in res.output

    def test_a_full_mlx_record_prints_no_such_excuse(self, tmp_path):
        """Control — the explanation must be conditional, not boilerplate."""

        from soup_cli.commands.adapters import app

        self._write(tmp_path, _mlx_record())
        res = _runner().invoke(app, ["audit", ".", "--config", "soup.yaml"])

        assert "transformers path" not in res.output

    def test_the_reason_is_in_the_json_payload_too(self, tmp_path):
        """A CI job reading `--json` needs the same explanation the table gives."""

        from soup_cli.commands.adapters import app

        self._write(tmp_path, self.PEFT_RECORD)
        res = _runner().invoke(app, ["audit", ".", "--config", "soup.yaml", "--json"])
        payload = json.loads(res.stdout)

        assert payload["unknown_reason"] and "PEFT" in payload["unknown_reason"]

    def test_the_reason_is_null_when_everything_was_checkable(self, tmp_path):

        from soup_cli.commands.adapters import app

        self._write(tmp_path, _mlx_record())
        res = _runner().invoke(app, ["audit", ".", "--config", "soup.yaml", "--json"])

        assert json.loads(res.stdout)["unknown_reason"] is None


class TestAChecklessCleanBillIsDistinguishable:
    """#763 review, finding 5: a `{}` record produces ten `unknown` rows, a
    green "No divergences" and exit 0 — correct per the issue (`unknown` must
    not fail), but a CI job could not tell it from "everything agreed".

    `checked_count` gives it that without touching the exit contract.
    """

    def test_nothing_checkable_reports_zero_checked(self):
        from soup_cli.utils.adapter_audit import audit_adapter

        result = audit_adapter(_config(), {})
        assert result.checked_count == 0
        assert result.unknown_count == len(result.rows)
        assert result.exit_code == 0, "the exit contract is unchanged"

    def test_a_full_record_reports_every_row_checked(self):
        from soup_cli.utils.adapter_audit import audit_adapter

        result = audit_adapter(_config(), _mlx_record())
        assert result.checked_count == len(result.rows)
        assert result.unknown_count == 0

    def test_a_divergence_still_counts_as_checked(self):
        """`checked` means "the record could speak to it", not "it agreed"."""
        from soup_cli.utils.adapter_audit import audit_adapter

        result = audit_adapter(_config(), _mlx_record(optimizer="SGD"))
        assert result.checked_count == len(result.rows)
        assert result.diverged_count == 1

    def test_it_reaches_the_json_payload(self):
        from soup_cli.utils.adapter_audit import audit_adapter

        payload = audit_adapter(_config(), {}).to_dict()
        assert payload["checked_count"] == 0
        assert payload["unknown_count"] > 0


class TestAHalfRecordCannotProveAbsenceOfMasking:
    """The mirror of the false clean bill, and the same rule catches it.

    `mlx_sft.py` writes `mask_prompt` and `response_token_mask` together, so a
    record carrying one without the other is foreign or truncated. Reading it
    as `mask_prompt or response_token_mask` turns a missing key into `False`
    and reports a confident DIVERGED for a run whose token-mask path is simply
    unrecorded — claiming to have checked something the record cannot speak to,
    which is the substitution this module exists to refuse, arrived from the
    other direction.

    A truthy key still settles it: masking demonstrably happened, whatever the
    absent key would have said. Only *absence* needs both keys to be provable.
    """

    ASKED = {"data": {"train_on_responses_only": True}}

    @pytest.mark.parametrize(
        "record_keys",
        [
            {"mask_prompt": False},
            {"response_token_mask": False},
        ],
        ids=["only-mask_prompt", "only-response_token_mask"],
    )
    def test_one_false_key_alone_is_unknown_not_diverged(self, record_keys):
        from soup_cli.utils.adapter_audit import audit_adapter

        record = _mlx_record()
        del record["mask_prompt"]
        del record["response_token_mask"]
        record.update(record_keys)

        result = audit_adapter(_config(**self.ASKED), record)
        row = next(r for r in result.rows if r.setting == "data.train_on_responses_only")
        assert row.status == "unknown", (
            f"a single false key proved absence: {row.status} / {row.detail}"
        )
        assert result.exit_code == 0

    @pytest.mark.parametrize(
        "record_keys",
        [
            {"mask_prompt": True},
            {"response_token_mask": True},
        ],
        ids=["only-mask_prompt", "only-response_token_mask"],
    )
    def test_one_true_key_alone_still_settles_it(self, record_keys):
        """Presence is provable from one key; absence is not."""
        from soup_cli.utils.adapter_audit import audit_adapter

        record = _mlx_record()
        del record["mask_prompt"]
        del record["response_token_mask"]
        record.update(record_keys)

        row = next(
            r
            for r in audit_adapter(_config(**self.ASKED), record).rows
            if r.setting == "data.train_on_responses_only"
        )
        assert row.status == "ok"

    def test_both_keys_false_is_still_a_divergence(self):
        """Control — the blocker fix must survive this refinement. Both keys
        present and false is what a real plain-text run writes, and it is
        proof, not a gap."""
        from soup_cli.trainer.mlx_masking import plan_response_masking
        from soup_cli.utils.adapter_audit import audit_adapter

        plan = plan_response_masking(True, {"text": "a plain row"})
        record = _mlx_record(
            mask_prompt=plan.mask_prompt,
            response_token_mask=plan.token_mask,
            train_on_responses_only=True,
        )
        assert "mask_prompt" in record and "response_token_mask" in record

        result = audit_adapter(_config(**self.ASKED), record)
        row = next(r for r in result.rows if r.setting == "data.train_on_responses_only")
        assert row.status == "diverged"
        assert result.exit_code != 0


class TestTheMaskingKeysMustBeRealBooleans:
    """#763 review round 3: `_audit_masking` was the one comparison path that
    skipped the type guard `_cmp` got last round.

    `if bool(prefix_mask) or bool(token_mask)` reads the JSON **string**
    `"false"` as truthy, so a record saying `"mask_prompt": "false"` reported
    `ok` — a false clean bill on the headline row, arrived at through a type
    rather than through a missing key. `mlx_sft.py` always writes real bools,
    so this needs a foreign or hand-edited record: the same threat model the
    `_for_terminal` hardening rests on.
    """

    ASKED = {"data": {"train_on_responses_only": True}}

    @pytest.mark.parametrize(
        "value", ["false", "true", 0, 1, "", None.__class__.__name__],
        ids=["str-false", "str-true", "int-0", "int-1", "empty-str", "str-NoneType"],
    )
    def test_a_non_boolean_effect_key_is_never_a_clean_bill(self, value):
        from soup_cli.utils.adapter_audit import audit_adapter

        record = _mlx_record(mask_prompt=value, response_token_mask=value)
        row = next(
            r
            for r in audit_adapter(_config(**self.ASKED), record).rows
            if r.setting == "data.train_on_responses_only"
        )
        assert row.status != "ok", (
            f"mask_prompt={value!r} was read as agreement"
        )
        assert row.detail, "a refusal has to say why"

    def test_the_string_false_specifically_does_not_read_as_masking(self):
        """The reviewer's exact record."""
        from soup_cli.utils.adapter_audit import audit_adapter

        record = _mlx_record(
            mask_prompt="false", response_token_mask="false", train_on_responses_only=True
        )
        result = audit_adapter(_config(**self.ASKED), record)
        row = next(r for r in result.rows if r.setting == "data.train_on_responses_only")
        assert row.status == "diverged"
        assert "bool" in row.detail or "str" in row.detail, row.detail

    def test_real_booleans_still_behave(self):
        """Control — the guard must not reject the records real runs write."""
        from soup_cli.trainer.mlx_masking import plan_response_masking
        from soup_cli.utils.adapter_audit import audit_adapter

        for sample, expected in (
            ({"messages": [{"role": "user", "content": "q"}]}, "ok"),
            ({"prompt": "q", "completion": "a"}, "ok"),
            ({"text": "plain"}, "diverged"),
        ):
            plan = plan_response_masking(True, sample)
            record = _mlx_record(
                mask_prompt=plan.mask_prompt,
                response_token_mask=plan.token_mask,
                train_on_responses_only=True,
            )
            row = next(
                r
                for r in audit_adapter(_config(**self.ASKED), record).rows
                if r.setting == "data.train_on_responses_only"
            )
            assert row.status == expected, f"{sample} -> {row.status}"


class TestTheExitCodeSeparatesVerdictFromError:
    """#763 review round 3, the maintainer's call: exit 1 meant both "the run
    diverged" and "you typed the path wrong".

    #762 exists for CI composability, and a gate that cannot tell a verdict
    from a usage error fails at exactly that. Repo convention is
    0 = pass / 2 = failed gate / 1 = error (`ship`, `shrink`, `data canary
    check`, `reward stress`). `adapters` itself had no convention to appeal to
    — `scan` uses 3/1, `verify` 1, `bisect` 2 — so this was an unmade decision
    rather than a violation.
    """

    @pytest.fixture(autouse=True)
    def _run_from_tmp(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _pin_console(monkeypatch)

    def _write(self, tmp_path, record=None, config=None):
        import yaml

        if record is not None:
            (tmp_path / "adapter_config.json").write_text(json.dumps(record))
        (tmp_path / "soup.yaml").write_text(yaml.safe_dump(config or _config()))

    def test_a_divergence_exits_2(self, tmp_path):
        from soup_cli.commands.adapters import app

        self._write(tmp_path, _mlx_record(optimizer="SGD"))
        res = _runner().invoke(app, ["audit", ".", "--config", "soup.yaml"])
        assert res.exit_code == 2, res.output

    def test_agreement_still_exits_0(self, tmp_path):
        from soup_cli.commands.adapters import app

        self._write(tmp_path, _mlx_record())
        res = _runner().invoke(app, ["audit", ".", "--config", "soup.yaml"])
        assert res.exit_code == 0, res.output

    def test_unknown_rows_still_exit_0(self, tmp_path):
        """`unknown` is visible, not fatal — unchanged by the new taxonomy."""
        from soup_cli.commands.adapters import app

        self._write(tmp_path, {})
        res = _runner().invoke(app, ["audit", ".", "--config", "soup.yaml"])
        assert res.exit_code == 0, res.output

    @pytest.mark.parametrize(
        "case", ["missing-record", "missing-config", "outside-cwd", "bad-json"]
    )
    def test_every_usage_error_exits_1_not_2(self, tmp_path, case):
        """The whole point: a mistyped path must not look like a verdict."""
        from soup_cli.commands.adapters import app

        if case == "missing-record":
            self._write(tmp_path)
            args = ["audit", ".", "--config", "soup.yaml"]
        elif case == "missing-config":
            self._write(tmp_path, _mlx_record())
            args = ["audit", ".", "--config", "nope.yaml"]
        elif case == "outside-cwd":
            self._write(tmp_path, _mlx_record())
            args = ["audit", str(tmp_path.parent), "--config", "soup.yaml"]
        else:
            self._write(tmp_path, _mlx_record())
            (tmp_path / "adapter_config.json").write_text("{not json")
            args = ["audit", ".", "--config", "soup.yaml"]

        res = _runner().invoke(app, args)
        assert res.exit_code == 1, f"{case} exited {res.exit_code}: {res.output}"

    def test_the_two_are_actually_distinguishable(self, tmp_path):
        """Stated as the property CI depends on, so collapsing them fails."""
        from soup_cli.commands.adapters import app

        self._write(tmp_path, _mlx_record(optimizer="SGD"))
        verdict = _runner().invoke(app, ["audit", ".", "--config", "soup.yaml"])
        usage = _runner().invoke(app, ["audit", ".", "--config", "nope.yaml"])

        assert verdict.exit_code != usage.exit_code
        assert {verdict.exit_code, usage.exit_code} == {2, 1}

    def test_the_docstring_documents_the_taxonomy(self):
        """`scan` states its codes in the docstring, which is what `--help`
        shows; a contract CI depends on must be discoverable there."""
        from soup_cli.commands.adapters import audit

        doc = audit.__doc__ or ""
        for token in ("0", "1", "2"):
            assert token in doc, f"exit code {token} is undocumented"
        assert "diverg" in doc.lower()


class TestTheExplanationsSayTheSubstance:
    """#763 review round 3: every `.detail` was checked for truthiness only,
    so six mutations lived through all 76 tests.

    The dangerous one: **swapping the two DIVERGED masking explanations
    survives.** A run that asked for masking and got none would be told "not
    requested, but the run masked anyway" — the exactly inverted remediation —
    with CI green. The two strings carry opposite user actions and differed by
    nothing any test could see.

    These assert what each message must *do for the reader*: name the
    direction, the resolved value, or the arithmetic. Not the wording, so a
    rewrite stays free; the substance, so a gutting does not.
    """

    MASK_ASKED_GOT_NONE = {"data": {"train_on_responses_only": True}}
    MASK_NOT_ASKED_GOT_SOME = {"data": {"train_on_responses_only": False}}

    def _mask_row(self, cfg_over, **record_over):
        from soup_cli.utils.adapter_audit import audit_adapter

        result = audit_adapter(_config(**cfg_over), _mlx_record(**record_over))
        return next(
            r for r in result.rows if r.setting == "data.train_on_responses_only"
        )

    def test_asked_for_masking_and_got_none_says_so(self):
        row = self._mask_row(
            self.MASK_ASKED_GOT_NONE, mask_prompt=False, response_token_mask=False
        )
        assert row.status == "diverged"
        detail = row.detail.lower()
        assert "requested" in detail
        assert "masked nothing" in detail, row.detail
        assert "anyway" not in detail, (
            "this is the opposite message; the two explanations are swapped"
        )

    def test_did_not_ask_for_masking_but_got_it_says_so(self):
        row = self._mask_row(
            self.MASK_NOT_ASKED_GOT_SOME, mask_prompt=False, response_token_mask=True
        )
        assert row.status == "diverged"
        detail = row.detail.lower()
        assert "not requested" in detail
        assert "anyway" in detail, row.detail
        assert "masked nothing" not in detail, (
            "this is the opposite message; the two explanations are swapped"
        )

    def test_the_two_masking_explanations_are_not_interchangeable(self):
        """Stated directly, so a swap cannot pass by satisfying both loosely."""
        got_none = self._mask_row(
            self.MASK_ASKED_GOT_NONE, mask_prompt=False, response_token_mask=False
        ).detail
        got_some = self._mask_row(
            self.MASK_NOT_ASKED_GOT_SOME, mask_prompt=False, response_token_mask=True
        ).detail

        assert got_none != got_some
        # Each must carry the remediation for its own direction and not the
        # other's: "use chatml data" is useless advice to someone who asked for
        # no masking and got some.
        assert "chatml" in got_none and "chatml" not in got_some, (
            f"got_none={got_none!r}\ngot_some={got_some!r}"
        )

    def test_the_optimizer_detail_names_the_resolved_backend_value(self):
        """Gutting it to "the optimizer differs" survived. The whole value of
        this row is that `adamw_torch` resolves to `AdamW` on MLX — without
        that, the reader cannot tell a real swap from a naming difference."""
        from soup_cli.utils.adapter_audit import audit_adapter

        row = next(
            r
            for r in audit_adapter(_config(), _mlx_record(optimizer="SGD")).rows
            if r.setting == "optimizer"
        )
        assert row.status == "diverged"
        assert "adamw_torch" in row.detail, row.detail
        assert "AdamW" in row.detail, "the resolved name is the point of the row"
        assert "SGD" in row.detail, "and what actually ran"

    def test_the_warmup_detail_shows_the_arithmetic_that_rounded_away(self):
        """This string *is* the motivating case of #762. "warmup differs" does
        not tell anyone that 0.03 x 12 is zero steps, which is the finding."""
        from soup_cli.utils.adapter_audit import audit_adapter

        cfg = _config(training={"warmup_ratio": 0.03})
        row = next(
            r
            for r in audit_adapter(cfg, _mlx_record(warmup_updates=0)).rows
            if r.setting == "warmup_ratio"
        )
        assert row.status == "diverged"
        assert "0.03" in row.detail, row.detail
        assert "12" in row.detail, "the update count is half the arithmetic"
        assert "0" in row.detail
        assert "warmup_ratio" in row.detail or "gradient_accumulation" in row.detail, (
            "a divergence the user cannot act on is only half reported"
        )

    def test_the_expected_warmup_count_is_shown_when_it_simply_disagrees(self):
        """The non-rounding branch has its own arithmetic to show."""
        from soup_cli.utils.adapter_audit import audit_adapter

        cfg = _config(training={"warmup_ratio": 0.25})  # 0.25 x 12 = 3
        row = next(
            r
            for r in audit_adapter(cfg, _mlx_record(warmup_updates=7)).rows
            if r.setting == "warmup_ratio"
        )
        assert row.status == "diverged"
        assert "3" in row.detail, row.detail
        assert "0.25" in row.detail and "12" in row.detail

    def test_the_bool_type_detail_names_both_types(self):
        from soup_cli.utils.adapter_audit import audit_adapter

        row = next(
            r
            for r in audit_adapter(
                _config(training={"max_grad_norm": 1.0}), _mlx_record(max_grad_norm=True)
            ).rows
            if r.setting == "max_grad_norm"
        )
        assert row.status == "diverged"
        assert "bool" in row.detail, row.detail
        assert "float" in row.detail, "the reader needs to know what was expected"


class TestTheSummaryCountsWhatItFound:
    """#763 review round 3: hardcoding the summary's divergence count to 1
    survived, because no test read the number back."""

    @pytest.fixture(autouse=True)
    def _run_from_tmp(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _pin_console(monkeypatch)

    def _run(self, tmp_path, record):
        import yaml

        from soup_cli.commands.adapters import app

        (tmp_path / "adapter_config.json").write_text(json.dumps(record))
        (tmp_path / "soup.yaml").write_text(yaml.safe_dump(_config()))
        return _runner().invoke(app, ["audit", ".", "--config", "soup.yaml"])

    @pytest.mark.parametrize(
        "overrides,expected",
        [
            ({"optimizer": "SGD"}, 1),
            ({"optimizer": "SGD", "scheduler": "linear"}, 2),
            (
                {"optimizer": "SGD", "scheduler": "linear", "weight_decay": 0.9},
                3,
            ),
        ],
        ids=["one", "two", "three"],
    )
    def test_the_printed_count_matches_the_number_found(
        self, tmp_path, overrides, expected
    ):
        res = self._run(tmp_path, _mlx_record(**overrides))
        assert f"{expected} divergence(s)" in res.output, res.output
        assert res.exit_code == 2

    def test_the_unchecked_count_is_printed_too(self, tmp_path):
        record = _mlx_record(optimizer="SGD")
        for key in ("max_grad_norm", "weight_decay"):
            del record[key]
        res = self._run(tmp_path, record)
        assert "1 divergence(s)" in res.output, res.output
        assert "2 unchecked" in res.output, res.output


class TestAdapterSuppliedTextCannotDriveTheTerminal:
    """#763 review round 3: the control-byte hardening had no test at all.

    `adapter_config.json` is downloadable, so its strings are untrusted input
    that this command prints. `rich.markup.escape` neutralises Rich's `[...]`
    tags but not raw ANSI/OSC bytes, hence `_for_terminal` beside it — a
    recorded `"AdamW\\x1b[2J"` would clear the screen, and `\\x1b]0;...\\x07`
    would rewrite the title bar, which is how a refusal gets hidden.

    Asserted as the *absence of specific dangerous sequences* rather than as
    "no \\x1b anywhere", because Rich legitimately emits its own colour codes
    under `FORCE_COLOR=1` and the suite is run that way.
    """

    # Erase-display, then an OSC title-set terminated by BEL.
    PAYLOAD = "AdamW\x1b[2J\x1b]0;pwned\x07"

    @pytest.fixture(autouse=True)
    def _run_from_tmp(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _pin_console(monkeypatch)

    def _run(self, tmp_path, record, *extra):
        import yaml

        from soup_cli.commands.adapters import app

        (tmp_path / "adapter_config.json").write_text(json.dumps(record))
        (tmp_path / "soup.yaml").write_text(yaml.safe_dump(_config()))
        return _runner().invoke(app, ["audit", ".", "--config", "soup.yaml", *extra])

    def test_no_escape_sequence_from_the_record_reaches_the_terminal(self, tmp_path):
        res = self._run(tmp_path, _mlx_record(optimizer=self.PAYLOAD))

        assert "\x1b[2J" not in res.output, "erase-display survived to the terminal"
        assert "\x1b]0;" not in res.output, "OSC title-set survived to the terminal"
        assert "\x07" not in res.output, "BEL survived to the terminal"

    def test_the_value_is_neutralised_rather_than_dropped(self, tmp_path):
        """Stripping must not silently swallow the row — the audit still has to
        report what it found, just inertly."""
        res = self._run(tmp_path, _mlx_record(optimizer=self.PAYLOAD))

        assert "AdamW" in res.output
        assert "[2J" in res.output, "the payload should remain visible as inert text"
        assert "DIVERGED" in res.output
        assert res.exit_code == 2

    def test_rich_markup_in_a_table_cell_is_printed_not_interpreted(self, tmp_path):
        """`escape()` is the other half of the pair: `repr()` neutralises
        control bytes but does nothing about `[red]`, which Rich would act on.

        Uses `scheduler`, which diverges through `_cmp` and therefore carries
        an **empty** detail -- so the payload reaches the table cell and
        nowhere else. Asserting on `optimizer` instead would pass with the
        cell's `escape()` deleted, because the same text reappears in that
        row's detail line, which escapes separately.
        """
        res = self._run(tmp_path, _mlx_record(scheduler="cosine[red]X[/]"))

        assert "[red]" in res.output, "the tag was interpreted instead of shown"

    def test_rich_markup_in_a_detail_line_is_printed_not_interpreted(self, tmp_path):
        """The detail line escapes separately from the cell, so it is asserted
        separately -- on that line alone, for the same reason as above."""
        res = self._run(tmp_path, _mlx_record(optimizer="AdamW[red]X[/]"))

        detail_lines = [
            line for line in res.output.splitlines() if line.strip().startswith("optimizer:")
        ]
        assert detail_lines, res.output
        assert "[red]" in detail_lines[0], (
            f"markup was interpreted in the detail line: {detail_lines[0]!r}"
        )

    def test_no_explanation_can_carry_a_control_byte_in_the_first_place(self):
        """The invariant the detail line's `_for_terminal` insures against.

        Every ``detail`` that interpolates a record value does so through
        ``repr()``, which renders ``\x1b`` as four printable characters -- so
        no raw control byte can reach ``row.detail`` by any current path, and
        stripping it there is defence in depth rather than a live guard. That
        makes removing it an *equivalent* mutation, which is only an honest
        claim if the invariant is actually pinned. This pins it.
        """
        from soup_cli.utils.adapter_audit import audit_adapter

        payload = "X\x1b[2J\x1b]0;p\x07".encode().decode("unicode_escape")
        hostile = _mlx_record(
            optimizer=payload,
            scheduler=payload,
            mask_prompt=payload,
            response_token_mask=payload,
            max_grad_norm=payload,
            weight_decay=payload,
            grad_checkpoint=payload,
            grad_accumulation_steps=payload,
            lora_parameters={"rank": payload, "scale": payload},
        )
        rows = audit_adapter(_config(), hostile).rows
        assert any(r.status == "diverged" for r in rows), "the fixture must exercise it"
        for row in rows:
            assert "\x1b" not in row.detail, f"{row.setting}: {row.detail!r}"
            assert "\x07" not in row.detail, f"{row.setting}: {row.detail!r}"

    def test_the_unknown_reason_text_is_a_fixed_string_not_record_derived(self):
        """The third `_for_terminal` call site, and the same honesty problem.

        `unknown_reason` is a lookup over `classify_record`'s three literal
        results, so nothing an adapter can write reaches it -- also an
        equivalent mutation, and also worth pinning rather than asserting.
        """
        from soup_cli.utils.adapter_audit import classify_record, unknown_reason

        kinds = {
            classify_record(r)
            for r in ({}, {"peft_type": "LORA"}, {"fine_tune_type": "lora"},
                      {"total_updates": 1}, _mlx_record())
        }
        assert kinds == {"unknown", "peft", "mlx"}, kinds
        for kind in kinds:
            reason = unknown_reason(kind)
            if reason is None:
                continue
            assert not any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in reason), reason

    def test_the_json_path_is_unaffected_by_the_stripping(self, tmp_path):
        """`--json` must carry the record's bytes faithfully; `_for_terminal`
        is a terminal concern, and a machine consumer wants the real value."""
        res = self._run(tmp_path, _mlx_record(optimizer=self.PAYLOAD), "--json")
        payload = json.loads(res.stdout)

        row = next(r for r in payload["rows"] if r["setting"] == "optimizer")
        assert row["ran"] == self.PAYLOAD, "JSON should not be terminal-sanitised"

    def test_a_control_byte_in_a_nested_lora_value_is_also_stripped(self, tmp_path):
        """Not just the top-level keys — `lora_parameters` is record-supplied
        too and reaches the same cells."""
        res = self._run(
            tmp_path,
            _mlx_record(lora_parameters={"rank": self.PAYLOAD, "scale": 2.0}),
        )

        assert "\x1b[2J" not in res.output
        assert "\x07" not in res.output


class TestAMalformedRecordIsReportedNotCrashedOn:
    """Found while pinning the control-byte invariant, not raised in review.

    The two arithmetic paths trusted the record's types. A hand-edited or
    foreign `adapter_config.json` with string numerics reached
    `int(total_updates)` and `scale * rank` and raised out of the command:

        Error: ValueError: invalid literal for int() with base 10: 'twelve'

    Same threat model as the terminal-control hardening and the masking type
    guard -- an untrusted downloaded file -- and a traceback is a worse
    outcome than either, because `audit` is meant to be a CI gate and a crash
    is not a verdict.
    """

    @pytest.fixture(autouse=True)
    def _run_from_tmp(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _pin_console(monkeypatch)

    def test_a_non_numeric_total_updates_does_not_raise(self):
        from soup_cli.utils.adapter_audit import audit_adapter

        result = audit_adapter(
            _config(), _mlx_record(total_updates="twelve", warmup_updates=0)
        )
        row = next(r for r in result.rows if r.setting == "warmup_ratio")
        assert row.status != "ok", "a record it cannot read is not agreement"
        assert row.detail

    def test_a_non_numeric_warmup_updates_does_not_raise(self):
        from soup_cli.utils.adapter_audit import audit_adapter

        result = audit_adapter(_config(), _mlx_record(warmup_updates="three"))
        row = next(r for r in result.rows if r.setting == "warmup_ratio")
        assert row.status != "ok"

    def test_non_numeric_lora_parameters_do_not_raise(self):
        from soup_cli.utils.adapter_audit import audit_adapter

        result = audit_adapter(
            _config(), _mlx_record(lora_parameters={"rank": "eight", "scale": "two"})
        )
        alpha = next(r for r in result.rows if r.setting == "lora.alpha")
        rank = next(r for r in result.rows if r.setting == "lora.r")
        assert alpha.status != "ok", "a derived value it cannot compute is not agreement"
        assert rank.status != "ok"

    def test_the_command_reports_rather_than_tracebacks(self, tmp_path):
        """End to end: the shape that raised out of the CLI."""
        import yaml

        from soup_cli.commands.adapters import app

        (tmp_path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "fine_tune_type": "lora",
                    "total_updates": "twelve",
                    "warmup_updates": 0,
                    "lora_parameters": {"rank": "eight", "scale": "two"},
                }
            )
        )
        (tmp_path / "soup.yaml").write_text(yaml.safe_dump(_config()))

        res = _runner().invoke(app, ["audit", ".", "--config", "soup.yaml"])

        assert res.exception is None or isinstance(res.exception, SystemExit), (
            f"raised out of the command: {res.exception!r}"
        )
        assert res.exit_code in (0, 2), res.output
        assert "Traceback" not in res.output

    def test_a_boolean_warmup_count_is_not_a_number(self):
        """`_is_number` excludes `bool` on purpose, and this is why.

        `warmup_ratio: 0` over 12 updates expects 0 warmup updates, and
        `0 == False` is True in Python -- so a record saying
        `warmup_updates: false` was handed a clean bill by the arithmetic.
        Same defect `_cmp` and `_audit_masking` already guard.
        """
        from soup_cli.utils.adapter_audit import audit_adapter

        result = audit_adapter(
            _config(training={"warmup_ratio": 0.0}), _mlx_record(warmup_updates=False)
        )
        row = next(r for r in result.rows if r.setting == "warmup_ratio")
        assert row.status != "ok", "a bool was read as the number 0"

    def test_boolean_lora_scale_and_rank_are_not_numbers(self):
        """`True * True` is `1`, so a config asking for alpha 1 agreed with a
        record carrying two booleans."""
        from soup_cli.utils.adapter_audit import audit_adapter

        cfg = _config(training={"lora": {"r": 1, "alpha": 1}})
        result = audit_adapter(
            cfg, _mlx_record(lora_parameters={"rank": True, "scale": True})
        )
        row = next(r for r in result.rows if r.setting == "lora.alpha")
        assert row.status != "ok", "two bools multiplied into a matching alpha"

    def test_well_formed_records_are_unaffected(self):
        """Control — the guards must not reject what real runs write."""
        from soup_cli.utils.adapter_audit import audit_adapter

        result = audit_adapter(_config(), _mlx_record())
        assert result.diverged_count == 0
        assert result.unknown_count == 0
