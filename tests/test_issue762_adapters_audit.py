"""Issue #762 — `soup adapters audit`: did the run do what the config asked?

Four merged fixes (#683, #684, #685, #686, #749) taught the MLX path to record
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
        """#683: `train_on_responses_only: true` reaching nothing."""
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

    def _write(self, tmp_path, record, config):
        import yaml

        (tmp_path / "adapter_config.json").write_text(json.dumps(record))
        cfg = tmp_path / "soup.yaml"
        cfg.write_text(yaml.safe_dump(config))
        return cfg

    def test_a_clean_run_exits_zero(self, tmp_path):
        from typer.testing import CliRunner

        from soup_cli.commands.adapters import app

        cfg = self._write(tmp_path, _mlx_record(), _config())
        res = CliRunner().invoke(app, ["audit", ".", "--config", cfg.name])
        assert res.exit_code == 0, res.output

    def test_a_divergent_run_exits_non_zero(self, tmp_path):
        """So it composes into CI and `soup ship` rather than only being read."""
        from typer.testing import CliRunner

        from soup_cli.commands.adapters import app

        cfg = self._write(tmp_path, _mlx_record(optimizer="SGD"), _config())
        res = CliRunner().invoke(app, ["audit", ".", "--config", cfg.name])
        assert res.exit_code != 0
        assert "optimizer" in res.output

    def test_json_output_is_machine_readable(self, tmp_path):
        from typer.testing import CliRunner

        from soup_cli.commands.adapters import app

        cfg = self._write(tmp_path, _mlx_record(optimizer="SGD"), _config())
        res = CliRunner().invoke(app, ["audit", ".", "--config", cfg.name, "--json"])
        payload = json.loads(res.output)
        assert payload["diverged_count"] == 1
        assert any(r["setting"] == "optimizer" for r in payload["rows"])

    def test_a_missing_adapter_config_fails_clearly(self, tmp_path):
        import yaml
        from typer.testing import CliRunner

        from soup_cli.commands.adapters import app

        cfg = tmp_path / "soup.yaml"
        cfg.write_text(yaml.safe_dump(_config()))
        res = CliRunner().invoke(app, ["audit", ".", "--config", cfg.name])
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
