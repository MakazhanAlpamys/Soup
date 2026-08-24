"""Issue #320 — TrainerCallback modules must not import transformers at module scope.

Eleven modules resolved their TrainerCallback base at class-definition time,
eagerly importing transformers (and therefore torch) on every module import.
The fix defers class construction via PEP 562 ``__getattr__``: importing the
module no longer pulls transformers; the callback class is built on first
attribute access, inheriting from the real ``TrainerCallback`` (or ``object``
when transformers is absent).

This suite verifies:
1. Each callback class is still a real ``TrainerCallback`` subclass (the #308
   property — HF's CallbackHandler dispatches events via ``getattr``).
2. Each callback still inherits the no-op stubs for unimplemented events.
3. The lazy class is cached (same object on repeated access).
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

CALLBACK_MODULES = [
    ("soup_cli.utils.reward_hack_control", "RewardHackMitigationCallback"),
    ("soup_cli.utils.reward_hacking", "RewardHackCallback"),
    ("soup_cli.utils.echo_trap", "EchoTrapCallback"),
    ("soup_cli.utils.minillm", "MiniLLMCallback"),
    ("soup_cli.utils.rl_checkpoint", "RLCheckpointCallback"),
    ("soup_cli.utils.relora", "ReLoRACallback"),
    ("soup_cli.utils.lisa", "LisaCallback"),
    ("soup_cli.monitoring.hf_push", "HFPushCallback"),
    ("soup_cli.monitoring.curriculum_callback", "DynamicCurriculumCallback"),
    ("soup_cli.monitoring.grpo_stability_callback", "GRPOStabilityCallback"),
    ("soup_cli.monitoring.callback", "SoupTrainerCallback"),
]


def _ids(params):
    return [f"{m}.{c}" for m, c in params]


@pytest.mark.parametrize("mod_path,cls_name", CALLBACK_MODULES, ids=_ids(CALLBACK_MODULES))
def test_callback_is_trainer_callback_subclass(mod_path, cls_name):
    """The lazily-built class must be a real TrainerCallback subclass."""
    try:
        from transformers import TrainerCallback
    except ImportError:
        pytest.skip("transformers not installed")
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    assert issubclass(cls, TrainerCallback), (
        f"{cls_name} is not a TrainerCallback subclass; bases = {cls.__bases__}"
    )


@pytest.mark.parametrize("mod_path,cls_name", CALLBACK_MODULES, ids=_ids(CALLBACK_MODULES))
def test_callback_has_noop_event_stubs(mod_path, cls_name):
    """Each callback must inherit no-op stubs for all Trainer lifecycle events."""
    try:
        from transformers import TrainerCallback  # noqa: F401
    except ImportError:
        pytest.skip("transformers not installed")
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    for event in (
        "on_epoch_begin",
        "on_epoch_end",
        "on_train_begin",
        "on_train_end",
        "on_step_begin",
        "on_step_end",
        "on_log",
        "on_save",
        "on_evaluate",
        "on_prediction_step",
    ):
        assert hasattr(cls, event), f"{cls_name} missing {event}"


@pytest.mark.parametrize("mod_path,cls_name", CALLBACK_MODULES, ids=_ids(CALLBACK_MODULES))
def test_lazy_class_is_cached(mod_path, cls_name):
    """Repeated access must return the same class object (cached in globals)."""
    mod = importlib.import_module(mod_path)
    first = getattr(mod, cls_name)
    second = getattr(mod, cls_name)
    assert first is second, f"{cls_name} was rebuilt on second access"


@pytest.mark.parametrize(
    "module,builder_call",
    [
        (
            "soup_cli.utils.echo_trap",
            "m.build_echo_trap_callback(threshold=0.5)",
        ),
        (
            "soup_cli.utils.reward_hacking",
            "m.build_reward_hack_callback(detector='info_rm')",
        ),
        (
            "soup_cli.utils.minillm",
            "m.build_minillm_callback(m.MiniLLMConfig())",
        ),
        (
            "soup_cli.utils.rl_checkpoint",
            "m.build_rl_checkpoint_callback("
            "m.RLCheckpointConfig(save_every_steps=1), output_dir='out')",
        ),
        (
            "soup_cli.monitoring.hf_push",
            "m.build_push_callback("
            "repo_id='user/repo', output_dir='out', explicit_token='tok')",
        ),
    ],
)
def test_builder_invocation_no_nameerror_in_subprocess(module: str, builder_call: str):
    """Builder calls must resolve lazy callback names without NameError."""
    probe = (
        "import importlib\n"
        f"m = importlib.import_module('{module}')\n"
        "try:\n"
        f"    {builder_call}\n"
        "except NameError as exc:\n"
        "    raise SystemExit(f'NAMEERROR:{exc}')\n"
        "print('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "NAMEERROR:" not in proc.stdout + proc.stderr
