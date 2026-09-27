"""Where torchao's config classes actually live, and the one version floor (#826).

Soup used to resolve torchao configs with ``hasattr(torchao.quantization, name)``
against names torchao does not export. Three of the four export schemes and the
whole ``training.nvfp4`` path asked for ``NVFP4Config``,
``Int8DynActInt4Config`` and ``Float8DynActFloat8Config``; none of those exists
in 0.18.0, the newest release. The refusal told the user to upgrade, and there
was nothing newer to upgrade to.

So each name is resolved by importing the module that defines it. A rename then
fails at a named import with the paths it tried in the message, rather than
degrading into advice that cannot work. Measured against torchao 0.18.0:

=========================  =========================================================
what Soup wants            where it is
=========================  =========================================================
int4 weight-only export    ``torchao.quantization.Int4WeightOnlyConfig``
int8-act/int4-weight       ``torchao.prototype.quantization.int4.inference_workflow``
                           ``.Int8DynamicActivationInt4WeightConfig``
float8 act/weight          ``torchao.quantization``
                           ``.Float8DynamicActivationFloat8WeightConfig``
NVFP4 export (PTQ)         ``torchao.prototype.mx_formats.inference_workflow``
                           ``.NVFP4DynamicActivationNVFP4WeightConfig``
NVFP4 *training*           ``torchao.prototype.moe_training.nvfp4_training``
                           ``.nvfp4_training.NVFP4TrainingConfig``
=========================  =========================================================

Three of those are ``torchao.prototype``, which is by definition not a promised
location, so each key carries a tuple of candidate modules and the **public**
path is tried first. A release that graduates one of these into
``torchao.quantization`` is then picked up with no change here, which is the
shape ``trainer/_trl_compat.py:resolve_trl_symbol`` already uses for the three
configs ``trl`` moved into ``trl.experimental``. The candidates below are the
paths that were probed against 0.18.0, not guesses: today the public candidate
misses for those three and the prototype one answers.

The training and export entries are deliberately different classes.
``quantize_(model, <inference config>)`` is post-training weight quantization;
only ``NVFP4TrainingConfig`` says, in its own docstring, that it "replaces
nn.Linear modules with NVFP4Linear, which quantizes all three GEMMs (forward and
backward) to NVFP4", which is what ``training.nvfp4`` promises.
"""

from __future__ import annotations

from typing import Any, Tuple

#: The one torchao floor. Every message and ``pyproject.toml``'s ``[qat]`` extra
#: read it; ``tests/test_issue826_torchao_class_names.py`` pins the pyproject
#: pin to this constant, because four different floors (0.4.0, 0.5.0, 0.7.0 and
#: "0.5") disagreeing is how #826 stayed invisible.
#:
#: 0.18.0 rather than something older because that is the release these module
#: paths were verified against; the prototype paths are not promised to be
#: stable, so claiming an older floor would be a guess.
TORCHAO_MIN_VERSION = "0.18.0"

#: ``key -> ((module path, attribute), ...)``, most public candidate first.
#: The attribute is repeated per candidate because a graduated class is free to
#: be renamed on the way out of ``prototype``.
TORCHAO_CLASSES: dict[str, Tuple[Tuple[str, str], ...]] = {
    "Int4WeightOnly": (("torchao.quantization", "Int4WeightOnlyConfig"),),
    "Int8DynActInt4": (
        ("torchao.quantization", "Int8DynamicActivationInt4WeightConfig"),
        (
            "torchao.prototype.quantization.int4.inference_workflow",
            "Int8DynamicActivationInt4WeightConfig",
        ),
    ),
    "Float8DynActFloat8": (
        ("torchao.quantization", "Float8DynamicActivationFloat8WeightConfig"),
    ),
    "NVFP4": (
        ("torchao.quantization", "NVFP4DynamicActivationNVFP4WeightConfig"),
        (
            "torchao.prototype.mx_formats.inference_workflow",
            "NVFP4DynamicActivationNVFP4WeightConfig",
        ),
        ("torchao.prototype.mx_formats", "NVFP4DynamicActivationNVFP4WeightConfig"),
    ),
    "NVFP4Training": (
        ("torchao.quantization", "NVFP4TrainingConfig"),
        (
            "torchao.prototype.moe_training.nvfp4_training.nvfp4_training",
            "NVFP4TrainingConfig",
        ),
    ),
    "quantize_": (("torchao.quantization", "quantize_"),),
}


def torchao_install_hint(what: str) -> str:
    """The one install line, so the floor is never typed twice."""
    return f"{what} requires torchao (pip install 'torchao>={TORCHAO_MIN_VERSION}')."


def resolve_torchao_class(key: str) -> Any:
    """Return the torchao class ``key`` names, or raise saying where it looked.

    Args:
        key: a key of :data:`TORCHAO_CLASSES`.

    Raises:
        RuntimeError: torchao is absent, or the class moved out of every
            candidate module. The message carries each path tried, so a rename
            is actionable instead of "upgrade torchao" against the newest
            release.
    """
    import importlib

    # The root package first, before any submodule. Importing
    # ``torchao.prototype.…`` directly answers from ``sys.modules`` when an
    # earlier import in the same process cached it, so an absent torchao was
    # only absent if nothing had imported it yet: the "torchao is missing"
    # branch was unreachable in a process that had ever touched torchao, and
    # test_v07121.py's absent-torchao test passed alone and failed after
    # tests/test_issue826_torchao_class_names.py had imported the real package.
    try:
        importlib.import_module("torchao")
    except ImportError as exc:
        raise RuntimeError(
            f"{torchao_install_hint(key)} Could not import torchao "
            f"({type(exc).__name__}: {exc})."
        ) from exc

    candidates = TORCHAO_CLASSES[key]
    tried: list[str] = []
    for module_path, attribute in candidates:
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            tried.append(f"{module_path}.{attribute} ({type(exc).__name__}: {exc})")
            continue
        try:
            return getattr(module, attribute)
        except AttributeError:
            tried.append(f"{module_path}.{attribute} (not defined there)")

    looked = "; ".join(tried)
    raise RuntimeError(
        f"this torchao has no class for {key}. Soup looked for {looked}. The "
        f"last of those exists in torchao {TORCHAO_MIN_VERSION}; a newer release "
        "may have moved it, in which case soup_cli/utils/torchao_compat.py is the "
        "one place to update."
    )
