"""#1850 — the MLX SFT seeding wiring, on every platform.

tests/test_mlx_sft_seed.py checks the behaviour on a real mlx-lm model, so it
needs Apple Silicon and is skipped everywhere else. This file runs the real
``MLXSFTTrainerWrapper.train()`` body against the fake ``mlx`` / ``mlx_lm``
modules from tests/test_issue634_mlx_resume.py and records the calls that
matter: ``mx.random.seed``, the LoRA conversion, ``np.random.seed`` and
mlx-lm's ``train()``. It pins which value reaches each RNG (``0`` is a seed,
``data_seed`` owns the batch order, unset is 42) and the order: ``mx.random``
before the adapter is built, numpy after it and immediately before ``train()``.

No ``importorskip``: the fakes replace whatever mlx is (or is not) installed.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from tests.test_issue634_mlx_resume import _FakeMlxModel, _install_fake_mlx, _mlx_wrapper


def _run(tmp_path, monkeypatch, **training):
    _install_fake_mlx(monkeypatch)
    events = []
    sys.modules["mlx.core"].random = types.SimpleNamespace(
        seed=lambda value: events.append(("mx", value))
    )
    sys.modules["mlx_lm.tuner.utils"].linear_to_lora_layers = (
        lambda model, num_layers, config: events.append(("lora", None))
    )
    original = np.random.seed
    monkeypatch.setattr(
        np.random,
        "seed",
        lambda value=None: (events.append(("np", value)), original(value)),
    )
    sys.modules["mlx_lm.tuner.trainer"].train = lambda **_kw: events.append(("train", None))

    wrapper = _mlx_wrapper(tmp_path, epochs=1, lr=1e-4, batch_size=1, **training)
    wrapper.model, wrapper.tokenizer = _FakeMlxModel(), object()
    wrapper._dataset = {"train": [{"text": "hi"}], "val": []}
    wrapper.train()
    return events


@pytest.mark.parametrize(
    "training, mx_seed, np_seed",
    [
        ({}, 42, 42),
        ({"seed": 7}, 7, 7),
        ({"seed": 0}, 0, 0),
        ({"seed": 7, "data_seed": 9}, 7, 9),
        ({"seed": 7, "data_seed": 0}, 7, 0),
    ],
)
def test_seeds_reach_mlx_and_numpy_in_order(tmp_path, monkeypatch, training, mx_seed, np_seed):
    events = _run(tmp_path, monkeypatch, **training)

    assert [e for e in events if e[0] == "mx"] == [("mx", mx_seed)]
    assert [e for e in events if e[0] == "np"] == [("np", np_seed)]
    names = [e[0] for e in events]
    assert names.count("lora") == 1 and names.count("train") == 1
    assert (
        names.index("mx") < names.index("lora") < names.index("np") == names.index("train") - 1
    )
