"""Tests for unlearn_alpha weighting in UnlearnTrainerWrapper (#2108)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from soup_cli.config.schema import SoupConfig
from soup_cli.trainer.unlearn import UnlearnTrainerWrapper


def _make_wrapper(method: str, unlearn_alpha: float | None) -> UnlearnTrainerWrapper:
    training_dict: dict = {"unlearn_method": method}
    if unlearn_alpha is not None:
        training_dict["unlearn_alpha"] = unlearn_alpha
    cfg = SoupConfig(
        base="test-model",
        task="unlearn",
        data={"train": "test.jsonl", "forget_set": "f.jsonl", "retain_set": "r.jsonl"},
        training=training_dict,
    )
    wrapper = UnlearnTrainerWrapper(cfg)
    wrapper.tokenizer = MagicMock()
    wrapper._retain = [("r_prompt", "r_target")]
    return wrapper


def _dummy_tokenize_pair(tok, p, t, max_length=None):
    return (torch.tensor([[1, 2]]), torch.tensor([[1, 2]]))


class TestUnlearnAlphaPreference:
    """_step_preference (NPO / SimNPO) retain loss weighting (#2108)."""

    @pytest.mark.parametrize("method", ["npo", "simnpo"])
    @pytest.mark.parametrize(
        ("alpha_config", "expected_alpha"),
        [
            (0.0, 0.0),
            (None, 1.0),
            (0.5, 0.5),
        ],
    )
    def test_preference_retain_weight(
        self, method: str, alpha_config: float | None, expected_alpha: float,
    ):
        wrapper = _make_wrapper(method, alpha_config)
        wrapper._retain_ce = lambda tp, dev: torch.tensor(10.0)
        wrapper.model = MagicMock(return_value=MagicMock(logits=torch.zeros((1, 2, 10))))
        wrapper.ref_model = MagicMock(return_value=MagicMock(logits=torch.zeros((1, 2, 10))))

        with (
            patch("soup_cli.utils.unlearn_kernels.npo_loss", return_value=torch.tensor(1.0)),
            patch("soup_cli.utils.unlearn_kernels.simnpo_loss", return_value=torch.tensor(1.0)),
            patch(
                "soup_cli.utils.unlearn_kernels.sequence_logprob",
                return_value=torch.tensor([0.0]),
            ),
        ):
            total_loss = wrapper._step_preference(
                _dummy_tokenize_pair, "prompt", "target", method, "cpu",
            )

        expected_total = 1.0 + expected_alpha * 10.0
        assert total_loss is not None
        msg = f"Expected loss {expected_total} for alpha={alpha_config}, got {total_loss.item()}"
        assert torch.isclose(total_loss, torch.tensor(expected_total)), msg


class TestUnlearnAlphaRMU:
    """_step_rmu retain loss weighting (#2108)."""

    @pytest.mark.parametrize(
        ("alpha_config", "expected_alpha"),
        [
            (0.0, 0.0),
            (None, 1.0),
            (0.5, 0.5),
        ],
    )
    def test_rmu_retain_weight(self, alpha_config: float | None, expected_alpha: float):
        wrapper = _make_wrapper("rmu", alpha_config)

        class DummyLayer(torch.nn.Module):
            def forward(self, x):
                return x

        dummy_layer = DummyLayer()
        wrapper._resolve_ref_rmu_layer = lambda: dummy_layer

        class DummyModel(torch.nn.Module):
            def forward(self, input_ids):
                acts = torch.ones((1, 2, 4))
                for hook in list(dummy_layer._forward_hooks.values()):
                    hook(dummy_layer, None, (acts,))
                return acts

        wrapper.model = DummyModel()
        wrapper.ref_model = DummyModel()

        recorded_alphas = []

        def fake_rmu_loss(forget_acts, control_vec, retain_acts, retain_frozen, *, alpha=1.0):
            recorded_alphas.append(alpha)
            return torch.tensor(0.0)

        with patch("soup_cli.utils.unlearn_kernels.rmu_loss", fake_rmu_loss):
            wrapper._step_rmu(
                _dummy_tokenize_pair,
                "f_prompt",
                "f_target",
                ("r_prompt", "r_target"),
                torch.zeros(4),
                dummy_layer,
                "cpu",
            )

        assert len(recorded_alphas) == 1
        msg = f"Expected alpha {expected_alpha} for alpha={alpha_config}, got {recorded_alphas[0]}"
        assert recorded_alphas[0] == expected_alpha, msg
