from tests.test_repeat_backward import module


def test_later_corruption_is_not_ignored():
    assert module.repeated_backward_verdict([4, 4, 3], 4)


def test_nonfinite_gradient_is_rejected():
    torch = __import__("pytest").importorskip("torch")

    with __import__("pytest").raises(module.MeasurementInvalidError):
        module.compare_gradient_maps(
            {"lora_A": torch.tensor([float("nan")])},
            {"lora_A": torch.tensor([1.0])},
        )
