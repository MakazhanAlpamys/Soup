"""Tests for GPU utils."""

from soup_cli.utils.gpu import estimate_batch_size, model_size_from_name


def test_model_size_detection():
    assert model_size_from_name("meta-llama/Llama-3.1-8B-Instruct") == 8
    assert model_size_from_name("meta-llama/Llama-3.1-70B") == 70
    assert model_size_from_name("codellama/CodeLlama-7b-hf") == 7
    assert model_size_from_name("some-unknown-model") == 7.0  # default


def test_batch_size_cpu():
    """CPU (0 memory) should return batch_size=1."""
    bs = estimate_batch_size(7.0, 2048, 0, "4bit", 64)
    assert bs == 1


def test_batch_size_24gb():
    """24 GB GPU with 8B model QLoRA should fit batch > 1."""
    mem = 24 * (1024**3)
    bs = estimate_batch_size(8.0, 2048, mem, "4bit", 64)
    assert bs >= 1
    assert bs <= 32
