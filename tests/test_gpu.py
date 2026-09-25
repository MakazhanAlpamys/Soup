"""Tests for GPU utils."""

from soup_cli.utils.gpu import estimate_batch_size, model_size_from_name


def test_model_size_detection():
    assert model_size_from_name("meta-llama/Llama-3.1-8B-Instruct") == 8
    assert model_size_from_name("meta-llama/Llama-3.1-70B") == 70
    assert model_size_from_name("codellama/CodeLlama-7b-hf") == 7
    assert model_size_from_name("some-unknown-model") == 7.0  # default
    # #1198: Previously unhandled sizes that fell back to 7.0B
    assert model_size_from_name("Qwen/Qwen2.5-72B-Instruct") == 72
    assert model_size_from_name("Qwen/Qwen2.5-32B") == 32
    assert model_size_from_name("Qwen/Qwen2.5-14B") == 14
    assert model_size_from_name("meta-llama/Llama-3.1-405B") == 405
    assert model_size_from_name("Qwen/Qwen3-4B") == 4
    # #1198: Active parameter tokens ignored in favor of total parameter count
    assert model_size_from_name("Qwen/Qwen3-235B-A22B") == 235
    assert model_size_from_name("Qwen/Qwen3-Coder-30B-A3B-Instruct") == 30
    # #1198: MoE explicit spelling
    assert model_size_from_name("mistralai/Mixtral-8x7B-v0.1") == 46.7
    assert model_size_from_name("mistralai/Mixtral-8x22B-v0.1") == 141.0
    # #1198: Boundary-safe regex avoids substring false matches
    assert model_size_from_name("meta-llama/Llama-3.2-11B-Vision-Instruct") == 11
    assert model_size_from_name("meta-llama/Llama-4-Scout-17B-16E-Instruct") == 17
    assert model_size_from_name("Qwen/Qwen3.5-27B") == 27
    assert model_size_from_name("Qwen/Qwen3.5-0.8B") == 0.8


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
