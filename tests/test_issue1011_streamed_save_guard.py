"""#1011 — a streamed LoRA save that holds nothing must raise, not warn.

Twice a streamed run reported success and wrote an ``adapter_model.safetensors``
that reloads as nothing: v0.72.0's ``.inner.`` keys, and #1005's 0-tensor file
under peft 0.21. #1010 removed the second cause; these tests pin the check that
makes the whole class loud at save time, on the production save path (the final
``save_model`` and every ``checkpoint-*``), without a GPU.

The reference is the model's own trainable ``lora_*`` parameter count, so the
check holds on peft 0.20 and 0.21 alike.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

STREAMABLE_TRAINERS = ("sft", "dpo", "orpo", "simpo", "kto")


def _requires_train_extra():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("safetensors")


# --------------------------------------------------------------------------
# fixtures (standalone, mirroring tests/test_issue1005_peft021_canonical_names.py)
# --------------------------------------------------------------------------


def _tiny_llama_dir(tmp_path):
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = tmp_path / "model"
    weights.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))
    return str(weights)


@pytest.fixture
def streamed(tmp_path):
    _requires_train_extra()
    from peft import LoraConfig, TaskType

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    weights = _tiny_llama_dir(tmp_path)
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(weights, shards, dtype="float32", arch="llama")
    model, runtime = build_streamed_model(
        model_id=weights,
        shard_dir=shards,
        index=index,
        lora_config=LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
        ),
        device="cpu",
        dtype="float32",
        buffers=2,
        pin=False,
        seed=3,
    )
    try:
        yield model
    finally:
        runtime.close()


def _trainable_lora_count(model):
    return sum(1 for n, p in model.named_parameters() if p.requires_grad and "lora_" in n)


def _write_adapter(path, keys):
    import torch
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    save_file({k: torch.zeros(1) for k in keys}, str(path / "adapter_model.safetensors"))


# --------------------------------------------------------------------------
# the check, on a real streamed save
# --------------------------------------------------------------------------


def test_a_normal_streamed_save_passes(streamed, tmp_path):
    from soup_cli.utils.layer_stream_runtime import assert_streamed_adapter_saved

    out = tmp_path / "adapter"
    streamed.save_pretrained(str(out))

    # 2 layers x (q_proj, v_proj) x (lora_A, lora_B)
    assert _trainable_lora_count(streamed) == 8
    assert_streamed_adapter_saved(streamed, str(out))


def test_the_1005_divergence_reintroduced_raises_instead_of_saving_silently(
    streamed, tmp_path, monkeypatch
):
    """Mutation evidence: undo #1010's canonical names on the wrapper.

    Under peft 0.21 the structural adapter selection then disagrees with the
    canonical state_dict keys and ``save_pretrained`` writes an empty file, which
    is #1005. Under peft 0.20 the save survives the mutation, which is why the
    count reference, not a version table, decides.
    """
    import torch

    from soup_cli.utils.layer_stream_runtime import assert_streamed_adapter_saved

    layers = streamed.get_submodule("base_model.model.model.layers")
    wrapper_cls = type(layers[0])
    assert wrapper_cls.__name__ == "StreamedDecoderLayer"
    monkeypatch.setattr(wrapper_cls, "named_modules", torch.nn.Module.named_modules)
    monkeypatch.setattr(wrapper_cls, "named_children", torch.nn.Module.named_children)

    out = tmp_path / "adapter"
    streamed.save_pretrained(str(out))

    from safetensors import safe_open

    with safe_open(str(out / "adapter_model.safetensors"), framework="pt") as handle:
        saved = len(list(handle.keys()))
    if saved == _trainable_lora_count(streamed):
        pytest.skip("this peft still selects adapter keys by substring; the mutation is inert")
    with pytest.raises(
        RuntimeError, match=r"holds 0 LoRA tensors, but the streamed model trained 8"
    ):
        assert_streamed_adapter_saved(streamed, str(out))


# --------------------------------------------------------------------------
# each failure shape, on a hand-written file
# --------------------------------------------------------------------------


def test_wrapper_segment_keys_raise(streamed, tmp_path):
    from soup_cli.utils.layer_stream_runtime import assert_streamed_adapter_saved

    keys = [
        f"base_model.model.model.layers.{i}.inner.self_attn.{m}.lora_{ab}.weight"
        for i in range(2)
        for m in ("q_proj", "v_proj")
        for ab in ("A", "B")
    ]
    _write_adapter(tmp_path / "adapter", keys)
    with pytest.raises(RuntimeError, match=r"'\.inner\.' segment in 8 keys"):
        assert_streamed_adapter_saved(streamed, str(tmp_path / "adapter"))


def test_a_partial_adapter_raises(streamed, tmp_path):
    from soup_cli.utils.layer_stream_runtime import assert_streamed_adapter_saved

    keys = [
        f"base_model.model.model.layers.0.self_attn.{m}.lora_{ab}.weight"
        for m in ("q_proj", "v_proj")
        for ab in ("A", "B")
    ]
    _write_adapter(tmp_path / "adapter", keys)
    with pytest.raises(
        RuntimeError, match=r"holds 4 LoRA tensors, but the streamed model trained 8"
    ):
        assert_streamed_adapter_saved(streamed, str(tmp_path / "adapter"))


def test_a_missing_adapter_file_raises(streamed, tmp_path):
    from soup_cli.utils.layer_stream_runtime import assert_streamed_adapter_saved

    (tmp_path / "adapter").mkdir()
    with pytest.raises(RuntimeError, match=r"saved no adapter_model\.safetensors .* expected 8"):
        assert_streamed_adapter_saved(streamed, str(tmp_path / "adapter"))


# --------------------------------------------------------------------------
# reachable from the production save path
# --------------------------------------------------------------------------


def test_the_checkpoint_callback_checks_the_checkpoint_it_names(streamed, tmp_path):
    from soup_cli.utils.layer_stream_runtime import build_streamed_save_guard_callback

    callback = build_streamed_save_guard_callback()
    (tmp_path / "checkpoint-5").mkdir()
    state = SimpleNamespace(global_step=5)

    # a rank that wrote nothing is left alone
    callback.on_save(
        SimpleNamespace(output_dir=str(tmp_path), should_save=False), state, None, model=streamed
    )
    with pytest.raises(RuntimeError, match="checkpoint-5"):
        callback.on_save(
            SimpleNamespace(output_dir=str(tmp_path), should_save=True), state, None, model=streamed
        )

    streamed.save_pretrained(str(tmp_path / "checkpoint-5"))
    callback.on_save(
        SimpleNamespace(output_dir=str(tmp_path), should_save=True), state, None, model=streamed
    )


def test_the_mixin_checks_only_streamed_runs(streamed, tmp_path):
    from soup_cli.trainer.stream_setup import StreamingSetupMixin

    class _Wrapper(StreamingSetupMixin):
        pass

    wrapper = _Wrapper()
    wrapper.trainer = SimpleNamespace(
        model=streamed, args=SimpleNamespace(should_save=True), add_callback=lambda cb: None
    )
    (tmp_path / "empty").mkdir()

    # a resident run has no stream runtime: nothing to check
    wrapper._assert_streamed_adapter_saved(str(tmp_path / "empty"))

    wrapper._stream_runtime = object()
    with pytest.raises(RuntimeError, match="saved no adapter_model"):
        wrapper._assert_streamed_adapter_saved(str(tmp_path / "empty"))

    added = []
    wrapper.trainer.add_callback = added.append
    wrapper._attach_streamed_save_guard()
    assert [type(cb).__name__ for cb in added] == ["StreamedSaveGuardCallback"]


@pytest.mark.parametrize("module", STREAMABLE_TRAINERS)
def test_every_streamable_trainer_checks_its_saves(module):
    """The final save calls the check right after ``save_model``, and the
    checkpoint callback is attached before ``train()`` runs."""
    source = Path(__file__).resolve().parents[1] / "src" / "soup_cli" / "trainer" / f"{module}.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "_attach_streamed_save_guard" in calls
    assert "_assert_streamed_adapter_saved" in calls

    text = source.read_text(encoding="utf-8")
    save = text.index("self.trainer.save_model(self._output_dir)")
    check = text.index("self._assert_streamed_adapter_saved(self._output_dir)")
    attach = text.index("self._attach_streamed_save_guard()")
    train = text.index("self.trainer.train(")
    assert save < check and attach < train


def test_the_streamable_task_list_matches_the_schema():
    """If streaming is opened to another task, its trainer needs the check too."""
    from soup_cli.utils.layer_stream import SUPPORTED_STREAM_TASKS

    # the tasks the schema lets stream are exactly the trainers checked above
    assert set(SUPPORTED_STREAM_TASKS) == set(STREAMABLE_TRAINERS)

    trainer_dir = Path(__file__).resolve().parents[1] / "src" / "soup_cli" / "trainer"
    trainers = {
        path.stem
        for path in trainer_dir.glob("*.py")
        if "_training_context(" in path.read_text(encoding="utf-8")
        and path.stem != "stream_setup"
    }
    assert trainers == set(STREAMABLE_TRAINERS)


@pytest.mark.parametrize("task", STREAMABLE_TRAINERS)
def test_a_real_streamed_train_checks_every_checkpoint_and_the_final_save(
    tmp_path, monkeypatch, task
):
    """The wiring, by behaviour rather than by source text: a real
    ``wrapper.train()`` with ``save_steps=1`` runs the check on
    ``checkpoint-1``, ``checkpoint-2`` and the final output directory."""
    _requires_train_extra()
    from tests.test_v07204 import _build_streamed_wrapper

    try:
        from transformers.trainer_utils import SaveStrategy
    except ImportError:  # transformers < 4.46
        from transformers.trainer_utils import IntervalStrategy as SaveStrategy

    from soup_cli.utils import layer_stream_runtime

    wrapper, _, _ = _build_streamed_wrapper(tmp_path, monkeypatch, task=task)
    args = wrapper.trainer.args
    args.max_steps = 2
    args.save_steps = 1
    args.save_strategy = SaveStrategy.STEPS

    checked = []
    real_check = layer_stream_runtime.assert_streamed_adapter_saved

    def spy(model, output_dir):
        checked.append(Path(output_dir).resolve())
        return real_check(model, output_dir)

    monkeypatch.setattr(layer_stream_runtime, "assert_streamed_adapter_saved", spy)
    wrapper.train()

    out = Path(wrapper._output_dir).resolve()
    assert checked == [out / "checkpoint-1", out / "checkpoint-2", out]
