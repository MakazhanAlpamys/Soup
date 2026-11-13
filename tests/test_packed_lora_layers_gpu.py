"""The packed step against the stock Hugging Face forward and backward, on tiny random models.

GPU tests (``pytest -m gpu --no-cov``, Linux CUDA, WSL2 included). A 4-bit Qwen2 (or Qwen3)
with a peft LoRA is built twice over: the stock forward under autocast is the reference, the
packed step is ``PackedRun.forward``. What is compared is the loss and the gradient of every LoRA
parameter. The bars: loss 8e-5 relative (measured 0 to 5.6e-5 in bf16, 2e-6 in fp16); worst LoRA
gradient 3 % in bf16 and the exact mode (measured 1.0 to 2.0 % on these tiny models), 4 % with the
int8 backward (round to nearest; measured 1.5 to 2.4 %; stochastic rounding is noisier per step
by design and has its own bar), 0.6 % in fp16, whose compute type is 8x finer (measured 0.3 %,
1.5 % with int8). The bf16 numbers are those of the stock forward's own rounding as much as of
the kernels': the stock backward rounds its LoRA matmuls to bf16 in another order.

Also here: resident and offloaded activations are bit-identical, tables parked in pinned host
memory give the same step, a run can be closed and another started in the same process, and every
refusal of ``model.prepare`` names its reason.
"""

import gc
import tempfile

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("bitsandbytes")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.gpu

ALL = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TARGETS = {"all": ALL, "qv": ["q_proj", "v_proj"], "mlp": ["gate_proj", "up_proj", "down_proj"]}
LOSS_REL = 8e-5
BUDGET_MIB = 700


def _dtype(name):
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[name]


def build(dt, arch="qwen2", targets="all", tied=False, hidden=256, inter=512, heads=2, kv=1,
          layers=2, head_dim=None, vocab=1024, b_std=0.05, dropout=0.0, modules=None):
    """A random 4-bit (NF4, double quantisation) model with a peft LoRA, trainable like Soup's."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        Qwen2Config,
        Qwen2ForCausalLM,
        Qwen3Config,
        Qwen3ForCausalLM,
    )

    torch.manual_seed(0)
    config_cls, model_cls = (Qwen3Config, Qwen3ForCausalLM) if arch == "qwen3" else (
        Qwen2Config, Qwen2ForCausalLM)
    config = config_cls(vocab_size=vocab, hidden_size=hidden, intermediate_size=inter,
                        num_hidden_layers=layers, num_attention_heads=heads,
                        num_key_value_heads=kv, max_position_embeddings=512, rope_theta=10000.0,
                        tie_word_embeddings=tied, rms_norm_eps=1e-6)
    config.head_dim = head_dim or hidden // heads
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dt)
    with tempfile.TemporaryDirectory() as tmp:
        base = model_cls(config)
        if arch == "qwen3":  # random norm weights around 1 so that the q / k norms matter
            for name, param in base.named_parameters():
                if name.endswith("q_norm.weight") or name.endswith("k_norm.weight"):
                    param.data = 1.0 + 0.3 * torch.randn_like(param)
        base.to(dt).save_pretrained(tmp)
        del base
        model = AutoModelForCausalLM.from_pretrained(tmp, quantization_config=quant,
                                                     device_map={"": 0}, dtype=dt)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=dropout, target_modules=modules or TARGETS[targets],
        task_type="CAUSAL_LM"))
    gen = torch.Generator().manual_seed(7)
    for name, param in model.named_parameters():
        if "lora_B" in name:
            param.data = (torch.randn(param.shape, generator=gen) * b_std).to(param.device,
                                                                               param.dtype)
        elif "lora_A" in name:
            param.data = (torch.randn(param.shape, generator=gen) * 0.1).to(param.device,
                                                                            param.dtype)
    model.train()
    return model


def batch(rows=3, width=48, vocab=1024):
    gen = torch.Generator().manual_seed(11)
    lengths = [width, width - 9, width - 21][:rows] + [width - 5] * max(0, rows - 3)
    ids = torch.randint(3, vocab, (rows, width), generator=gen)
    mask = torch.zeros(rows, width, dtype=torch.long)
    labels = torch.full((rows, width), -100)
    for row, n in enumerate(lengths):
        mask[row, :n] = 1
        labels[row, :n] = ids[row, :n]
    return ids.cuda(), mask.cuda(), labels.cuda()


def lora_params(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def stock(model, dt, ids, mask, labels):
    """(loss, {name: grad}) of the stock forward and backward under autocast."""
    params = lora_params(model)
    for p in params.values():
        p.grad = None
    with torch.autocast("cuda", dtype=dt):
        out = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
    out.loss.backward()
    torch.cuda.synchronize()
    return out.loss.item(), {n: p.grad.detach().clone() for n, p in params.items()
                             if p.grad is not None}


def packed(model, run, ids, mask, labels):
    """(loss, {name: grad}) of the packed step of ``run``."""
    params = lora_params(model)
    for p in params.values():
        p.grad = None
    result = run.forward(ids, mask, labels, None)
    assert result is not None, "the batch was not right-padded"
    loss, _ = result
    loss.backward()
    torch.cuda.synchronize()
    return loss.item(), {n: p.grad.detach().clone() for n, p in params.items()
                         if p.grad is not None}


def worst_rel(ref, got):
    worst = 0.0
    for name, g_ref in ref.items():
        g = got[name]
        worst = max(worst, float((g.float() - g_ref.float()).norm() / g_ref.float().norm()))
    return worst


def make_run(model, dt, ids, **policy):
    from soup_cli.kernels.packed_lora.policy import Policy
    from soup_cli.kernels.packed_lora.runtime import PackedRun
    from soup_cli.kernels.packed_lora.support import PORTED_ARCHITECTURES

    lines = []
    run = PackedRun(Policy(**policy), log=lines.append)
    run.prepare(model, dt, ids.numel(), architectures=tuple(PORTED_ARCHITECTURES.values()))
    run.lines = lines
    return run


@pytest.fixture(autouse=True)
def _free_gpu():
    yield
    gc.collect()
    torch.cuda.empty_cache()


# --- parity with the stock forward and backward -------------------------------------------


@pytest.mark.parametrize("name,kwargs,int8,grad_bar", [
    ("bf16", {}, "off", 0.03),
    ("bf16", {}, "on", 0.04),
    ("bf16", {}, "stash", 0.03),
    ("fp16", {}, "off", 0.006),
    ("fp16", {}, "on", 0.025),
    ("bf16", {"tied": True}, "off", 0.03),
    ("bf16", {"targets": "qv"}, "off", 0.02),
    ("bf16", {"targets": "mlp"}, "on", 0.04),
    ("bf16", {"inter": 704, "hidden": 256}, "on", 0.04),
])
def test_qwen2_layers_match_the_stock_forward(name, kwargs, int8, grad_bar):
    dt = _dtype(name)
    model = build(dt, **kwargs)
    ids, mask, labels = batch()
    loss_ref, grads_ref = stock(model, dt, ids, mask, labels)
    run = make_run(model, dt, ids, int8_backward=int8, int8_rounding="nearest",
                   activations="offload")
    try:
        loss, grads = packed(model, run, ids, mask, labels)
        assert abs(loss - loss_ref) / abs(loss_ref) < LOSS_REL, (loss, loss_ref)
        assert set(grads) == set(grads_ref)
        worst = worst_rel(grads_ref, grads)
        print(f"[layers] {name} {kwargs} int8={int8}: loss rel "
              f"{abs(loss - loss_ref) / abs(loss_ref):.2e}, worst LoRA grad rel {worst:.4f}")
        assert worst < grad_bar, worst
        assert run.steps_packed == 1
    finally:
        run.close()


@pytest.mark.parametrize("name,int8,grad_bar", [("bf16", "off", 0.03), ("bf16", "on", 0.04),
                                                ("fp16", "off", 0.006)])
def test_qwen3_layers_match_the_stock_forward(name, int8, grad_bar):
    """Qwen3's per-head q / k norms; q_dim (512) differs from the hidden size (256)."""
    dt = _dtype(name)
    model = build(dt, arch="qwen3", hidden=256, heads=4, kv=2, head_dim=128, inter=512)
    ids, mask, labels = batch()
    loss_ref, grads_ref = stock(model, dt, ids, mask, labels)
    run = make_run(model, dt, ids, int8_backward=int8, int8_rounding="nearest",
                   activations="offload")
    try:
        loss, grads = packed(model, run, ids, mask, labels)
        assert abs(loss - loss_ref) / abs(loss_ref) < LOSS_REL, (loss, loss_ref)
        worst = worst_rel(grads_ref, grads)
        print(f"[qwen3] {name} int8={int8}: loss rel "
              f"{abs(loss - loss_ref) / abs(loss_ref):.2e}, worst LoRA grad rel {worst:.4f}")
        assert worst < grad_bar, worst
    finally:
        run.close()


def test_stochastic_rounding_keeps_the_gradients_close():
    """The default numerics: int8 backward with stochastic rounding of the gradients. Noisier
    per step than round-to-nearest (and unbiased), so the bar is looser."""
    dt = torch.bfloat16
    model = build(dt)
    ids, mask, labels = batch()
    loss_ref, grads_ref = stock(model, dt, ids, mask, labels)
    run = make_run(model, dt, ids)  # the defaults of training.packed_lora
    assert (run.policy.int8_backward, run.policy.int8_rounding) == ("auto", "grad")
    try:
        loss, grads = packed(model, run, ids, mask, labels)
        assert abs(loss - loss_ref) / abs(loss_ref) < LOSS_REL
        print(f"[stochastic] worst LoRA grad rel {worst_rel(grads_ref, grads):.4f}")
        assert worst_rel(grads_ref, grads) < 0.05, worst_rel(grads_ref, grads)
    finally:
        run.close()


def test_resident_and_offloaded_activations_are_bit_identical():
    dt = torch.bfloat16
    model = build(dt)
    ids, mask, labels = batch()
    results = {}
    for mode in ("resident", "offload"):
        run = make_run(model, dt, ids, activations=mode, int8_backward="on")
        try:
            results[mode] = packed(model, run, ids, mask, labels)
            assert run.packed.resident_layers == (len(run.packed.layers) if mode == "resident"
                                                  else 0)
        finally:
            run.close()
    (loss_a, grads_a), (loss_b, grads_b) = results["resident"], results["offload"]
    assert loss_a == loss_b
    assert all(torch.equal(grads_a[n], grads_b[n]) for n in grads_a)


def test_a_partial_plan_is_bit_identical_too():
    """Some layers resident, some offloaded (what a memory plan decides): same numbers."""
    from soup_cli.kernels.packed_lora.policy import Policy
    from soup_cli.kernels.packed_lora.runtime import PackedRun
    from soup_cli.kernels.packed_lora.support import PORTED_ARCHITECTURES

    class TwoResident(PackedRun):
        def stash_for_step(self, *args):
            stash = super().stash_for_step(*args)
            stash.plan_count(2)
            return stash

    dt = torch.bfloat16
    model = build(dt, layers=4)
    ids, mask, labels = batch()
    full = make_run(model, dt, ids, activations="resident")
    try:
        loss_a, grads_a = packed(model, full, ids, mask, labels)
    finally:
        full.close()
    run = TwoResident(Policy(activations="offload"))
    run.prepare(model, dt, ids.numel(), tuple(PORTED_ARCHITECTURES.values()))
    try:
        loss_b, grads_b = packed(model, run, ids, mask, labels)
        assert run.packed.resident_layers == 2
    finally:
        run.close()
    assert loss_a == loss_b and all(torch.equal(grads_a[n], grads_b[n]) for n in grads_a)


# --- the host-parked tables ---------------------------------------------------------------


@pytest.mark.parametrize("tied", [False, True])
def test_tables_in_pinned_host_memory_give_the_same_step(tied):
    from soup_cli.kernels.packed_lora.hostembed import HostTables

    dt = torch.bfloat16
    model = build(dt, tied=tied)
    ids, mask, labels = batch()
    run = make_run(model, dt, ids, activations="resident", int8_backward="off")
    try:
        loss_ref, grads_ref = packed(model, run, ids, mask, labels)
        table = run.packed.embed
        gpu_values = table.detach().clone()
        run.tables = HostTables(run.pool, tied)
        embed_p, head_p = run._tables_args()
        assert run.tables.park(embed_p, head_p)
        assert not table.is_cuda and run.tables.active_embed() and run.tables.active_head()
        loss, grads = packed(model, run, ids, mask, labels)
        assert abs(loss - loss_ref) / abs(loss_ref) < 1e-5
        # the head's sums run slab by slab, and its bf16 rounding differs with the order
        assert worst_rel(grads_ref, grads) < 0.03, worst_rel(grads_ref, grads)
        run.restore_tables()
        assert table.is_cuda and torch.equal(table.detach(), gpu_values)
        assert not run.tables.any_active()
        assert run.pool.locked_bytes() > 0
    finally:
        run.close()
    assert run.pool.locked_bytes() == 0


# --- the stock forward keeps working ------------------------------------------------------


def test_the_stock_forward_after_prepare_equals_the_stock_forward_before():
    """prepare() keeps the tables in the compute dtype; the stock forward then needs the
    embedding as fp32 (embed_for_stock_forward) to keep its fp32 residual stream."""
    from soup_cli.kernels.packed_lora import model as packed_model

    dt = torch.bfloat16
    model = build(dt)
    ids, mask, labels = batch()
    with torch.no_grad(), torch.autocast("cuda", dtype=dt):
        before = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False).loss
    run = make_run(model, dt, ids)
    try:
        causal = packed_model.unwrap(model)
        assert getattr(causal, "_packed_lora_fp32_embeds", False)
        with torch.no_grad(), torch.autocast("cuda", dtype=dt):
            after = model(inputs_embeds=packed_model.embed_for_stock_forward(causal, ids),
                          attention_mask=mask, labels=labels, use_cache=False).loss
        assert abs(before.item() - after.item()) / abs(before.item()) < 1e-6
    finally:
        run.close()


def test_keep_tables_in_compute_dtype_leaves_the_tables_unconverted():
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Qwen2Config, Qwen2ForCausalLM

    from soup_cli.kernels.packed_lora import model as packed_model

    dt = torch.bfloat16
    config = Qwen2Config(vocab_size=1024, hidden_size=256, intermediate_size=512,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dt)
    with tempfile.TemporaryDirectory() as tmp:
        Qwen2ForCausalLM(config).to(dt).save_pretrained(tmp)
        model = AutoModelForCausalLM.from_pretrained(tmp, quantization_config=quant,
                                                     device_map={"": 0}, dtype=dt)
    with packed_model.keep_tables_in_compute_dtype(model):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    assert model.get_input_embeddings().weight.dtype == dt
    assert model.get_input_embeddings().weight.is_cuda
    assert model._packed_lora_fp32_embeds is True
    # the norms are upcast as before
    assert model.model.norm.weight.dtype == torch.float32


# --- the run object -----------------------------------------------------------------------


def test_a_run_can_be_closed_and_another_started_in_the_same_process():
    dt = torch.bfloat16
    model = build(dt)
    ids, mask, labels = batch()
    first = make_run(model, dt, ids, activations="offload")
    loss_a, grads_a = packed(model, first, ids, mask, labels)
    pinned = first.pool.locked_bytes()
    assert pinned > 0  # the offloaded layers' buffers are page-locked
    first.pool.alloc(1 << 20)  # a buffer nobody owns: close() unlocks everything the run locked
    assert first.pool.locked_bytes() == pinned + (1 << 20)
    first.close()
    first.close()  # twice is fine
    assert first.pool.locked_bytes() == 0 and first.stash is None and first.packed is None
    second = make_run(model, dt, ids, activations="offload")
    try:
        loss_b, grads_b = packed(model, second, ids, mask, labels)
    finally:
        second.close()
    assert loss_a == loss_b
    assert all(torch.equal(grads_a[n], grads_b[n]) for n in grads_a)  # the same seeds, rerun


def test_the_run_counts_steps_and_names_a_stock_step_once():
    dt = torch.bfloat16
    model = build(dt)
    ids, mask, labels = batch()
    run = make_run(model, dt, ids, activations="resident")
    try:
        packed(model, run, ids, mask, labels)
        left_padded = mask.clone()
        left_padded[1] = left_padded[1].flip(0)
        assert run.forward(ids, left_padded, labels, None) is None
        run.note_stock("a batch that is not right-padded")
        run.note_stock("a batch that is not right-padded")
        assert (run.steps_packed, run.steps_stock) == (1, 2)
        assert run.summary() == ("packed_lora: 1 packed, 2 stock steps "
                                 "(2x a batch that is not right-padded)")
        assert sum("stock forward for a training step" in line for line in run.lines) == 1
    finally:
        run.close()


def test_the_start_up_checks_are_logged_and_decide_the_int8_path():
    dt = torch.bfloat16
    model = build(dt, inter=1024)
    ids, mask, labels = batch()
    run = make_run(model, dt, ids)
    try:
        assert run.int8.decided is True, run.lines
        assert run.int8.check["cosine"] > 0.998
        text = "\n".join(run.lines)
        assert "packed forward: Qwen2" in text and "int8_backward=auto -> on" in text
        assert "stochastic rounding" in text
    finally:
        run.close()


def test_a_budget_run_makes_a_plan_that_covers_the_measured_peak():
    """A mid-size random model (Qwen2.5-0.5B's layer shapes, 6 layers): torch's peak allocated
    memory for the step stays under the budget, and the planner's estimate is within 10 % of it
    (the fits were made on the 1.5B and 7B shapes and read 0.4-4 % high there; here 3 % low, which
    the 2 % + 64 MiB margin of the limit covers)."""
    dt = torch.bfloat16
    model = build(dt, hidden=896, inter=4864, heads=14, kv=2, layers=6, vocab=32000)
    ids, mask, labels = batch(rows=4, width=128, vocab=32000)
    run = make_run(model, dt, ids, gpu_budget_bytes=BUDGET_MIB << 20, int8_backward="on",
                   int8_rounding="nearest")
    try:
        assert run.plan is not None and "plan: weights resident" in "\n".join(run.lines)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        packed(model, run, ids, mask, labels)
        peak = torch.cuda.max_memory_allocated()
        print(f"plan estimate {run.plan.est_peak / 2**20:.0f} MiB, measured {peak / 2**20:.0f} MiB")
        assert peak <= BUDGET_MIB << 20, (peak / 2**20, BUDGET_MIB)  # the target, within the margin
        assert abs(run.plan.est_peak - peak) <= 0.1 * peak, (peak / 2**20,
                                                             run.plan.est_peak / 2**20)
    finally:
        run.close()


def test_a_budget_below_the_floor_is_refused_at_prepare():
    from soup_cli.kernels.packed_lora.budget import BudgetTooSmallError
    from soup_cli.kernels.packed_lora.policy import Policy
    from soup_cli.kernels.packed_lora.runtime import PackedRun

    dt = torch.bfloat16
    model = build(dt, hidden=896, inter=4864, heads=14, kv=2, layers=6, vocab=32000)
    ids, mask, labels = batch(rows=4, width=128, vocab=32000)
    run = PackedRun(Policy(gpu_budget_bytes=64 << 20))
    try:
        with pytest.raises(BudgetTooSmallError) as excinfo:
            run.prepare(model, dt, ids.numel())
        assert excinfo.value.minimum_mib > 64
    finally:
        run.close()


# --- what prepare refuses -----------------------------------------------------------------


def _prepare_error(model, dt, **kwargs):
    from soup_cli.kernels.packed_lora import model as packed_model

    with pytest.raises(packed_model.UnsupportedError) as excinfo:
        packed_model.prepare(model, dt, **kwargs)
    return str(excinfo.value)


def test_lora_dropout_is_refused():
    dt = torch.bfloat16
    message = _prepare_error(build(dt, dropout=0.05), dt)
    assert "LoRA dropout 0.05 (the kernels need lora.dropout: 0)" in message


def test_qwen3_is_refused_unless_the_gate_is_opened():
    dt = torch.bfloat16
    model = build(dt, arch="qwen3", hidden=256, heads=4, kv=2, head_dim=128)
    message = _prepare_error(model, dt)
    assert message.startswith("architecture Qwen3ForCausalLM (supported: Qwen2ForCausalLM)")


def test_a_lora_on_another_module_is_refused():
    dt = torch.bfloat16
    model = build(dt, modules=ALL + ["lm_head"])
    assert "a LoRA on " in _prepare_error(model, dt) and "lm_head" in _prepare_error(model, dt)


def test_another_compute_dtype_is_refused():
    model = build(torch.bfloat16)
    message = _prepare_error(model, torch.float16)
    assert "4-bit weights not quantized for compute dtype torch.float16" in message


def test_a_trainable_table_is_refused():
    dt = torch.bfloat16
    model = build(dt)
    model.get_input_embeddings().weight.requires_grad_(True)
    assert "a trainable embedding or lm_head" in _prepare_error(model, dt)


# --- the end of a run ---------------------------------------------


def test_close_puts_parked_tables_back_on_the_gpu_and_releases_the_host_copies():
    from soup_cli.kernels.packed_lora.hostembed import HostTables

    dt = torch.bfloat16
    model = build(dt)
    ids, mask, labels = batch()
    run = make_run(model, dt, ids, activations="resident")
    table = run.packed.embed
    run.tables = HostTables(run.pool, run.packed.tied)
    embed_p, head_p = run._tables_args()
    assert run.tables.park(embed_p, head_p) and not table.is_cuda
    run.close()
    assert table.is_cuda and run.pool.locked_bytes() == 0 and run.tables is None
    with torch.no_grad(), torch.autocast("cuda", dtype=dt):  # the model is usable again
        model(inputs_embeds=table[ids].float(), attention_mask=mask, use_cache=False)
