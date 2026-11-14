"""Smoke tests on a real checkpoint (``pytest -m "smoke and gpu" --no-cov``; no CI run has one).

Defaults to ``Qwen/Qwen2.5-1.5B-Instruct`` from the local Hugging Face cache; skipped when it is
not there. ``SOUP_PACKED_LORA_SMOKE_MODEL`` picks another Qwen2 checkpoint
(``Qwen/Qwen2.5-7B-Instruct`` for the 7000 MiB plan of the 7B, which needs an 8 GB card with
nothing else on it) and ``SOUP_PACKED_LORA_SMOKE_BUDGET_MIB`` the budget of the second test.
"""

import gc
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("bitsandbytes")
pytest.importorskip("peft")
pytest.importorskip("transformers")

pytestmark = [pytest.mark.gpu, pytest.mark.smoke]

MODEL = os.environ.get("SOUP_PACKED_LORA_SMOKE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
BUDGET_MIB = int(os.environ.get("SOUP_PACKED_LORA_SMOKE_BUDGET_MIB", "3000"))


def _load(dt):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    try:
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dt)
        model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant,
                                                     device_map={"": 0}, dtype=dt,
                                                     local_files_only=True)
    except OSError as exc:
        pytest.skip(f"{MODEL} is not in the local Hugging Face cache: {exc}")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                        "down_proj"]))
    gen = torch.Generator().manual_seed(7)
    for name, param in model.named_parameters():
        if "lora_B" in name:  # a LoRA that has trained a little, so that every branch matters
            param.data = (torch.randn(param.shape, generator=gen) * 0.002).to(param.device,
                                                                               param.dtype)
    model.train()
    return model, tok


def _rows(tok, n_rows, width):
    text = ("Soup fine-tunes language models from the command line. " * 40
            + "Бұл жол қазақ тіліндегі мәтінді тексеруге арналған. " * 40)
    ids = tok(text, return_tensors="pt").input_ids[0]
    rows = [ids[i * 17:i * 17 + width - (i * 11) % 40] for i in range(n_rows)]
    out = torch.full((n_rows, width), tok.pad_token_id or 0)
    mask = torch.zeros(n_rows, width, dtype=torch.long)
    for i, row in enumerate(rows):
        out[i, :len(row)] = row
        mask[i, :len(row)] = 1
    labels = out.masked_fill(mask == 0, -100)
    return out.cuda(), mask.cuda(), labels.cuda()


def _grad_vector(model):
    return torch.cat([p.grad.detach().float().flatten() for p in model.parameters()
                      if p.requires_grad and p.grad is not None])


@pytest.fixture(autouse=True)
def _free_gpu():
    yield
    gc.collect()
    torch.cuda.empty_cache()


def test_a_real_checkpoint_matches_the_stock_step():
    """Loss within 0.5 % of the stock step's (measured 0.03 to 0.27 %: bf16 matmuls in another
    order), and the LoRA gradient vector within a cosine of 0.998 (exact mode; measured 0.9990)
    or 0.99 (the default int8 backward with stochastic rounding; measured 0.9941 to 0.9946)."""
    from soup_cli.kernels.packed_lora.policy import Policy
    from soup_cli.kernels.packed_lora.runtime import PackedRun

    dt = torch.bfloat16
    model, tok = _load(dt)
    ids, mask, labels = _rows(tok, 4, 96)
    for p in model.parameters():
        p.grad = None
    with torch.autocast("cuda", dtype=dt):
        ref_loss = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False).loss
    ref_loss.backward()
    ref_grad = _grad_vector(model)
    for policy, bar in ((Policy(int8_backward="off"), 0.998), (Policy(), 0.99)):
        run = PackedRun(policy, log=lambda _: None)
        run.prepare(model, dt, ids.numel())
        try:
            for p in model.parameters():
                p.grad = None
            loss, _ = run.forward(ids, mask, labels, None)
            loss.backward()
            grad = _grad_vector(model)
            cosine = float(torch.nn.functional.cosine_similarity(grad, ref_grad, dim=0))
            rel = abs(loss.item() - ref_loss.item()) / abs(ref_loss.item())
            print(f"[smoke] {MODEL} int8_backward={policy.int8_backward}: loss rel {rel:.2e}, "
                  f"gradient cosine {cosine:.5f}")
            assert rel < 5e-3 and cosine > bar
        finally:
            run.close()


def test_a_real_checkpoint_stays_under_its_budget():
    """The plan's promise: torch's peak allocated memory for a step of 8 x 512 tokens does not
    exceed the budget, and the estimate is within 10 % of it (7B at 7000 MiB: est 6707, measured
    6515 in the challenge runs)."""
    from soup_cli.kernels.packed_lora.budget import BudgetTooSmallError
    from soup_cli.kernels.packed_lora.policy import Policy
    from soup_cli.kernels.packed_lora.runtime import PackedRun

    dt = torch.bfloat16
    model, tok = _load(dt)
    ids, mask, labels = _rows(tok, 8, 512)
    run = PackedRun(Policy(gpu_budget_bytes=BUDGET_MIB << 20), log=print)
    try:
        run.prepare(model, dt, ids.numel())
    except BudgetTooSmallError as exc:
        run.close()
        pytest.skip(str(exc))
    try:
        for p in model.parameters():
            p.grad = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            loss, _ = run.forward(ids, mask, labels, None)
            loss.backward()
            torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        print(f"[smoke] {MODEL} budget {BUDGET_MIB} MiB: estimate {run.plan.est_peak / 2**20:.0f}"
              f" MiB, measured {peak / 2**20:.0f} MiB, {run.plan.k}/{len(run.packed.layers)} "
              f"layers resident")
        assert peak <= BUDGET_MIB << 20
        assert abs(run.plan.est_peak - peak) <= 0.1 * peak
        assert torch.isfinite(loss).item()
    finally:
        run.close()
