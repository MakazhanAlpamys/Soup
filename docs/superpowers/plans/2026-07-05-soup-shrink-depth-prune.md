# `soup shrink` (depth-prune + distill-heal) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Ship `soup shrink` — rank a model's layers by residual-stream angular distance, drop the least-important contiguous block, optionally distill-heal, and emit a single dense smaller model with a before/after perplexity verdict.

**Architecture:** New `utils/shrink.py` (pure verdict half + torch-lazy prune/importance half) + `commands/shrink.py` (Typer orchestration: load → scan → prune → save → reload+ppl → optional subprocess distill heal → fuse → verdict → registry attach). Reuses `live_eval.load_model_and_tokenizer`/`compute_eval_loss`, mirrors `ship_verdict`'s dataclass+panel shape and `ra_dit_run._run_train_subprocess`.

**Tech Stack:** Python 3.10+, Typer, Rich, Pydantic v2, torch/transformers (lazy), pytest.

## Global Constraints

- Line length 100 (ruff E,F,I,N,W). No bare `print()` — Rich `Console`.
- Heavy deps (torch/transformers/peft) lazy-imported inside functions.
- Path containment via `os.path.realpath` + `os.path.commonpath` (NOT `Path.resolve().relative_to()`).
- Verdict half of `utils/shrink.py` has **no top-level torch/transformers import** (assert with a test).
- Exit codes: 0 = SHIP, 2 = DON'T SHIP, 1 = runtime error (mirror `soup ship`/`diagnose`).
- Arch allowlist v1: Llama / Qwen / SmolLM (`model.model.layers` + `config.num_hidden_layers`).
- Off-by-one invariant: `len(hidden_states) == num_layers + 1`; block `[L, L+n)` input `hidden_states[L]`, output `hidden_states[L+n]`.
- Position bound: dropped block ⊆ `[1, num_layers-2]` (protect first + last layer).
- Reload-before-measure: pruned/final ppl measured on the reloaded saved dir, not the in-memory sliced module.
- Tests go in `tests/test_v07129.py`; assert specific messages, use `assert result.exit_code == 0, (result.output, repr(result.exception))`.

---

### Task 1: Pure verdict half — `LayerImportance`, `ShrinkVerdict`, `decide_shrink`, render + serialize

**Files:**
- Create: `src/soup_cli/utils/shrink.py` (verdict half only this task)
- Test: `tests/test_v07129.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) LayerImportance(start:int, block_size:int, angular_distance:float)`
  - `@dataclass(frozen=True) ShrinkVerdict(decision, ppl_original, ppl_final, ppl_ratio, tolerance, layers_before, layers_after, params_saved_pct, healed, soup_version)`
  - `DECISION_SHIP="SHIP"`, `DECISION_DONT_SHIP="DON'T SHIP"`, `DEFAULT_TOLERANCE=0.10`, `MAX_TOLERANCE=5.0`
  - `decide_shrink(ppl_original, ppl_final, *, tolerance=DEFAULT_TOLERANCE, layers_before, layers_after, params_saved_pct=0.0, healed=False, soup_version=__version__) -> ShrinkVerdict`
  - `render_shrink_panel(verdict) -> Panel`
  - `shrink_verdict_to_dict(verdict) -> dict`

- [ ] **Step 1: Write failing tests** (append to `tests/test_v07129.py`)

```python
import math
import pytest
from soup_cli.utils.shrink import (
    DECISION_SHIP, DECISION_DONT_SHIP, DEFAULT_TOLERANCE,
    LayerImportance, ShrinkVerdict, decide_shrink,
    render_shrink_panel, shrink_verdict_to_dict,
)

class TestDecideShrink:
    def test_within_tolerance_ships(self):
        v = decide_shrink(10.0, 10.5, tolerance=0.10, layers_before=30, layers_after=24)
        assert v.decision == DECISION_SHIP
        assert math.isclose(v.ppl_ratio, 1.05)

    def test_exceeds_tolerance_dont_ship(self):
        v = decide_shrink(10.0, 12.0, tolerance=0.10, layers_before=30, layers_after=24)
        assert v.decision == DECISION_DONT_SHIP

    def test_boundary_exactly_at_tolerance_ships(self):
        v = decide_shrink(10.0, 11.0, tolerance=0.10, layers_before=30, layers_after=24)
        assert v.decision == DECISION_SHIP  # ratio-1 == tolerance -> <=, SHIP

    def test_improved_ppl_ships(self):
        v = decide_shrink(10.0, 9.5, tolerance=0.10, layers_before=30, layers_after=24)
        assert v.decision == DECISION_SHIP

    def test_rejects_nonpositive_ppl(self):
        with pytest.raises(ValueError):
            decide_shrink(0.0, 5.0, layers_before=30, layers_after=24)

    def test_rejects_nonfinite(self):
        with pytest.raises(ValueError):
            decide_shrink(10.0, float("inf"), layers_before=30, layers_after=24)

    def test_rejects_bad_tolerance(self):
        with pytest.raises(ValueError):
            decide_shrink(10.0, 10.0, tolerance=-0.1, layers_before=30, layers_after=24)
        with pytest.raises(ValueError):
            decide_shrink(10.0, 10.0, tolerance=6.0, layers_before=30, layers_after=24)

    def test_frozen(self):
        v = decide_shrink(10.0, 10.5, layers_before=30, layers_after=24)
        with pytest.raises(Exception):
            v.decision = "x"

    def test_to_dict_roundtrip(self):
        v = decide_shrink(10.0, 10.5, layers_before=30, layers_after=24, healed=True)
        d = shrink_verdict_to_dict(v)
        assert d["decision"] == v.decision
        assert d["healed"] is True
        assert set(d) >= {"decision","ppl_original","ppl_final","ppl_ratio","tolerance",
                          "layers_before","layers_after","params_saved_pct","healed","soup_version"}

    def test_render_panel_names_decision(self):
        from rich.console import Console
        from io import StringIO
        v = decide_shrink(10.0, 12.0, layers_before=30, layers_after=24)
        buf = StringIO(); Console(file=buf, width=100).print(render_shrink_panel(v))
        assert "DON'T SHIP" in buf.getvalue()

class TestNoTopLevelTorch:
    def test_shrink_module_imports_without_torch(self):
        import ast, pathlib
        src = pathlib.Path("src/soup_cli/utils/shrink.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = []
        for n in top:
            if isinstance(n, ast.Import):
                names += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                names.append(n.module or "")
        assert not any(m.split(".")[0] in {"torch","transformers","peft"} for m in names)
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError: soup_cli.utils.shrink`)

Run: `pytest tests/test_v07129.py -x -q --no-cov`

- [ ] **Step 3: Implement verdict half of `utils/shrink.py`**

```python
"""soup shrink — depth-prune + distill-heal (v0.71.29, arXiv:2403.17887).

Verdict half is PURE (no top-level torch); the prune/importance half lazy-
imports torch inside functions.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from rich.panel import Panel

from soup_cli import __version__

DECISION_SHIP = "SHIP"
DECISION_DONT_SHIP = "DON'T SHIP"
DEFAULT_TOLERANCE = 0.10
MAX_TOLERANCE = 5.0


@dataclass(frozen=True)
class LayerImportance:
    start: int
    block_size: int
    angular_distance: float


@dataclass(frozen=True)
class ShrinkVerdict:
    decision: str
    ppl_original: float
    ppl_final: float
    ppl_ratio: float
    tolerance: float
    layers_before: int
    layers_after: int
    params_saved_pct: float
    healed: bool
    soup_version: str


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return out


def decide_shrink(
    ppl_original: object,
    ppl_final: object,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    layers_before: int,
    layers_after: int,
    params_saved_pct: float = 0.0,
    healed: bool = False,
    soup_version: str = __version__,
) -> ShrinkVerdict:
    """SHIP iff ppl_final/ppl_original - 1 <= tolerance."""
    orig = _finite_positive(ppl_original, "ppl_original")
    final = _finite_positive(ppl_final, "ppl_final")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError("tolerance must be a number")
    tol = float(tolerance)
    if not math.isfinite(tol) or not (0.0 <= tol <= MAX_TOLERANCE):
        raise ValueError(f"tolerance must be in [0.0, {MAX_TOLERANCE}]")
    ratio = final / orig
    decision = DECISION_SHIP if (ratio - 1.0) <= tol + 1e-9 else DECISION_DONT_SHIP
    return ShrinkVerdict(
        decision=decision,
        ppl_original=round(orig, 4),
        ppl_final=round(final, 4),
        ppl_ratio=round(ratio, 4),
        tolerance=tol,
        layers_before=int(layers_before),
        layers_after=int(layers_after),
        params_saved_pct=round(float(params_saved_pct), 2),
        healed=bool(healed),
        soup_version=str(soup_version),
    )


def shrink_verdict_to_dict(verdict: ShrinkVerdict) -> dict:
    return asdict(verdict)


def render_shrink_panel(verdict: ShrinkVerdict) -> Panel:
    color = "green" if verdict.decision == DECISION_SHIP else "red"
    body = (
        f"[bold]{verdict.decision}[/]\n\n"
        f"Layers: {verdict.layers_before} -> {verdict.layers_after}  "
        f"(params saved {verdict.params_saved_pct:.1f}%)\n"
        f"Perplexity: {verdict.ppl_original:.3f} -> {verdict.ppl_final:.3f}  "
        f"(x{verdict.ppl_ratio:.3f}, tol {verdict.tolerance:.0%})\n"
        f"Healed: {'yes' if verdict.healed else 'no'}"
    )
    return Panel(body, title="soup shrink", border_style=color)
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/test_v07129.py -x -q --no-cov`

- [ ] **Step 5: Commit** — `git add src/soup_cli/utils/shrink.py tests/test_v07129.py && git commit -m "feat(shrink): pure verdict half (decide_shrink) (v0.71.29)"`

---

### Task 2: Arch allowlist + `prune_model_layers`

**Files:** Modify `src/soup_cli/utils/shrink.py`; Test `tests/test_v07129.py`

**Interfaces:**
- Produces:
  - `SUPPORTED_SHRINK_ARCHS = ("llama","qwen","qwen2","qwen3","smollm")` (or a regex helper)
  - `shrink_arch_of(model) -> str` (returns family name; raises ValueError with the supported list on unsupported)
  - `layer_list(model) -> nn.ModuleList` (returns `model.model.layers`, arch-guarded)
  - `prune_model_layers(model, start:int, block_size:int) -> None` — mutates: deletes layers `[start, start+block_size)`, patches `config.num_hidden_layers`. Raises on out-of-position (must not touch layer 0 or last).

- [ ] **Step 1: Write failing tests** (use a tiny real config, e.g. build a 6-layer `LlamaConfig` model on CPU)

```python
class TestPrune:
    def _tiny(self, layers=6):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
                          num_attention_heads=4, num_key_value_heads=4, vocab_size=128,
                          max_position_embeddings=64)
        return LlamaForCausalLM(cfg)

    def test_arch_detected(self):
        from soup_cli.utils.shrink import shrink_arch_of
        assert shrink_arch_of(self._tiny()) == "llama"

    def test_prune_removes_block_and_patches_config(self):
        from soup_cli.utils.shrink import prune_model_layers
        m = self._tiny(6)
        prune_model_layers(m, start=2, block_size=2)  # drop layers 2,3
        assert len(m.model.layers) == 4
        assert m.config.num_hidden_layers == 4

    def test_prune_rejects_touching_last_layer(self):
        from soup_cli.utils.shrink import prune_model_layers
        m = self._tiny(6)
        with pytest.raises(ValueError):
            prune_model_layers(m, start=4, block_size=2)  # would include last (idx 5)

    def test_prune_rejects_touching_first_layer(self):
        from soup_cli.utils.shrink import prune_model_layers
        m = self._tiny(6)
        with pytest.raises(ValueError):
            prune_model_layers(m, start=0, block_size=2)
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError: cannot import name 'shrink_arch_of'`)
- [ ] **Step 3: Implement** (append to `utils/shrink.py`, torch/nn lazy inside)

```python
import re

_ARCH_PATTERNS = {
    "llama": re.compile(r"llama", re.I),
    "qwen": re.compile(r"qwen", re.I),
    "smollm": re.compile(r"smol", re.I),
}
SUPPORTED_SHRINK_ARCHS = tuple(_ARCH_PATTERNS)


def shrink_arch_of(model) -> str:
    arch = getattr(getattr(model, "config", None), "model_type", "") or ""
    names = list(getattr(getattr(model, "config", None), "architectures", []) or [])
    haystack = " ".join([arch, *names])
    for fam, pat in _ARCH_PATTERNS.items():
        if pat.search(haystack):
            return fam
    raise ValueError(
        f"soup shrink v1 supports {SUPPORTED_SHRINK_ARCHS}; "
        f"got model_type={arch!r} (unsupported)."
    )


def layer_list(model):
    shrink_arch_of(model)  # arch guard
    try:
        return model.model.layers
    except AttributeError as exc:
        raise ValueError("model has no .model.layers ModuleList") from exc


def prune_model_layers(model, start: int, block_size: int) -> None:
    import torch.nn as nn

    layers = layer_list(model)
    n_total = len(layers)
    if block_size < 1 or block_size >= n_total:
        raise ValueError(f"block_size must be in [1, {n_total - 1}]")
    end = start + block_size  # exclusive
    if start < 1 or end > n_total - 1:
        raise ValueError(
            f"dropped block [{start}, {end}) must stay within [1, {n_total - 1}) "
            "(the first and last layer are protected)"
        )
    kept = [layers[i] for i in range(n_total) if not (start <= i < end)]
    model.model.layers = nn.ModuleList(kept)
    model.config.num_hidden_layers = len(kept)
```

- [ ] **Step 4: Run — expect PASS**
- [ ] **Step 5: Commit** — `git commit -m "feat(shrink): arch allowlist + prune_model_layers (v0.71.29)"`

---

### Task 3: `compute_layer_importance` (off-by-one pinned) + `select_drop_block`

**Files:** Modify `src/soup_cli/utils/shrink.py`; Test `tests/test_v07129.py`

**Interfaces:**
- Produces:
  - `compute_layer_importance(model, tokenizer, prompts, *, block_size:int, device:str, max_prompts:int=256) -> list[LayerImportance]` — one `output_hidden_states=True` forward per prompt; accumulates per-token angular distance `d = arccos(cos)/pi` between `hidden_states[L]` and `hidden_states[L+block_size]` over every non-pad token; returns one `LayerImportance` per **position-valid** start `L in [1, num_layers-block_size-1]`, sorted by ascending distance.
  - `select_drop_block(importances) -> LayerImportance` — the min-distance (first) entry; raises if empty.
  - `resolve_drop_count(num_layers, *, drop_ratio, drop_layers) -> int` — exactly one of the two set; `round(ratio*num_layers)`; validate `1 <= count <= num_layers-2`.

- [ ] **Step 1: Write failing tests** — the off-by-one boundary test uses a monkeypatched model returning known hidden states.

```python
class TestImportance:
    def test_off_by_one_boundary_indices(self, monkeypatch):
        # A fake model whose output_hidden_states are deterministic per layer so
        # we can assert the block uses hidden_states[L] and hidden_states[L+n].
        import numpy as np
        import torch
        from soup_cli.utils import shrink

        num_layers = 4
        # hidden_states length must be num_layers+1
        hs = [torch.ones(1, 3, 8) * (k + 1) for k in range(num_layers + 1)]

        class _Cfg:
            model_type = "llama"; architectures = ["LlamaForCausalLM"]
            num_hidden_layers = num_layers
        class _Out:
            hidden_states = tuple(hs)
        class _Model:
            config = _Cfg()
            def eval(self): return self
            def __call__(self, **kw): return _Out()
        class _Tok:
            def __call__(self, text, **kw):
                return {"input_ids": torch.ones(1, 3, dtype=torch.long),
                        "attention_mask": torch.ones(1, 3, dtype=torch.long)}
            def to(self, device): return self

        # cos(h_L, h_{L+n}) between constant vectors == 1 -> distance 0 for all.
        imps = shrink.compute_layer_importance(
            _Model(), _Tok(), ["hi"], block_size=1, device="cpu")
        # valid starts for n=1, num_layers=4: L in [1, 4-1-1=2] -> {1,2}
        starts = sorted(i.start for i in imps)
        assert starts == [1, 2]
        assert all(abs(i.angular_distance) < 1e-6 for i in imps)

    def test_len_hidden_states_invariant_documented(self):
        # guardrail: block_size too large -> no valid starts -> raises
        import torch
        from soup_cli.utils import shrink
        # reuse a real tiny model path is covered in smoke; here assert select raises empty
        with pytest.raises(ValueError):
            shrink.select_drop_block([])

    def test_resolve_drop_count_ratio(self):
        from soup_cli.utils.shrink import resolve_drop_count
        assert resolve_drop_count(30, drop_ratio=0.25, drop_layers=None) == 8  # round(7.5)
        assert resolve_drop_count(30, drop_ratio=None, drop_layers=6) == 6

    def test_resolve_drop_count_rejects_both_or_neither(self):
        from soup_cli.utils.shrink import resolve_drop_count
        with pytest.raises(ValueError):
            resolve_drop_count(30, drop_ratio=0.25, drop_layers=6)
        with pytest.raises(ValueError):
            resolve_drop_count(30, drop_ratio=None, drop_layers=None)

    def test_resolve_drop_count_position_bound(self):
        from soup_cli.utils.shrink import resolve_drop_count
        with pytest.raises(ValueError):
            resolve_drop_count(4, drop_ratio=None, drop_layers=3)  # > num_layers-2
```

- [ ] **Step 2: Run — expect FAIL**
- [ ] **Step 3: Implement** (append; torch/numpy lazy)

```python
def resolve_drop_count(num_layers, *, drop_ratio, drop_layers) -> int:
    if (drop_ratio is None) == (drop_layers is None):
        raise ValueError("set exactly one of --drop-ratio / --drop-layers")
    if drop_layers is not None:
        if isinstance(drop_layers, bool) or not isinstance(drop_layers, int):
            raise ValueError("drop_layers must be an int")
        count = drop_layers
    else:
        if isinstance(drop_ratio, bool) or not isinstance(drop_ratio, (int, float)):
            raise ValueError("drop_ratio must be a number")
        if not (0.0 < float(drop_ratio) < 1.0):
            raise ValueError("drop_ratio must be in (0, 1)")
        count = round(float(drop_ratio) * num_layers)
    max_count = num_layers - 2  # protect first + last
    if not (1 <= count <= max_count):
        raise ValueError(
            f"drop count {count} out of range [1, {max_count}] for "
            f"{num_layers} layers (first + last protected)"
        )
    return count


def compute_layer_importance(model, tokenizer, prompts, *, block_size, device,
                             max_prompts=256):
    import numpy as np
    import torch

    n_layers = int(model.config.num_hidden_layers)
    valid_starts = list(range(1, n_layers - block_size))  # L in [1, n-bs-1]
    if not valid_starts:
        raise ValueError(
            f"block_size {block_size} leaves no position-valid block for "
            f"{n_layers} layers (first + last protected)"
        )
    prompt_list = [p for p in prompts if isinstance(p, str) and p.strip()][:max_prompts]
    if not prompt_list:
        raise ValueError("calib prompts must contain at least one non-empty string")

    sums = {s: 0.0 for s in valid_starts}
    counts = {s: 0 for s in valid_starts}
    model.eval()
    with torch.no_grad():
        for text in prompt_list:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=512)
            inputs = {k: v.to(device) for k, v in dict(inputs).items()} \
                if hasattr(inputs, "items") else inputs.to(device)
            out = model(**inputs, output_hidden_states=True)
            hs = out.hidden_states  # len == n_layers + 1
            if len(hs) != n_layers + 1:
                raise ValueError(
                    f"expected {n_layers + 1} hidden states, got {len(hs)}")
            mask = inputs.get("attention_mask")
            for start in valid_starts:
                h_in = hs[start][0].to(torch.float32)          # [seq, D]
                h_out = hs[start + block_size][0].to(torch.float32)
                cos = torch.nn.functional.cosine_similarity(h_in, h_out, dim=-1)
                cos = cos.clamp(-1.0, 1.0)
                dist = torch.arccos(cos) / torch.pi              # [seq]
                if mask is not None:
                    m = mask[0].to(torch.bool)
                    dist = dist[m]
                sums[start] += float(dist.sum().item())
                counts[start] += int(dist.numel())
    imps = [
        LayerImportance(start=s, block_size=block_size,
                        angular_distance=(sums[s] / counts[s]) if counts[s] else float("inf"))
        for s in valid_starts
    ]
    imps.sort(key=lambda x: x.angular_distance)
    return imps


def select_drop_block(importances) -> LayerImportance:
    if not importances:
        raise ValueError("no importance scores to select from")
    return min(importances, key=lambda x: x.angular_distance)
```

- [ ] **Step 4: Run — expect PASS**
- [ ] **Step 5: Commit** — `git commit -m "feat(shrink): importance scan + block selection (v0.71.29)"`

---

### Task 4: `commands/shrink.py` — prune-only orchestration + CLI registration

**Files:** Create `src/soup_cli/commands/shrink.py`; Modify `src/soup_cli/cli.py`; Test `tests/test_v07129.py`

**Interfaces:**
- Consumes all Task 1-3 symbols + `live_eval.load_model_and_tokenizer`, `live_eval.compute_eval_loss`.
- Produces: Typer `app` with a single `shrink` command; registered in `cli.py` as `app.add_typer` OR `app.command`. Path guard `_under_cwd(path) -> str` (realpath+commonpath), `_load_calib(path) -> list[str]` (O_NOFOLLOW, size cap, JSONL `{"text":...}` or messages), `_perplexity(model, tok, prompts, device) -> float` (= `exp(compute_eval_loss(...))`).

- [ ] **Step 1: Write failing CLI tests** (help + plan-only + failure modes via CliRunner; the live prune happy path is covered by the Step-6 smoke, but a CPU tiny-model prune-only run is added here)

```python
class TestShrinkCli:
    def test_registered_and_help(self):
        from typer.testing import CliRunner
        from soup_cli.cli import app
        r = CliRunner().invoke(app, ["shrink", "--help"])
        assert r.exit_code == 0, (r.output, repr(r.exception))
        assert "drop-ratio" in r.output

    def test_rejects_both_drop_flags(self, tmp_path):
        from typer.testing import CliRunner
        from soup_cli.cli import app
        calib = tmp_path / "c.jsonl"; calib.write_text('{"text":"hi"}\n', encoding="utf-8")
        # run from tmp_path so calib is under cwd
        r = CliRunner().invoke(app, ["shrink", "--model", "x", "--drop-ratio", "0.25",
                                     "--drop-layers", "5", "--calib", str(calib)])
        assert r.exit_code != 0

    def test_calib_outside_cwd_rejected(self, tmp_path, monkeypatch):
        # calib pointed outside cwd -> reject
        ...
```

Plus a CPU prune happy-path test that builds a tiny Llama, saves it to a temp dir, and invokes `shrink` against that path with `--device cpu --drop-layers 2 --no-heal`, asserting `<out>/model/config.json` has reduced `num_hidden_layers` and a verdict JSON with `--output`.

- [ ] **Step 2: Run — expect FAIL**
- [ ] **Step 3: Implement `commands/shrink.py`** — Typer command:
  - parse/validate flags; `_under_cwd` on calib/heal/out; bounds on tolerance/heal-steps.
  - `load_model_and_tokenizer(model, device=...)` (trust_remote_code probe+warn).
  - `resolve_drop_count` → `compute_layer_importance` → `select_drop_block`.
  - `--plan-only`: render importance table + chosen block + would-run heal cmd; exit 0.
  - `prune_model_layers`; `model.save_pretrained(out/model)`; `tokenizer.save_pretrained(out/model)`.
  - reload from `out/model`; `ppl_original` (orig model), `ppl_pruned` (reloaded).
  - (heal deferred to Task 5 — here `healed=False`, `ppl_final=ppl_pruned`).
  - `decide_shrink(...)`; `render_shrink_panel`; write `out/shrink_report.json` if `--output`; `raise typer.Exit(0 if SHIP else 2)`.
- Register in `cli.py` (`from soup_cli.commands import shrink as shrink_cmd; app.add_typer` or `app.command("shrink")(shrink_cmd.shrink)` — match how other single commands like `ship`/`diagnose` are wired).
- [ ] **Step 4: Run — expect PASS**
- [ ] **Step 5: Commit** — `git commit -m "feat(shrink): prune orchestration + CLI (v0.71.29)"`

---

### Task 5: Subprocess distill-heal + fuse

**Files:** Modify `src/soup_cli/commands/shrink.py`; Test `tests/test_v07129.py`

**Interfaces:**
- Produces: `_run_heal(pruned_dir, teacher, heal_data, steps, out_dir, device) -> str` — writes a distill `soup.yaml` (validated via `load_config_from_string`), runs `soup train` as a subprocess (argv list, no shell, timeout), returns adapter dir; then fuse adapter into `pruned_dir` via the existing merge path (reuse `commands/merge.py` helper or subprocess `soup merge`). Mirrors `ra_dit_run._run_train_subprocess`.

- [ ] **Step 1: Write failing tests** — assert the generated distill yaml parses (`load_config_from_string`) with `task=distill`, `teacher_model=<orig>`, `base=<pruned>`, LoRA on; and that `_run_heal` is invoked when `--heal` is passed (monkeypatch the subprocess runner + fuse to no-op and assert `healed=True` in the report).

```python
class TestHeal:
    def test_generated_distill_config_parses(self, tmp_path):
        from soup_cli.commands.shrink import _build_heal_config_yaml
        from soup_cli.config.loader import load_config_from_string
        y = _build_heal_config_yaml(pruned_dir="./out/model", teacher="orig/model",
                                    heal_data="./heal.jsonl", steps=200, out_dir="./out/heal")
        cfg = load_config_from_string(y)
        assert cfg.task == "distill"
        assert cfg.training.teacher_model == "orig/model"
        assert cfg.base == "./out/model"

    def test_heal_path_sets_healed(self, tmp_path, monkeypatch):
        # monkeypatch _run_train_subprocess + fuse; assert report.healed True
        ...
```

- [ ] **Step 2: Run — expect FAIL**
- [ ] **Step 3: Implement** `_build_heal_config_yaml` + `_run_heal`; wire into the command when `--heal` set: run heal → fuse adapter into `out/model` → reload → `ppl_final` on the fused reloaded model → `healed=True`.
- [ ] **Step 4: Run — expect PASS**
- [ ] **Step 5: Commit** — `git commit -m "feat(shrink): subprocess distill-heal + fuse (v0.71.29)"`

---

### Task 6: Registry attach + docs + version bump

**Files:** Modify `commands/shrink.py`, `pyproject.toml`, `src/soup_cli/__init__.py`, `docs/*`, `.claude/CLAUDE.md`, `CHANGELOG.md`, `.claude/history/*`; Test `tests/test_v07129.py`

- [ ] **Step 1:** Test `--attach-to-registry` attaches a `shrink_report` artifact (reuse `registry.attach.attach_artifact` / `lookup_entry_by_output_dir`; mirror `diagnose --attach-to-registry`). Write failing test.
- [ ] **Step 2:** Run — FAIL.
- [ ] **Step 3:** Implement attach; then docs pass (deferred to Release Checklist steps 7-13 but the code-facing `--attach-to-registry` lands here).
- [ ] **Step 4:** Run — PASS.
- [ ] **Step 5:** Commit.

---

## Self-Review

- **Spec coverage:** importance (T3) · prune+reload (T2,T4) · verdict (T1) · heal (T5) · registry (T6) · CLI/plan-only/security (T4) · arch allowlist (T2) · off-by-one pinned (T3) · position bounds (T2,T3) — all mapped.
- **Placeholders:** the two `...` test bodies (T4 outside-cwd, T5 heal-path) are the only stubs — expand to real assertions during execution (they are named + scoped, not hand-wavy).
- **Type consistency:** `LayerImportance`/`ShrinkVerdict`/`decide_shrink`/`prune_model_layers`/`compute_layer_importance`/`select_drop_block`/`resolve_drop_count` names identical across tasks. `block_size` (not `n`/`block_len`) used everywhere.

## Release Checklist (post-implementation)

After Tasks 1-6: run the full `.claude/CLAUDE.md` Release Checklist — step 5 = 5 sequential ECC reviews (python/code/security/tdd/verification-loop, fix every finding), step 6 = live RTX-3050 smoke (SmolLM2-360M drop 25% → ppl → heal 200 steps → verdict + all failure modes), steps 7-21 docs+ship. Version `0.71.28 → 0.71.29`.
