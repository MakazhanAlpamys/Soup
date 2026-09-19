Rebased onto current `main` (it was 14 commits behind) and re-ran everything. The commit content is unchanged; only the base moved.

Verification on the rebased head `295882d`:

```
pytest tests/test_issue839_fast_lora_single_projection.py
15 passed, 2 skipped
```

The two skips are the CUDA-gated ones I flagged when claiming: NF4 parity needs `bitsandbytes` on a card, and the benchmark reports only. They skip on `needs a CUDA device`, so they have not been run by me and the PR does not claim numbers from them.

I sabotaged the kernel three ways rather than trust the green:

| mutation | result |
|---|---|
| drop the scaling fold in forward (`alpha=ctx.scaling` to `1.0`, both sites) | **2 failed** |
| drop the scaling from the two backward gradient terms | **5 failed** |
| save one tensor instead of five in `save_for_backward` | **errored**: `ValueError: not enough values to unpack (expected 5, got 4)` |

The middle one is the one worth noting: it fails five tests, not just the gradcheck, so backward parity against unpatched peft is genuinely pinned rather than asserted.

`gradcheck` runs in float64 on CPU for `x`, `A` and `B`, with and without bias.

@MakazhanAlpamys this is the first PR of the #792 tracker and the shape the other two reuse, so it is the one worth getting right. Happy to change the instance-patching or the `save_for_backward` split if you would rather it looked different before the second one is built on it.
