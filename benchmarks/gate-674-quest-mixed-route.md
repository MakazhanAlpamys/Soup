# QuEST mixed W4/A4+A16 route: evaluation gate passed (#674)

**The predeclared seven-A16 rescue passed its evaluation-only quality gate:
0.086344 nat/target [0.080241, 0.093190] above fixed FP over 704 examples and
25,017 targets. This is one model, fake quantization and no training in this
confirmation; it is not strict W4A4, upstream QuEST parity, a packed-INT4
result or an efficiency claim.**

Measured by [@Shutaru](https://github.com/Shutaru). This record follows the
[failed strict-W4A4 SFT gate](gate-674-quest-w4a4-sft.md); it does not revise or
replace that result. Later bounded diagnostics localized enough of the quality
gap to block 23 to justify one frozen mixed-precision confirmation.

## Predeclared hierarchy

The two routes were selected on the already-spent RD70 DEV192 panel and frozen
before either was scored on DEV192B or CONFIRM512:

1. **Primary — one A16:** all 168 transformer linear weights remain W4;
   `model.layers.23.mlp.down_proj` uses an A16 activation and the other 167 use
   A4.
2. **Rescue — seven A16:** all 168 weights remain W4; all seven attention and
   MLP linears in block 23 use A16 activations and the other 161 use A4. This
   arm was eligible only if the primary failed.

Both use group size 128 and the frozen full-width normalized Hadamard route.
No training, clipping, calibration, route selection or learned-parameter update
was allowed in this confirmation.

The primary analysis pools two disjoint panels:

- DEV192B: 192 examples / 7,033 supervised response targets, first scored here;
- CONFIRM512: 512 examples / 17,984 targets; fixed-FP and native-W4A4 controls
  were already frozen, but neither mixed route had been scored there;
- pooled: 704 examples / 25,017 targets.

Every statistic is a paired sequence-level ratio of sums, averaged equally over
training seeds 42, 7 and 123. Intervals use 2,000 paired percentile-bootstrap
resamples with bootstrap seed 76.

A candidate passes only if:

1. the pooled gap to fixed FP has a paired 95% upper bound at or below
   0.1 nat/target;
2. the pooled change from native W4A4 has a paired 95% upper bound below zero;
3. the point change from native W4A4 is negative for every training seed on
   both panels.

## Result

Lower gap is better. Negative change from native W4A4 is an improvement.

| Panel | Candidate | Gap vs fixed FP | Paired 95% interval | Change vs native W4A4 |
| --- | --- | ---: | ---: | ---: |
| DEV192B | one A16 | 0.092837 | [0.078875, 0.106514] | -0.007607 |
| DEV192B | seven A16 | 0.083981 | [0.070769, 0.096989] | -0.016463 |
| CONFIRM512 | one A16 | 0.096218 | [0.088824, 0.103862] | -0.006107 |
| CONFIRM512 | seven A16 | 0.087268 | [0.080281, 0.094832] | -0.015057 |
| Pooled 704 | one A16 | **0.095267** | **[0.088739, 0.102443]** | **-0.006529** |
| Pooled 704 | seven A16 | **0.086344** | **[0.080241, 0.093190]** | **-0.015452** |

The one-A16 primary improved on native W4A4 in every seed and on both panels,
but its pooled upper bound was 0.102443. It therefore failed without rounding.

The seven-A16 rescue passed. Its pooled change from native W4A4 was
**-0.015452 [-0.017552, -0.013528]**, and its point change was negative for each
of seeds 42, 7 and 123 on each panel. The independent audit recomputed the panel,
pooled and per-seed statistics and selected `seven-a16-block23` from the declared
hierarchy.

## Instrument card and execution accounting

| Item | Recorded setup |
| --- | --- |
| OS / Python | Windows 11 `10.0.26200`; Python `3.12.10` |
| GPUs | 2 x NVIDIA GeForce RTX 3090, 24,576 MiB each |
| NVIDIA driver | `591.86` |
| PyTorch / CUDA build | `2.11.0+cu128` / `12.8` |
| Transformers / NumPy | `5.16.1` / `2.5.3` |
| Other captured packages | safetensors `0.8.0`; huggingface-hub `1.30.0`; tokenizers `0.23.2` |
| Model | `ahxt/LiteLlama-460M-1T`, revision `77b8a976440e7d1ea5a890eaf1e0175b1cac0078` |
| Dataset | `databricks/databricks-dolly-15k`, revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` |
| Work performed | 2,496 forwards; **zero backwards and zero optimizer updates** |
| Worker | complete and reaped; 721.312 s; maximum sampled RSS 5,426,659,328 bytes |

Both cards were visible to the evaluation. These figures describe
the confirmation run; they are not controlled latency, throughput or memory
benchmarks.

## Panel-spending history

| Panel | Status after RD76 |
| --- | --- |
| RD70 DEV192 | Already spent for route selection; excluded from this confirmation |
| RD73 DEV192B | First evaluated here; now spent |
| RD75 CONFIRM512 | Controls existed; both mixed routes first evaluated here; now spent |
| Pooled 704 | Primary analysis over DEV192B + CONFIRM512; no additional examples |
| Reserved FINAL256 | **Never evaluated. Still reserved.** |

The pass therefore supports the frozen mixed route on one model. FINAL256 was
not consumed to make the engineering decision.

## Provenance bindings

The retained files remain local; the hashes bind this public record to the
frozen protocol, code, panels, endpoints, result, worker receipt and independent
audit.

| Artifact | SHA-256 |
| --- | --- |
| Freeze manifest | `52af1943044d7b874546bd2727c742db1b14a3477d336bb71942a1106ee00c91` |
| Result | `94fee10542da29281f7753cbf221a3421ad5acf67f2b290e52d65acece359cdf` |
| Worker receipt | `f9f4527c9e7a376c08e41f0041b5e7121b524183c60199346242d39cd50a492e` |
| Independent audit | `09676b7efed3dcdf1951ebd6ad03fe4ddbd492904a4f3d624476b22dbe1f6eb7` |
| Protocol text | `df888abfc9a9bcd4e01f1f17fa39747d546f8d866d24692d4570688aa5dfe4ef` |
| DEV192B manifest | `9fe9b2b1edafd739eefa5dcfc341a6500aae98b80f656d21116dabf6d896b9a0` |
| DEV192B panel | `3c5e438f9dd7282396721da81423314a7111ea297feac3883d6a68cbfe641961` |
| CONFIRM512 manifest | `879c7d080dae1aa4c3c04c580197370f101eca5656612588045792d7a6b763eb` |
| CONFIRM512 panel | `6b981b44622ead9ca5a90f2475a3546615a9513d448a23938297386e9b095a8d` |

Frozen endpoint hashes:

| Training seed | SHA-256 |
| ---: | --- |
| 42 | `a2f92e94ba8364709acd924797ab9732df9e0205ffb4c4128807d65dca81bda4` |
| 7 | `15226c16dcac195fca285f4e3da52d1c675c82d1f963c03e58f0549687c0a801` |
| 123 | `fdd52fa24cebdb76343e18c7b0b5c3d8252a0388f548322c5d5d1ac831c5acdb` |

Frozen RD76 code hashes:

| File | SHA-256 |
| --- | --- |
| `rd76/__init__.py` | `d73f91d66bc04dc65a8d1fd284402eaac7623d0034152309e0fa7bb64671e965` |
| `rd76/confirm_mixed.py` | `c53c16616057bcef3c32af26aeea28fc765cdbc73e5ad27c2f5f9887723abec4` |
| `rd76/test_confirm_mixed.py` | `0854f9bceb34dc667e0bfdbe9f6211524b0f52697fa5b3104cc63ebd7cb86f35` |

The independent audit marked every declared check true, including code and
dependency hashes, compute-graph accounting, frozen panel bindings, exact panel
and pooled statistics, hierarchical selection, worker completion and receipt
binding. It records `final_evaluated: false`.

## Scope and limits

This result is deliberately narrow:

- it is **not strict W4A4**: seven of 168 activation routes are A16;
- it is **not upstream QuEST numerical parity**;
- it uses dense fake quantization, not packed INT4 kernels;
- it makes no latency, throughput, VRAM or deployment-efficiency claim;
- it contains no backward pass and does not validate mixed-route training quality;
- it covers one model architecture and does not establish cross-model generality;
- it is not itself a Soup integration or a generic save/load contract.

The engineering integration must preserve this exact route, make its metadata
explicit and reject unsupported topologies rather than silently generalizing
the result.
