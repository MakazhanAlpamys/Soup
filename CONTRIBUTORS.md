# Contributors

Soup is built by its community. Thank you to everyone who has contributed code,
tests, docs, and ideas. ❤️

This list is maintained by hand alongside the GitHub
[contributors graph](https://github.com/MakazhanAlpamys/Soup/graphs/contributors).
Merged a PR and don't see yourself here? Open a PR adding your line — that counts too.

## Maintainer

- **Alpamys** ([@MakazhanAlpamys](https://github.com/MakazhanAlpamys)) — creator & lead maintainer

## Contributors

Listed by first contribution. PR numbers link the work.

- **Salil Mhatre** ([@Deadpool2000](https://github.com/Deadpool2000))
  - `soup version --json` for machine-readable CI output ([#6](https://github.com/MakazhanAlpamys/Soup/pull/6))
  - RAM + disk-space checks in `soup doctor` ([#7](https://github.com/MakazhanAlpamys/Soup/pull/7))
  - `soup runs clean` for smart checkpoint space management ([#9](https://github.com/MakazhanAlpamys/Soup/pull/9))
  - Official Docker support for easier onboarding ([#20](https://github.com/MakazhanAlpamys/Soup/pull/20))
  - `soup bench` — model speed + VRAM measurement ([#25](https://github.com/MakazhanAlpamys/Soup/pull/25))
  - `--prompts-file` option for `soup bench` ([#30](https://github.com/MakazhanAlpamys/Soup/pull/30))
  - Happy-path + CPU-warning tests for `soup bench` ([#31](https://github.com/MakazhanAlpamys/Soup/pull/31))
  - `soup cost` — cloud GPU training cost estimation ([#42](https://github.com/MakazhanAlpamys/Soup/pull/42))
  - `--nccl` flag for `soup doctor` multi-GPU bandwidth checks ([#178](https://github.com/MakazhanAlpamys/Soup/pull/178))
  - Ready-made `qwen2.5-coder-7b-sft` recipe ([#285](https://github.com/MakazhanAlpamys/Soup/pull/285))
  - `soup data split --stratify-semantic` — a random split can leave a whole topic out of the validation set, so a regression in it is invisible; rows are now clustered by meaning and each cluster split proportionally ([#388](https://github.com/MakazhanAlpamys/Soup/pull/388))
- **Chinmaya Sahu** ([@csking101](https://github.com/csking101))
  - DPO example config, sample data, and tests ([#48](https://github.com/MakazhanAlpamys/Soup/pull/48))
  - FP8 `rowwise` + `rowwise_with_gw_hp` scaling recipes ([#62](https://github.com/MakazhanAlpamys/Soup/pull/62))
- **Yixuan Xu** ([@mzl2233](https://github.com/mzl2233))
  - Guard diagnose-gate on distributed worker ranks ([#169](https://github.com/MakazhanAlpamys/Soup/pull/169))
- **dreamer0129** ([@dreamer0129](https://github.com/dreamer0129))
  - Rich-markup escape fix in legacy `soup adapters` commands ([#175](https://github.com/MakazhanAlpamys/Soup/pull/175), adopted in-tree as [#174](https://github.com/MakazhanAlpamys/Soup/issues/174))
- **Vivaan Dhawan** ([@VIVAAN-DHAWAN](https://github.com/VIVAAN-DHAWAN))
  - Reject pickle/zip streams renamed to `.safetensors` via magic-byte check ([#198](https://github.com/MakazhanAlpamys/Soup/pull/198))
- **Shivam** ([@shivam2931120](https://github.com/shivam2931120))
  - Tokenizer-aware repetition scoring for the echo-trap detector ([#242](https://github.com/MakazhanAlpamys/Soup/pull/242))
- **gittihub-jpg** ([@gittihub-jpg](https://github.com/gittihub-jpg))
  - Manifest-level dotted-path custom transforms for `soup build` ([#255](https://github.com/MakazhanAlpamys/Soup/pull/255))
  - `--energy` flag for `soup bom emit` — thread energy/CO₂ into the ML-BOM ([#256](https://github.com/MakazhanAlpamys/Soup/pull/256))
- **shatakshi-1404** ([@shatakshi-1404](https://github.com/shatakshi-1404))
  - Unit tests for the `warmup.py` auto-warmup-steps helper ([#274](https://github.com/MakazhanAlpamys/Soup/pull/274))
- **Kondamwar Akshaya Shrikant** ([@Akshaya-reddy18](https://github.com/Akshaya-reddy18))
  - Friendlier error messages — richer CUDA-OOM hint + Hugging Face gated-repo and `trust_remote_code` mappings + tests ([#282](https://github.com/MakazhanAlpamys/Soup/pull/282))
- **Darsh** ([@CODING-DARSH](https://github.com/CODING-DARSH))
  - Harden judge-URL validation against hostname prefix bypass (`startswith` → `urlparse`) in `eval/gate.py` ([#288](https://github.com/MakazhanAlpamys/Soup/pull/288))
  - Apply configured vocabulary expansion (`data.add_new_tokens` / `new_special_tokens`) during SFT trainer init ([#287](https://github.com/MakazhanAlpamys/Soup/pull/287))
  - Reuse the shared vocab-expansion helper in the vision + audio SFT paths ([#291](https://github.com/MakazhanAlpamys/Soup/pull/291))
  - Honor configured vocab expansion in the DPO / IPO / KTO / BCO trainers ([#293](https://github.com/MakazhanAlpamys/Soup/pull/293))
  - Honor configured vocab expansion in the ORPO / SimPO / GRPO trainers ([#295](https://github.com/MakazhanAlpamys/Soup/pull/295))
  - `soup mcp serve --allow-execute` — the execution gate, kept a separate and stronger opt-in than `--allow-mutating`, with the tools still plan-only in this slice ([#391](https://github.com/MakazhanAlpamys/Soup/pull/391))
  - Gated `train_execute` / `export_execute` behind a single-use server confirmation token, with the config snapshotted at plan time and protected directories digested by content rather than by mtime ([#393](https://github.com/MakazhanAlpamys/Soup/pull/393))
  - Corrected a contributor's handle carried in the v0.73.2 CHANGELOG — a one-line fix to somebody else's credit, which is the kind of thing that normally goes unmade ([#400](https://github.com/MakazhanAlpamys/Soup/pull/400))
- **Ekaanksh Patil** ([@Ekaanksh-dev](https://github.com/Ekaanksh-dev))
  - Batch the PRM reward forward pass in `PRMScorer.__call__` (single `[B, T]` forward) ([#301](https://github.com/MakazhanAlpamys/Soup/pull/301))
- **Sanjay Santhanam** ([@Sanjays2402](https://github.com/Sanjays2402))
  - Run built-in benchmark gate tasks through `ForgettingDetector` — every `type: benchmark` eval-gate task had always failed ([#315](https://github.com/MakazhanAlpamys/Soup/pull/315))
- **Nicolás Ramos** ([@nicolasramos](https://github.com/nicolasramos))
  - `backend: mlx` was never dispatched — every MLX run trained through the transformers wrapper instead, and the saved MLX "adapter" was a full fine-tune because the model was never frozen before LoRA ([#362](https://github.com/MakazhanAlpamys/Soup/pull/362))
- **William Yang** ([@wilyan09007](https://github.com/wilyan09007))
  - `training.seed` reached the SFT wrapper and nothing else — seventeen other task wrappers trained at HF's default 42 with no error, so replicates that differed only in the seed were the same run; the seed is now applied before the adapter is drawn, not only inside `Trainer` ([#381](https://github.com/MakazhanAlpamys/Soup/pull/381))
  - Under `use_fsdp2_compile`, every `checkpoint-*` kept `torch.compile`'s key prefix and resumed **silently** from a re-zeroed adapter — normalisation now runs as each checkpoint is written, ahead of anything that publishes it ([#380](https://github.com/MakazhanAlpamys/Soup/pull/380))
- **Amir Fathi** ([@AmirF194](https://github.com/AmirF194))
  - A streamed model's `named_parameters()` carried the wrapper's `.inner.` segment, so a name-keyed comparison against a resident model shared no names at all and a correctness gate reported `0/0` as a pass ([#384](https://github.com/MakazhanAlpamys/Soup/pull/384))
  - `training.stream_vram_override` — the layer-streaming pre-flight measured free VRAM with a device-level driver query, so it could not see a per-process cap and there was no way to make it simulate one ([#386](https://github.com/MakazhanAlpamys/Soup/pull/386))
  - The VRAM pre-flight never called its own calibration hook, so the guard against a stack whose loss path under-budgets by 12.5% sat inert with no caller ([#390](https://github.com/MakazhanAlpamys/Soup/pull/390))
  - `kl_control` re-wrote the same β on every hold step, so a non-acting run was not the no-op `log_only` claims to be; the mitigation log now carries `held` / `acted` / `released` as a field rather than as free text ([#414](https://github.com/MakazhanAlpamys/Soup/pull/414))
- **Ben Younes** ([@ousamabenyounes](https://github.com/ousamabenyounes))
  - `MitigationLogWriter` dropped every record in silence once its parent directory vanished mid-run — the controller kept acting while its evidence stopped growing ([#398](https://github.com/MakazhanAlpamys/Soup/pull/398))
  - `soup draft distill --steps N` delivered only ~N/4.44 optimiser steps — `val_split` and `gradient_accumulation_steps` both divide the budget, and the epoch arithmetic ignored them ([#399](https://github.com/MakazhanAlpamys/Soup/pull/399))
  - The `soup ship` MCQ scorer read `oxed {A}` as no-answer — LaTeX permits a space before the brace and models emit it, and the cue tier cannot rescue it ([#396](https://github.com/MakazhanAlpamys/Soup/pull/396))
  - `--noise-floor` shipped without a config surface, so it was the one `soup ship` gate-policy flag that could not be committed to `soup.yaml`; the bounds import from `ship_verdict` so the schema and the CLI validator cannot disagree ([#410](https://github.com/MakazhanAlpamys/Soup/pull/410))
  - A dead MCP watcher left its run at `running` in the tracker forever — reconciled on read, with a Windows liveness branch because `os.kill(pid, 0)` there sends a console Ctrl+C rather than checking existence ([#407](https://github.com/MakazhanAlpamys/Soup/pull/407))
  - The one-active-execution cap lived in process memory, so a restarted MCP server could double-book it ([#408](https://github.com/MakazhanAlpamys/Soup/pull/408))
  - The `soup ship` leg-1 noise floor was measured in `--task-mode metric` only, so in the judge modes a win smaller than the instrument's resolution still counted; it is now measured everywhere and **labelled**, so a decode-only floor is distinguishable from one carrying judge variance ([#419](https://github.com/MakazhanAlpamys/Soup/pull/419))
  - `detect_disk_kind` could not see through virtio, so a 1.5 GB/s cloud disk was classified HDD and refused the streaming tier — and when review found that the *fix* cited a rate from module state the cache never reset, producing `'hdd' (measured 2.00 GB/s, under the 1.0 GB/s NVMe floor)`, they removed the global rather than clearing it on the cache branch: the rate now travels in a frozen classification that is stripped on override, so the message cannot cite a verdict it did not produce ([#411](https://github.com/MakazhanAlpamys/Soup/pull/411))
  - `training.bnb_4bit_use_double_quant` was validated and then read by nothing — every 4-bit path hardcoded `True`, so setting it changed the config fingerprint and nothing else. Made `Optional[bool] = None` rather than `True`, because a plain default emits the key into `model_dump()` and breaks round-tripping for 21 of 173 shipped configs; and when review showed the first round's tests fired on *spelling* rather than behaviour — and its companion passed against `main`'s untouched file — both were deleted rather than patched ([#418](https://github.com/MakazhanAlpamys/Soup/pull/418))
  - `soup env check` now audits the live environment against the bounds Soup declares about itself, so `pip install vllm` quietly downgrading `transformers` past the `<5.0.0` cap is caught. The bound is read from package metadata rather than restated — and when review found the false-positive fix had, in closing it, made #368's own case unreachable, they narrowed enforcement to the ABI-relevant packages so both properties hold at once ([#421](https://github.com/MakazhanAlpamys/Soup/pull/421))
  - `soup bom emit` / `soup attest emit --attach-to-registry` — a published `soup card` now carries its ML-BOM and in-toto attestation. Told the signed path registered the statement but not its detached `.sig`, so the card linked an attestation the registry alone could not verify, they fixed it the way they had already solved the same multi-file problem for `bom --format both` ([#420](https://github.com/MakazhanAlpamys/Soup/pull/420))
- **Faisal Fayaz** ([@Faisal01011](https://github.com/Faisal01011))
  - Added the `qwen3.5-4b-pretrain` recipe — and shipped it with a test pinning the literal repo id, which is the only thing that catches a *consistently* wrong id (wrong in both `RecipeMeta.model` and the inline `base:`, so the two still agree). Every catalog-wide invariant passes that mutation; this is the defect class that shipped `glm-5` pointing at `THUDM` instead of `zai-org` ([#422](https://github.com/MakazhanAlpamys/Soup/pull/422))
  - Added the `deepseek-v4-flash-grpo` recipe, carrying the same literal-repo-id guard a second time — the consistently-wrong-id mutation stays green through every cross-field invariant and is caught only by that test ([#432](https://github.com/MakazhanAlpamys/Soup/pull/432))
  - `materialize_meta_adapters` returns a count, and on newer peft it returns `0` as a matter of course — so `0` stopped distinguishing "nothing to do" from the silent no-training case its own docstring warns about. Demoted the count to a diagnostic and moved the decision into a separate postcondition, so the caller can no longer choose to ignore it; the trigger needs a peft this repo does not pin, so the test stubs the capability and asserts the decision ([#435](https://github.com/MakazhanAlpamys/Soup/pull/435))
  - Then took the non-blocking follow-up from that review unprompted: the guard's `lora_` restriction was called deliberate in its docstring but nothing pinned it, so a later broadening would have started refusing healthy streamed builds ([#437](https://github.com/MakazhanAlpamys/Soup/pull/437))
- **Shutaru** ([@Shutaru](https://github.com/Shutaru))
  - Kept the Transformers SFT import off the MLX dispatch route, so `backend: mlx` cannot reach the PyTorch/TRL stack even if `sft.py` stops being import-light later — and, told the PR did not fix the defect its title claimed, retitled it to match reality rather than defending the framing, leaving [#394](https://github.com/MakazhanAlpamys/Soup/issues/394) open for the unexplained hang. The `mlx-smoke` job it adds asserts mlx is *present*, not merely that torch is absent: the earlier shape went green having executed nothing ([#431](https://github.com/MakazhanAlpamys/Soup/pull/431))
  - Found and fixed a silent data-corruption bug in assistant-only loss masking: `BatchEncoding` is not a `dict`, so the guard missed and the mask was built from the mapping's **key strings** — no exception, normal loss curve. Split it out of #426 on request, then decided the case the issue asked to be decided rather than inherited: an all-zero mask with assistant messages present is rejected rather than honoured (measured: 0 trained tokens before, 2 after) ([#439](https://github.com/MakazhanAlpamys/Soup/pull/439))
- **Achuth Reddy Bangaru** ([@AchuthReddy-16](https://github.com/AchuthReddy-16))
  - `soup train --no-reexec` printed a launch command with the user's own flags dropped, so following it trained without `--fsdp` while still succeeding. Rather than patch the printed copy, they deleted it and derived the hint from the argv that actually launches the run — then, asked for a guard, wrote one whose exclusion set forces a *decision* for every new `soup train` flag instead of letting silence make it ([#415](https://github.com/MakazhanAlpamys/Soup/pull/415))
- **Emmanuel Ziggah** ([@blackcoderx](https://github.com/blackcoderx))
  - On Windows a process that genuinely exits with code **259** was indistinguishable from `STILL_ACTIVE`, so it read as alive forever — defeating reconcile-on-read and able to wedge the MCP execution cap shut with no error an operator could act on. Disambiguated with `WaitForSingleObject`, and folded in the deduplication rather than fixing only the headline: two ~50-line copies of the liveness check became one shared module, with a test asserting identity so a third copy fails ([#436](https://github.com/MakazhanAlpamys/Soup/pull/436))
  - `soup data mix --optimize` — the one command whose entire output is a config file — wrote one that would not parse, because `data.train` came out as a YAML list against a `str` field. Collapsed it to the highest-weighted dataset and kept the full ranked breakdown as a comment, so nothing the search learned is discarded; then found the same defect a second time in `soup data mix --live`'s overlay and flagged it instead of widening the diff, along with the reason the suite was green on it — every `--live` test mocks `subprocess.run`, so the artifact is built and never loaded ([#440](https://github.com/MakazhanAlpamys/Soup/pull/440))
  - Then found the same defect a second time, in `soup data mix --live`'s overlay renderer, and fixed that too: every candidate proxy run was handed a config it could not load. Asked whether the PR should exist at all once they took on #443 rather than pushing or stalling silently — the right question, and the answer turned on something they had not seen (#443 has two resolutions and only one reverts it) ([#445](https://github.com/MakazhanAlpamys/Soup/pull/445))
- **Harshit Sharma** ([@harshitthek](https://github.com/harshitthek))
  - `detect_device()` did not know MLX, so an Apple Silicon run reported "CPU (no GPU detected)" and **silently rewrote** `quantization: 4bit` to `none`. The label was never the harm; asked for an explicit decision rather than a disappeared warning, they extracted `resolve_quantization()` with the mechanism named in its docstring — and extraction is also what made it testable, since the surrounding function is 0% covered ([#428](https://github.com/MakazhanAlpamys/Soup/pull/428))

---

Want to join this list? See [CONTRIBUTING.md](CONTRIBUTING.md) — good first issues are
labelled in the [issue tracker](https://github.com/MakazhanAlpamys/Soup/labels/good%20first%20issue).
