# Security Policy

## Supported Versions

We provide security updates for the following versions:

- **Latest minor version:** Active support (e.g., v0.20.x)
- **Previous minor versions:** Bug-fix support only
- **Versions older than 3 minor versions:** No support

Example:
- v0.32.0-0.32.x -- Full support (latest)
- v0.31.0-0.31.x -- Bug-fix support only
- v0.30.x and below -- No support

## Reporting a Vulnerability

**Do not** open a public issue or pull request for security vulnerabilities.

Instead, use [GitHub Security Advisories](https://github.com/MakazhanAlpamys/Soup/security/advisories/new) to report privately, or email **vpn.alpamys@gmail.com** with:

1. **Description**: A clear explanation of the vulnerability
2. **Steps to Reproduce**: How to trigger or demonstrate the issue
3. **Affected Versions**: Which Soup versions are impacted
4. **Suggested Fix** (optional): Any proposed solutions
5. **Contact Info**: Your email for follow-up (optional)

### What to Include

```
To: vpn.alpamys@gmail.com
Subject: Security Vulnerability Report: [Brief Title]

Description:
[Explain the vulnerability in detail]

Affected Component:
[e.g., data/loader.py, trainer/sft.py, etc.]

Steps to Reproduce:
1. [Step 1]
2. [Step 2]
3. ...

Impact:
[What could go wrong? Data exposure? RCE? DoS?]

Suggested Fix (optional):
[Your proposed solution, if any]
```

## Response Timeline

- **Initial Response**: Within 48 hours
- **Assessment**: 1-3 business days
- **Fix Development**: Varies by severity
- **Patch Release**: As soon as possible after fix verification
- **Public Disclosure**: Coordinated with reporter (typically 90 days after patch release)

## Severity Levels

- **Critical**: Remote code execution, data exposure, complete compromise (patch within 24-48 hours)
- **High**: Authentication bypass, privilege escalation, denial of service (patch within 1 week)
- **Medium**: Information disclosure, partial compromise (patch within 2 weeks)
- **Low**: Minor issues with limited impact (patch in next regular release)

## Security Best Practices

When using Soup, follow these practices to stay secure:

### 1. Keep Soup Updated

```bash
pip install --upgrade soup-cli
```

### 2. Protect API Keys

Never commit API keys or secrets to version control. Use environment variables:

```bash
export HUGGINGFACE_TOKEN=your_token_here
export WANDB_API_KEY=your_key_here
soup train
```

### 3. Validate Data

- Only use trusted datasets
- Verify checksums for large datasets
- Inspect data for malicious content before training

### 4. Model Permissions

- Be cautious when downloading models from untrusted sources
- Use model hub providers with verified publishers (HuggingFace, Meta, etc.)
- Keep track of which models you've fine-tuned and their base model sources

### 5. GPU/Compute Safety

- Run on isolated machines if training on sensitive data
- Clear cache and temporary files after training
- Don't share fine-tuned models containing sensitive information

## Known Vulnerabilities

We maintain a log of known security issues and their fixes. This will be updated as issues are discovered and resolved.

### Current Status

No known critical vulnerabilities in current releases.

### Security Hardening History

- **v0.10.10**: Bearer token auth on Web UI, CORS restrictions, path traversal protection, SSRF prevention, max_tokens limits, supply-chain pinning (llama.cpp b5270), deprecated CLI secret flags
- **v0.12.0**: experiment_name path traversal validation, GaLore parameter type enforcement
- **v0.13.0**: Batch inference max_tokens capped at 16384, trust_remote_code warning
- **v0.14.0**: Plaintext loader UTF-8 encoding, MoE config validation (moe_aux_loss_coeff ge=0, moe_lora boolean only)
- **v0.14.3**: Data validate auto-detects format, Web UI `--show-token` flag + auth token documented
- **v0.15.0**: `rope_scaling_type` Literal constraint, `max_length` bounds (ge=64, le=1048576), FSDP config key allowlist, Liger Kernel exception handling narrowed
- **v0.16.0**: `embedding_loss` Literal constraint, `embedding_margin` gt=0 validation, ONNX export without trust_remote_code (with warning), TensorRT export subprocess list args (no shell injection), speculative decoding SSRF-protected (URL blocked) with warning panel, vLLM speculative model URL validation
- **v0.17.0**: Server data generation provider SSRF validation (scheme whitelist + localhost-only HTTP), audio model trust_remote_code warning panel, audio file path traversal protection (resolved paths confined to audio_dir), SGLang backend trust_remote_code warning panel
- **v0.18.0**: Ollama deploy GGUF path traversal protection + `.gguf` extension validation, model name validation (no path separators/null bytes), subprocess list args (no shell injection), Modelfile parameter key allowlist + value newline/null sanitization, overwrite warning panel
- **v0.19.0**: Custom eval JSONL schema validation + 10k task cap, regex scoring ReDoS guard, judge API SSRF protection + API key isolation, human eval local-only terminal UI + 10k prompt cap, leaderboard read-only SQLite queries
- **v0.20.0**: Ollama provider localhost-only validation (remote blocked), Anthropic provider API key from env only (never CLI arg), vLLM provider SSRF protection (scheme whitelist + localhost-only HTTP), output path traversal protection (`..` blocked), configurable rate limiting (`--requests-per-minute`)
- **v0.21.0**: Migrate input/output path traversal protection (resolve + relative_to(cwd)), Unsloth .ipynb AST-only parsing (no exec/eval), recipes output path traversal protection, NEFTune config bounded (ge=0.0, le=50.0)
- **v0.22.0**: Multi-adapter serving path traversal protection (resolve + relative_to(cwd)), adapter name validation (alphanumeric + hyphens only), unknown adapter returns 404 (not 500)
- **v0.23.0**: AWQ/GPTQ calibration data path traversal protection (resolve + relative_to(cwd)), AWQ/GPTQ output path stays under cwd, curriculum_buckets bounded (ge=1, le=20), AWQ/GPTQ trust_remote_code warning panel
- **v0.24.0**: HF download trust_remote_code=False + warning panel, HF download output path sanitized (Path.name), download --samples capped at 1M, dataset registry name validation (no path separators/null bytes), registry path traversal protection, loss_watchdog threshold le=100 + patience le=1000, freeze_layers le=1000
- **v0.24.1**: AWQ/GPTQ output path traversal validation moved before import check (previously unreachable when autoawq/auto-gptq not installed), Windows Unicode fix for Rich console output (replaced non-ASCII symbols with ASCII equivalents)
- **v0.24.2**: Chat proxy SSRF protection (localhost-only HTTP, HTTPS for remote), chat proxy max_tokens capped at 16384 + temperature/top_p bounded, chat proxy Bearer token auth required, XSS prevention (HTML-escape before markdown render), runs compare max 5 runs, config from-form validates via load_config_from_string, SSE read endpoints no auth (GET)
- **v0.25.0**: Tool-calling JSON-only parsing (no eval), RLVR math_verify regex-extracted numerics (no eval), code_exec 5s timeout + 512MB RLIMIT on POSIX + ephemeral cwd + socket patch + `python -I -S` + 10KB output cap, verifiable_domain Literal constraint, LoRA PEFT mutual exclusion (DoRA/VeRA/OLoRA), data augment path containment + caps, forgetting_detection bounds, checkpoint_intelligence bounds + symlink refusal, autopilot path containment (realpath + commonpath) + goal Literal + GPU/time budget bounds, MLX trainers no trust_remote_code
- **v0.26.0 — Registry**: name/tag validation (alphanumeric + `_-.` only, null-byte rejected, name ≤128 / tag ≤64 chars), artifact path containment (default `enforce_cwd=True` via `os.path.realpath + commonpath`, stored path is realpath), SQL LIKE wildcard escaping (`%` and `_` escaped with `ESCAPE '\\'` in `search()` and prefix `resolve()`), DB 600 perms on POSIX, lineage indirect-cycle detection (BFS ancestor walk before insert), CLI Rich markup escaped everywhere, `resolve()` raises `AmbiguousRefError` on ambiguous prefix (no silent None)
- **v0.26.0 — Eval Gate**: suite path via shared `utils/paths.is_under_cwd` containment, `regression_threshold` [0.0, 1.0], `every_n_epochs` [1, 100], `on_regression` Literal ("stop"/"warn"/"continue"), `GateTask.tasks`/`prompts` null-byte rejection, `judge_model` URL scheme allowlist (`ollama://`, `https://`, `http://localhost`/`http://127.0.0.1`) — SSRF hardening, callback fails-safe: structured errors treated as regressions under `on_regression="stop"`
- **v0.26.0 — Trace-to-Preference**: input/output path containment via shared `is_under_cwd`, trace line cap 100,000, `--format`/`--signal` Literal validation, PII warning panel before every run, JSON-only parsing (no eval), malformed JSON lines skipped silently
- **v0.26.0 — Quant-Lobotomy**: `--before`/`--after`/`--tasks` all containment-checked, `registry://` refs support optional `kinds` filter to avoid picking the wrong artifact, format Literal validated
- **v0.26.0 — Soup Cans**: Manifest format version pinned to 1; name alphanumeric+`_-.`; author max 128 chars, no null bytes/newlines; created_at must parse via `datetime.fromisoformat`; description max 4096; DataRef URL HTTPS-only; hf_dataset regex-validated; tar extraction uses `filter="data"` on Python 3.12+, fallback only on `TypeError`/`AttributeError` (not `TarError`); manual symlink/hardlink rejection + `commonpath` check; 100 MB size cap on pack + fork; dunder-key (`__*__`) and null-byte rejection in fork modifications to prevent prototype pollution; inspect/read_config refuse paths outside cwd
- **v0.27.0 — Multi-GPU Mastery**: `--gpus` bounds (reject bool, non-digit, zero, negative, values above `MAX_GPU_COUNT=128`); `--gpus auto` on 0-GPU host prints explicit yellow warning (no silent no-op); Rich markup escaped on `--config` path before embedding in the multi-GPU advice Panel; `accelerate launch` argv assembled via `shlex.quote` per element (copy-pasted command safe against crafted paths); `build_accelerate_argv` validates `num_processes >= 1`, `mixed_precision` Literal (`no/fp16/bf16/fp8`), `num_machines` bounded `[1, 256]`; ZeRO++ integer literals (`int(1e9)` not float) so DeepSpeed strict JSON validator accepts; `validate_fsdp2_compile_config` requires FSDP + CUDA + transformers + torch>=2.2/accelerate>=0.27; DeepSpeed-MII stub exits non-zero to prevent silent mis-start; `validate_pipeline_config` enforces `pipeline_stages >= 2` + CUDA + `gpu_count >= stages`; `pipeline_stages` Pydantic bounds `[1, 16]`; `parallelism` Literal `data|pipeline`; NCCL env (`NCCL_P2P_DISABLE`/`NCCL_IB_DISABLE`/`NCCL_NVLS_ENABLE`) applied via `os.environ.setdefault` only — user/launcher overrides are never stomped
- **v0.28.0 — Training Speed & Memory**: `quantization_aware: Union[bool, Literal["fp8"]]` rejects arbitrary strings (only `true` / `false` / `"fp8"`); FP8 path requires CUDA + Hopper+ SM capability + transformers backend; `gradient_checkpointing: Union[bool, Literal["selective","medium","full","auto"]]` rejects unknown tier strings and returns only HF-supported keys (no private markers leak into `TrainingArguments.gradient_checkpointing_kwargs`); `activation_offloading` Literal `cpu|disk`, scratch `save_dir` containment-enforced via shared `utils/paths.is_under_cwd` before disk writes, `torch.load(weights_only=True)` prevents arbitrary Python deserialization on reload, TOCTOU closed between `mkstemp` and `torch.save` by holding the fd open, best-effort cleanup on context exit (handles SIGKILL mid-backward); `kernel_picker.pick_best_kernel` raises `ValueError` when all candidates lack a finite `time_ms` (prevents silent promotion of an untimed combo); Cut CE architecture detector matches on last path component only (so `deepseek-ai/...-phi-...` org-prefix does not trigger a Phi patch on a DeepSeek model); `build_cross_doc_mask` numpy-vectorised to avoid O(seq_length²) pure-Python fill at `max_length` bound (1M); `@model_validator` requires `packing=true` when `packing_cross_doc_attn_mask=true` (prevents silent no-op); `SoupConfig._validate_v028_speed_memory_sft_only` rejects `use_cut_ce`/`quantization_aware="fp8"`/`kernel_auto_compose`/`activation_offloading` on non-SFT tasks — prevents legacy int8-QAT wrapper from crashing on the string `"fp8"` and prevents silent no-ops on DPO/GRPO/KTO/etc. (multi-trainer wiring tracked for v0.28.1)
- **v0.29.0 — HF Hub Deep Integration**: `HF_ENDPOINT` SSRF-hardened — scheme allowlist (http/https), null-byte rejection, `0.0.0.0` explicitly rejected, plain-HTTP only permitted for loopback (`localhost`/`127.0.0.1`/`::1`), RFC1918 / link-local / cloud-metadata (169.254.x) IPs rejected via `ipaddress.ip_address`; repo ID regex `[A-Za-z0-9][A-Za-z0-9._-]{0,95}` per component, ≤200 chars total, null-byte / whitespace / `..` / leading-`/` rejection (applied to `push --repo`, `train --push-as`, `data push --hf-dataset`, `deploy hf-space --model/--space`); collection slug `owner/slug-hash` regex-validated, ≤256 chars; HF token resolution single-sourced in `utils/hf.resolve_token` (env > cached login), explicit non-printable tokens rejected, `push --token` flag deprecated with yellow warning; `soup push --model` confined to cwd via `is_under_cwd` (prevents crafted `soup.yaml output:` from uploading system files); auto-push checkpoint `allow_patterns` restricts uploaded files to `*.safetensors`/`*.bin`/`*.pt`/`*.json`/`tokenizer*`/`trainer_state.json`/`training_args.bin`/`README.md` (keeps `.env` and source files out of auto-pushed branches); `prepare_hf_resume` enforces cwd containment and passes `local_dir_use_symlinks=False` (defeats symlink-based FS escape on older `huggingface_hub`); commit messages stripped to first line and capped at 200 chars (prevents multi-line injection into public HF commit history); `_render_eval_scorecard` neutralises `|`/`[`/`]`/`(`/`)`/`!`/newlines/tabs/`<`/`>` in task names and non-numeric scores; `data_lineage` HTML-escaped (defeats XSS on HF Hub README viewer); `render_space_template` validates `model_repo` via `validate_repo_id` before substitution into rendered `app.py` (crafted repo id cannot inject Python code); `HFPushCallback` uses sticky `_repo_failed` flag to short-circuit retries after hard failure (no log spam, no wasted API calls); `add_to_collection` prefers HfHubHTTPError 409 detection over string-match for duplicate handling
- **v0.30.0 — Inference Excellence**: OTLP endpoint SSRF-hardened matching v0.29.0 `HF_ENDPOINT` (scheme allowlist, `0.0.0.0` rejected, RFC1918 / link-local / cloud-metadata via `ipaddress.ip_address`, plain HTTP loopback-only); `pick_draft_model` rejects URL-scheme target names (`http://`/`https://`/`file://`), null bytes, names >200 chars; `validate_regex_pattern` length-capped at 2048 + null-byte rejection + must compile; `validate_json_schema` dict shape + 64KB serialised cap + required `type` field; `--json-schema` file path confined to cwd via shared `utils/paths.is_under_cwd`; `--structured-output json` requires `--json-schema` (fail-fast prevents silent no-op); FastAPI `/v1/adapters/activate/{name}` pattern `^[a-zA-Z0-9][a-zA-Z0-9\-]*$` enforced before handler runs; activate/deactivate state protected by `threading.Lock` (no race on concurrent hot-swap); `/v1/adapters` response omits filesystem paths (names + active flag only); CORS on transformers backend restricted to loopback origins (`^https?://(localhost|127\.0\.0\.1)(:\d+)?$`) since hot-swap endpoints mutate state without auth; `auto_quant.Candidate` name regex + score `[0.0, 1.0]` + finite-float check + non-negative latency; `pick_best` generator-safe (materialises to list) so error-message count is accurate; OTel span context uses `contextlib.ExitStack` so `__exit__` sees real exception info (spans correctly marked error on HTTPException); `record_latency` always runs in `finally` so tail-latency percentiles include failure paths; `build_tracer` idempotent — only installs provider when current is `ProxyTracerProvider`/`NoOpTracerProvider` (preserves operator-supplied instrumentation)
- **v0.32.0 — Training Stability & Auto-Tuning**: `--find-lr-output` containment via shared `utils/paths.is_under_cwd` (prevents writes outside cwd); `save_lr_finder_report` rejects NaN / Infinity floats in `lrs` / `losses` and serialises with `allow_nan=False` (keeps the report parser-safe); `compute_lr_schedule` rejects non-positive `start_lr`, inverted ranges, and `num_steps` outside `[2, 10_000]`; `pick_mixed_precision` rejects empty / null-byte / >200-char model names and resolves multi-version quirks (`qwen2.5` vs `qwen2`, `phi-3.5` vs `phi-3`) by longest-substring-first iteration so an added family can never accidentally make a more-specific entry dead code; `compute_warmup_steps` clamps to `[10, 1000]` with a `ratio==0.0` short-circuit matching HF Trainer's "no warmup" convention; `SpikeRecoveryStrategy` is `@dataclass(frozen=True)` (post-construction mutation cannot bypass validation), `max_attempts ∈ [1, 10]`, `lr_decay ∈ (0, 1)`, `min_lr > 0`; cross-validator `_validate_spike_recovery_requires_watchdog` rejects `loss_spike_recovery=true, loss_watchdog=false` at config-load (fails fast instead of never triggering); `convergence_window ∈ [5, 10_000]`, `convergence_rel_tol ∈ (0, 1]`, `recommend_action` reuses `detect_plateau` so plateau heuristic stays single-source-of-truth; `GradAccumMonitor.recommend()` caps doubled `accum` at `MAX_ACCUM=1024` so a runaway advisory loop cannot blow up DataLoader prefetch; `generate_config` validates BOTH the YAML output path AND the embedded `decisions["output"]` field via `is_under_cwd` (closes the gap where a crafted `decisions["output"]="../../etc"` would have silently propagated into the rendered YAML)

## Security Scanning

- All code is scanned with `ruff` for style and common issues
- Dependencies are regularly updated to patch known CVEs
- GitHub's dependency scanning alerts us to vulnerable dependencies
- We use GitHub Actions CI/CD for continuous integration

## Dependency Updates

We actively monitor and update dependencies:

- Major dependency updates: Tested in PR before merging
- Security patches: Applied immediately and released as patch versions
- Deprecated dependencies: Replaced proactively

## Coming Soon

- [x] Automated dependency scanning
- [ ] SBOM (Software Bill of Materials) for each release
- [ ] Third-party security audit (after 1.0.0 release)

## Questions?

If you have security questions (not vulnerability reports) or need clarification:

- Open a GitHub Discussion tagged `security`
- Open a [GitHub Issue](https://github.com/MakazhanAlpamys/Soup/issues) (non-vulnerability inquiries)
- Check our [CONTRIBUTING.md](CONTRIBUTING.md) for general support

## License

This Security Policy is provided under the Apache-2.0 license, same as the Soup project.

---

**Last Updated**: April 2026

For the latest version of this policy, visit: https://github.com/MakazhanAlpamys/Soup/blob/main/SECURITY.md
