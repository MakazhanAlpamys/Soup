# Command Reference

[← Back to the Soup README](../README.md)

> The full `soup` command list.

## All Commands

```
soup init [--template chat|code|...|audio]       Create config
soup autopilot --model <id> --data d.jsonl --goal <g>  Zero-configsoup train --config soup.yaml                 Start training
soup train --config soup.yaml --tensorboard   Train with TensorBoard logging
soup train --config soup.yaml --fsdp full_shard  Train with FSDP2
soup train --config soup.yaml --deepspeed zero++  DeepSpeed ZeRO++ (quantized comms)
soup train --config soup.yaml --gpus auto|N      Multi-GPU launch hint
soup train --config soup.yaml --gate evals/gate.yaml  Eval-gated training
soup train --config soup.yaml --push-as user/repo  Auto-push each checkpoint to HF as branch
soup train --config soup.yaml --push-as user/repo --hf-resume  Resume from latest HF checkpoint branch
soup train --config soup.yaml --find-lr        LR range finder: write recommended LR JSON
soup infer --model ./output --input p.jsonl   Batch inference
soup chat --model ./output                    Interactive chat
soup push --model ./output --repo user/name   Upload to HuggingFace
soup push --model ./output --repo user/name --collection user/coll-abc123  Add to HF Collection
soup merge --adapter ./output                 Merge LoRA with base model
soup merge-sharded-fsdp-weights ./shards -o merged.safetensors  Consolidate FSDP shards into one safetensors (v0.71.14; --plan-only previews)
soup export --model ./output --format gguf    Export to GGUF (Ollama)
soup export --model ./output --deploy ollama  Export GGUF + auto-deploy to Ollama
soup export --model ./output --format onnx    Export to ONNX
soup export --model ./output --format tensorrt Export to TensorRT-LLM
soup export --model ./output --format awq     Export to AWQ (4-bit)
soup export --model ./output --format gptq    Export to GPTQ (4-bit)
soup deploy ollama --model m.gguf --name x    Deploy GGUF to Ollama
soup deploy ollama --list                     List Soup-deployed models
soup deploy ollama --remove <name>            Remove model from Ollama
soup deploy hf-space --model user/m --space user/s --template gradio-chat|streamlit-chat  Create HF Space
soup deploy autopilot --target mac-m3|rtx-4090-24gb|...  Pick PEFT+quant+spec-decoding for a hardware target
soup deploy autopilot --list                  List all 10 deploy profiles
soup agent synth --spec api.yaml -o ds.jsonl  Parse OpenAPI/MCP/GraphQL spec into a tool-calling SFT dataset
soup agent train --spec api.yaml --base model  One-shot synth + planned soup train invocation
soup agent eval --spec api.yaml --predictions p.jsonl  Score predicted tool-calls vs spec catalog
soup eval benchmark --model ./output          Evaluate on standard benchmarks
soup eval custom --tasks eval.jsonl           Custom eval tasks from JSONL
soup eval judge --target resp.jsonl           LLM-as-a-judge evaluation
soup eval auto --config soup.yaml             Auto-eval from config
soup eval compare <run1> <run2>               Compare eval results
soup eval leaderboard                         Local model leaderboard
soup eval human --input p.jsonl               Human A/B evaluation
soup eval gate --suite gate.yaml              Run eval-gate suite standalonesoup eval quant-check --before X --after Y --tasks t.jsonl  Before/after quantsoup serve --model ./output --port 8000       OpenAI-compatible API server
soup serve --model ./output --backend vllm    vLLM backend (2-4x throughput)
soup serve --model ./output --backend sglang  SGLang backend
soup serve --model ./output --backend mii     DeepSpeed-MII backend (live)
soup serve --model ./output --speculative-decoding draft-model  Speculative decoding
soup serve --model <m> --auto-spec            Auto-pair draft model for speculative decoding
soup serve --model <m> --backend vllm --prefix-cache  vLLM prefix caching (RAG/agent)
soup serve --model <m> --structured-output json --json-schema s.json  Constrained output
soup serve --model <m> --structured-output regex --regex-pattern '...'  Regex-constrained output
soup serve --model <m> --dashboard            Live dashboard + /metrics endpoint
soup serve --model <m> --trace --trace-endpoint http://localhost:4317  OpenTelemetry tracing
soup serve --model <m> --trace-log ./serve.jsonl  Per-request JSONL log + rotation + secret redaction
soup serve --model <m> --record-thumbs ./rl.db  Capture 👍/👎 feedback into local-RL SQLite + POST /v1/thumbs (transformers)
soup serve --model <m> --kv-cache-type bf16|f16|q8_0|fp8  KV-cache type (transformers; q8_0 needs hqq; fp8 = vLLM+Hopper only) (v0.71.14)
POST /v1/adapters/activate/<name>             Hot-swap active LoRA adapter
soup sweep --config soup.yaml --param lr=...  Hyperparameter search
soup diff --model-a ./a --model-b ./b         Compare two models
soup data inspect <path>                      View dataset stats
soup data validate <path>                     Check format (auto-detect)
soup data convert <path> --to chatml          Convert between formats
soup data merge data1.jsonl data2.jsonl       Combine datasets
soup data dedup <path> --threshold 0.8        Remove duplicates (MinHash)
soup data stats <path>                        Extended statistics
soup data generate --prompt "..." --count 100 Generate synthetic data
soup data generate ... --provider ollama      Use local Ollama instance
soup data generate ... --provider anthropic   Use Claude API
soup data generate ... --provider vllm        Use local vLLM server
soup data generate ... --template code        Domain templates (code/conversation/qa/preference/reasoning)
soup data generate ... --quality-pipeline     Auto validate + filter + dedup
soup data augment <path> --strategy rephrase|translate|style [--provider ollama|vllm --model <m> --base-url <url>]  LLM-driven augmentation
soup data from-traces --logs l.jsonl --format langchain --signal thumbs_up --output p.jsonl  Preference pairs from traces
soup data from-traces ... --judge --min-confidence 0.7  LLM-judge confidence filter
soup data review prefs.jsonl --sample 10      Preview preference pairssoup data filter <path> --coherence 0.3       Quality filter (perplexity/coherence)
soup data sample <path> --n 1000             Random sample subset
soup data sample <path> --n 1000 --strategy diverse  Cluster-based diverse sampling
soup data sample <path> --n 1000 --strategy hard     Sample hardest examples
soup data sample <path> --pct 10             Sample by percentage
soup data split <path> --val 10 --test 10    Split into train/val/test
soup data split <path> --val 500 --absolute  Split with absolute counts
soup data split <path> --val 10 --stratify category  Stratified by field
soup data search "code instructions"         Search HuggingFace Hub for datasets
soup data search --sort likes --limit 10     Sort and paginate search results
soup data preview teknium/OpenHermes-2.5     Preview remote dataset metadata
soup data download user/dataset -o data.jsonl  Download HF dataset as JSONL
soup data download user/ds --samples 1000    Stream first 1000 samples
soup data register --name my-ds --path d.jsonl --format alpaca  Register dataset
soup data unregister --name my-ds            Remove from registry
soup data push --input d.jsonl --hf-dataset user/name  Upload local JSONL as HF dataset
soup data push --input d.jsonl --hf-dataset u/n --hub modelscope|modelers  Upload to an alternative hub
soup data registry                           List all registered datasets
soup data demo                                List bundled demo JSONL fixtures
soup data demo alpaca_demo --output ./d.jsonl Copy a bundled demo JSONL fixture
soup data forge --docs ./docs --task sft --target-rows 1000  Synthetic data pipeline + provenance
soup data forge --docs ./docs --hub modelscope --teacher owner/name  Pre-fetch the teacher from an alternative hub
soup data score --input rows.jsonl            Composite quality scorecard (PII + toxicity + lang + edu)
soup data decontaminate --input rows.jsonl --benchmarks mmlu,gsm8k  Drop benchmark-overlap rows
soup data toxicity --input rows.jsonl -o tox.jsonl  Flag toxic rows (keyword baseline)
soup data langdetect --input rows.jsonl -o tagged.jsonl  Tag each row with language code
soup data pii --input rows.jsonl -o pii.jsonl Flag rows containing email/phone/SSN/credit-card
soup data educational --input rows.jsonl -o scored.jsonl  Score educational value per row
soup train --config soup.yaml --tracker mlflow  MLflow / SwanLab / Trackio integration
soup profile --config soup.yaml              Estimate memory/speed before training
soup profile --config soup.yaml --gpu a100   Estimate for specific GPU
soup profile --config soup.yaml --json       Machine-readable output
soup cost --config soup.yaml                 Estimate training cost in USD across providers
soup cost --config soup.yaml --gpu H100      Estimate training cost for specific GPU
soup adapters list ./output/                 Scan for LoRA adapters
soup adapters info ./output/checkpoint-500/  Show adapter metadata
soup adapters compare adapter1/ adapter2/    Compare two adapters
soup loop init <model> --eval <s> --baseline <b> [--pre-wired]  Create .soup/loop.yaml (data flywheel; --pre-wired = real stages)
soup loop status                              Counters + status + pre_wired flag
soup loop watch [--detach] [--max-iter N] [--pre-wired] [--pack-cans]  Harvest → train → gate → deploy daemon (pre-wired stages + Soup Can packing)
soup loop pause / soup loop resume           Atomic status flip
soup loop canary <adapter> --traffic 5%      Promote canary + auto-rollback on MAJOR
soup loop replay [<iter-id>] [--extract <dir>]  Replay / unpack a recorded iteration manifest
soup serve --model m --adapters chat=./c code=./d  Multi-adapter serving
soup migrate --from llamafactory config.yaml  Import config from LLaMA-Factory
soup migrate --from axolotl config.yml        Import config from Axolotl
soup migrate --from unsloth notebook.ipynb    Import config from Unsloth notebook
soup migrate --from llamafactory c.yaml --dry-run  Preview without writing
soup recipes list                             List all 43 ready-made recipes
soup recipes show llama3.1-8b-sft            Print recipe YAML
soup recipes use llama3.1-8b-sft             Copy recipe to soup.yaml
soup recipes search "reasoning"              Search by keyword/task/size
soup registry push --run-id <id> --name n --tag v1  Register runsoup registry list [--name n] [--tag v1]     List registry entriessoup registry show <ref>                      Entry details + artifacts + ancestors
soup registry diff <a> <b>                    Side-by-side config + eval delta
soup registry search "medical"                Search name/base/task/notes
soup registry promote <ref> --tag prod        Tag an entry (e.g. promote to prod)
soup registry delete <ref> --yes              Remove entry (cascades)
soup history <name>                           Lineage DAG tree for a namesoup can pack --entry-id <id> --out r.can     Pack registry entry as .cansoup can inspect r.can                        Preview manifest without extracting
soup can verify r.can                         Verify schema + config parseability
soup can fork r.can --out fork.can --modify training.lr=5e-5  Fork + re-pack
soup can run r.can --yes [--deploy] [--env-capture env.txt]  Run a .can end-to-end
soup can publish r.can --hf-hub user/name    Publish .can to HF Hub as dataset
soup runs                                     List training runs
soup runs show <run_id>                       Run details + loss graph + cost
soup runs compare <run_1> <run_2>             Compare two runs
soup runs replay <run_id>                     Replay summary + loss curve from history (also plots a benchmark-score curve when the metric lives in eval_results)
soup why [run_id]                             Explain training anomalies (heuristic)
soup tui                                      Full-screen Textual dashboard (requires [tui] extra)
soup train --config soup.yaml --profile       Record torch.profiler trace to <output>/profiles/
soup --log-level quiet|normal|verbose|debug   Global logging tier (Rich-formatted)
soup ui [--port 7860]                         Web UI (experiments, training, data)
soup ui --public [--auth-token T]             Phone-scannable Web UI (v0.53.9)
soup tokenizer train --input c.jsonl --vocab-size N  Train BPE tokenizer (v0.53.9)
soup bench <model> --p50 --p95                Bench with tail-latency percentiles (v0.53.9)
soup bench <model> --backend auto             Auto-detect transformers/mlx backend (v0.53.9)
soup serve --reasoning-parser deepseek-r1     Strip <think> blocks from responses (v0.53.9)
soup doctor [--nccl]                          Check environment (optionally check NCCL bandwidth)
soup quickstart [--dry-run]                   Full demo
soup adapters scan <adapter>                  Spectral backdoor scan (rank-1 dominance + outlier detection)
soup adapters sign <adapter> [--backend unsigned|ed25519] [--key <pem>|--generate-key <pem>]  Merkle manifest + ed25519 sign
soup adapters verify <adapter> [--strict] [--public-key <pem>]  Verify manifest + ed25519 signature
soup adapters check-safetensors <adapter> [--strict]  Refuse pickle / PyTorch-classic weights
soup adapters merge ... [--license <id>] [--license-override <reason>] [--allow-unscanned]  License + backdoor-scan gates (auto-detect license; scan FAIL refused)
soup attest emit ... [--sign ed25519 --key <pem>] [-o att.json]  in-toto/SLSA-3 attestation (+ .sig sidecar)
soup attest verify <statement> --signature <sig> [--public-key <pem>]  Verify ed25519 attestation signature
soup airgap-bundle --model <m> --output <out.tar> [--repro-receipt <r.json>]  Signed tarball for data-diode transfer (embeds repro-receipt)
soup train --config soup.yaml --annex-xi <out.md|out.pdf>  EU AI Act Annex XI/XII doc (markdown or PDF; top_domains auto-filled)
soup train --config soup.yaml --track-energy [--energy-country USA]  codecarbon offline kWh/CO2 → annex-xi (pip install soup-cli[carbon])
soup train --config soup.yaml --track-energy --energy-out <energy.json>  persist measurement for `soup bom emit --energy <energy.json>`
soup train --config soup.yaml --repro-receipt <out.json>  SR 11-7 reproducibility receipt
soup can pack --entry-id <id> --out r.can --attest <statement.json>  Embed in-toto Statements into a v3 can manifest
soup audit-log tail / rotate  Tail / rotate the per-command HIPAA/SOC2 audit log (~/.soup/audit.jsonl)
soup --no-audit-log <cmd> / SOUP_NO_AUDIT_LOG=1  Opt out of the per-command audit line
soup eval unlearning <run-id> --benchmark tofu|muse|wmdp  Forget Quality + Model Utility + PrivLeak verdict
soup edit set --base <m> --method rome|memit|alphaedit|grace --subject "..." --target "..." [--output <dir>] [--device cpu] [--governor/--no-governor] [--registry-id <id>]  Live surgical knowledge edit (--plan-only available)
soup edit diff <before-run> <after-run> --probes p.jsonl [--before-model <m> --after-model <m>]  Knowledge-injection diff (live before/after generation when both models given)
soup train --task unlearn  NPO/SimNPO/RMU unlearning from data.forget_set (+ optional data.retain_set)
soup train  # data.format='raft'  Answer-only span-mask RAFT training (golden+distractor docs, [doc-N] citations); generator-stage configs auto-link the latest RA-DIT retriever
soup ra-dit --retriever-config <r.yaml> --generator-config <g.yaml> [--retriever-model <m>] [--plan-only]  One-shot two-stage RA-DIT: train retriever → record pairing → train generator
soup eval citation <data> [--style bracket|inline|footnote] [--shuffle-seed N] [--output o.json]  Citation precision/recall/F1 over predictions or RAFT rows
soup steer train --base <m> --method caa|iti|repe --name <id> --pairs <jsonl>  Fit a CAA/ITI/RepE activation-steering vector from {positive, negative} pairs
soup steer apply --name <id> --strength <s>  Preview a stored steering vector; soup steer list lists them
soup serve --steer <name> [--steer-strength <s>]  Apply a steering vector at decode time via a forward hook (transformers backend)
soup serve --bank <bank.json> [--bank-strength <s>]  Multi-tenant VeRA/VB-LoRA serving; active user per request via X-User-Id header (v0.71.12)
soup ingest --source langfuse|langsmith|helicone|openpipe|otel|openai-stored --logs <jsonl>  Universal trace importer (6 SaaS adapters → normalised JSONL)
soup prune-prompt --input <jsonl> --output <jsonl> --min-frequency 0.95  Detect + strip shared system-prompt prefix
soup prune-prompt ... --tokenizer <id-or-path>  Tokenizer-aware prefix detection (decodes remaining ids, boundary-safe)
soup data active-sample --input <jsonl> --output <jsonl> --budget N  Top-N uncertain prod traces for human review
soup ab --input <jsonl> --metric latency|judge_score|retry_rate  mSPRT sequential A/B (decision: continue / reject_h0 / accept_h0)
soup ingest|prune-prompt|ab|data active-sample ... --slack-url <https> | --discord-url <https>  Shared SSRF-validated webhook on completion
soup drift-alarm --reference <jsonl> --live <jsonl> --threshold 0.2  Rolling-KL drift alarm (exit 3 on drift)
soup drift-alarm ... --slack-url <https> | --discord-url <https>  Optional SSRF-validated webhook on drift detected
soup tunability --list                                   List built-in candidate-base catalogue
soup tunability --dataset <jsonl> [--candidates a,b,c]   Probe candidate bases + Pareto frontier report
soup tunability --dataset <jsonl> --live [--device cpu]  LIVE per-candidate LoRA probe (loads each repo)
soup plan --config soup.yaml                             Pre-flight summary + write soup.tfstate
soup apply --config soup.yaml [--dry-run]                Lock-and-execute; refuses on drift (exit 3)
soup env lock | status | check                           Hermetic env lockfile + ABI drift detection (exit 3)
soup env fix [--format uv-pip|requirements] [--output req.txt]  Render a reproducible install plan from soup-env.lock (print-only)
soup completions bash | zsh | fish                       Shell completion script (sourceable via eval)
soup license-advisor --target b2c|defense|embedded       Recommend license-clean base for deploy target
soup license-advisor ... --license <id> --mau N          Per-license downstream-risk check (exit 3 on block)
soup probe sae-diff <sae> <pre.json> <post.json> [--top-k N]  SAE feature diff between pre/post-FT activations (v0.66.0)
soup probe sae-diff <repo> <pre.json> <post.json> --auto-download  Fetch an allowlisted SAE into ~/.soup/sae-cache (v0.71.8)
soup probe sleeper <base> [--evidence ev.json] [--weights w.npz] [--output o.json]  Sleeper-agent defection probe; --weights = real calibrated probe (v0.66.0; v0.71.8)
soup probe truth <base> [--evidence ev.json] [--weights w.npz] [--output o.json]  TruthfulQA-style honesty probe (v0.71.8)
soup probe harm <base> [--evidence ev.json] [--weights w.npz] [--output o.json]  HarmBench-style misuse probe (v0.71.8)
soup probe interference <losses.json> [--output o.json]  Pairwise N×N adapter interference matrix (exit 2 on MAJOR; v0.66.0)
soup probe interference --measure <eval.jsonl> --base-model <m> --adapter name=path ... [--device cpu]  Auto-measure live interference (v0.71.8)
soup probe pack <base> [--output o.json]      Per-base calibrated probe pack manifest (v0.66.0; +truth/harm v0.71.8)
soup probe pack --list                        List bundled probe-pack bases (v0.66.0)
soup train --capture-activations <layer> --capture-prompts <jsonl>  Post-train SAE-diff-ready per-token activation snapshot (v0.71.8)
soup adapters blame ... --top-k 50            Live DataInf-style influence runner (v0.66.0, closes #171)
soup adapters merge ... --strategy cmaes --eval <s> --budget 1h  CMA-ES evolutionary merge — live loop (v0.67.0 schema / v0.71.4 live)
soup adapters merge ... --canary <suite.json> [--strict-verdict]  Live OK/MINOR/MAJOR canary verdict, exit 2 on MAJOR (v0.71.4)
soup adapters pr <title> --base-sha <hex> --adapter <path>  GitHub-shaped adapter PR Markdown / JSON (v0.67.0)
soup adapters pr <title> ... --push owner/repo#N  Post the PR as a GitHub comment via gh api (v0.71.4)
soup adapters branch <name> --from-registry <id> | --attach-to-registry <id>  Branch ↔ Registry lineage (v0.71.4)
soup adapters bisect <ckpt>... --eval-command "..."  Binary search over training history (v0.67.0)
soup lock write --base-sha <h> --dataset-sha <h> --env-hash <h>  Write soup.lock (v0.67.0)
soup lock write --base-sha <h> --dataset-sha <h> --env-lock soup-env.lock  Auto-derive --env-hash from soup-env.lock (v0.71.1)
soup lock show / soup lock check              Show + drift-check (exit 3 on drift)
soup compile <program.py> --eval <suite> [--optimizer mipro|gepa|textgrad|copro|bootstrap_fewshot] [--plan-only]  DSPy / GEPA / TextGrad prompt-program compiler — live (v0.71.13; pip install 'soup-cli[compile]')
soup distill-prompt --traces <jsonl> --teacher <m> --student <m> --strategy sft|preference|kl [--provider ollama|anthropic|vllm] [--base-url <url>] [--temperature F] [--max-rows N]  Distill prompt-heavy traces via a live teacher (v0.71.13)
soup compile-tools <spec.json|yaml> --eval <jsonl> [--optimizer textgrad|gepa] [--plan-only]  TextGrad / GEPA tool-schema optimiser — live (v0.71.13; pip install 'soup-cli[compile]')
soup apple-adapter <source-dir> --direction hf-to-mlx|mlx-to-hf|hf-to-apple|mlx-to-apple --output <dir> [--sign]  HF / MLX / Apple FoundationModels adapter conversion (v0.68.0)
soup local-rl init --db <path>                Create personal-LLM flywheel SQLite schema (v0.68.0)
soup local-rl status --db <path>              Print interactions / thumbs-up / thumbs-down counters
soup local-rl record --db <path> --prompt <q> --response <r> --thumb up|down  Append thumbs record
soup local-rl harvest --db <path> -o <pairs.jsonl>  Harvest DPO pairs from thumbs into JSONL
soup local-rl train --db <path> --model <id> --once [--train-method dpo|kto|orpo] [--min-pairs N] [-o <dir>]  Ad-hoc DPO/KTO/ORPO train from harvested thumbs — live (v0.71.13)
soup local-rl train --db <path> --model <id> [--scheduler-dir <dir>] [--hour H] [--minute M]  Render a systemd/launchd nightly-train scaffold (no --once) (v0.71.13)
soup build <manifest.yaml> [--dry-run] [--output-dir <dir>]  dbt-for-SFT DAG: validate + plan + live materialise (v0.69.0; live v0.71.6)
soup expect <data.jsonl> <suite.yaml>         Expectations suite: PII / token-length / refusal / judge (v0.69.0)
soup data gen-magpie --base <m> --provider ollama|vllm --target N --output <jsonl> [--base-url <url>] [--quality-filter]  Magpie synthetic generator — live (v0.69.0; live v0.71.6)
soup data persona-mix --prompts <jsonl> --n N --output <jsonl>  Persona-Hub diversity sampler (v0.69.0)
soup data brain-rot <data.jsonl> [--strict]   Brain-rot detector — arXiv 2510.13928 (v0.69.0)
soup iterative-dpo --base-model <m> --reward-model <rm> --prompts <p.jsonl> --output-dir <out> --rounds N --pairs-per-round N [--plan-only]  Iterative DPO loop driver — LIVE sample→score→pair→train (v0.70.0; live v0.71.11)
soup train --reward-hack-detector info_rm|rm_ensemble [--reward-hack-halt]  Reward-hacking detector for GRPO — LIVE callback (v0.70.0; live v0.71.11)
soup train --uld-strategy wasserstein|topk_align [--uld-top-k N]  Cross-tokenizer ULD on task='distill' — LIVE W1/topk loss (v0.70.0; live v0.71.11)
soup train --minillm-enabled [--minillm-teacher-mix-ratio 0.3]  MiniLLM reverse-KL distillation — LIVE (v0.70.0; live v0.71.11)
soup train --rl-checkpoint-save-every-steps N [--rl-checkpoint-keep-last N]  Mid-epoch checkpoint for GRPO/PPO — LIVE (v0.70.0; live v0.71.11)
soup train --echo-trap-enabled [--echo-trap-threshold 0.6 --echo-trap-halt]  RAGEN echo-trap detector for GRPO — LIVE callback (v0.70.0; live v0.71.11)
soup train  # task='moe_lora_routing' + mole_task_adapters  MoLE per-token gate over N frozen task LoRAs (gate-only train) — LIVE (v0.71.12)
soup train  # task='distill' + distill_mode=token|sequence  Token logit-KL or sequence-level teacher-continuation KD — LIVE (v0.71.12)
soup train  # task=classifier|reranker|cross_encoder + lora  LoRA-adapter classifier (frozen encoder) — LIVE (v0.71.12)
soup train  # use_mod | expand_layers | use_longlora  Mixture-of-Depths / LLaMA Pro / LongLoRA S² (Llama/Qwen/Mistral[/Phi]) — LIVE (v0.71.12)
soup version [--full] [--json]                Show version (--full: system info, --json: JSON output)
soup --verbose <command>                      Full traceback on errors
```


