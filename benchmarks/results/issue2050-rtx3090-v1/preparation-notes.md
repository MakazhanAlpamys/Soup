# Preparation record, before any official timing

- Six synthetic CPU SAC tests passed locally with torch2.11.0 and on the lab
  with torch2.14.1. Initial discovery through named_modules() missed wrappers;
  discovery was corrected to the runtime's explicit apply() structural visitor.
- PyTorch2.14's indexed SAC cache differs from2.11. The helper inspects the
  known cache wrappers/sentinels; mutation refusal remains enabled. This was
  tested with the actual installed versions.
- Ten tests passed on the lab: synthetic cases, real 4-layer/two-slot Mistral
  GQA on CPU, and actual CUDA NF4 with exact two-update AdamW equality.
- Initial tiny wrapper smoke failed at the existing shard-directory containment
  rule because the pair harness starts in harness/. The wrapper now uses the
  explicit source checkout as its CWD and places shards under that tree. The
  tiny preparation log was overwritten by the next smoke; the error and repair
  are recorded here. No Mistral-7B timings existed at this point.
- The wrapper was checked with full/full-repeat/QKV/QKVO and nonzero adapter B,
  using native PagedAdamW8bit. Full equality passed before the campaign freeze.
- The original #361 telemetry polls the first physical GPU. The wrapper samples
  only the UUID selected in CUDA_VISIBLE_DEVICES, for all modes including full.
  Its backend/telemetry qualification is recorded separately from source defaults.
- Broad local test attempt:83 passed,2 failed. Both failures were CUDA NF4
  tests in the old laptop environment, where bitsandbytes is absent and the
  installed transformers raised NameError for bnb. Re-running the CPU subset:
  83 passed,2 GPU tests deselected. The CUDA NF4 tests had already passed on
  the fully installed lab environment. No production code was changed for this.
- Seven selector/audit/hash tests subsequently passed locally. Full repository
  Ruff check passed. The frozen code was compared byte-for-byte against the
  lab's freeze manifest: all11 files match, including unchanged runtime sources.
