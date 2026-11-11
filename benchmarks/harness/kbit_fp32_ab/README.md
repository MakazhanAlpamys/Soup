# kbit_fp32_ab: the fp32 upcast of resident 4-bit SFT, one RTX 4090

Scripts behind [`../../gate-2609-kbit-fp32-ab.md`](../../gate-2609-kbit-fp32-ab.md) (issue
[#2609](https://github.com/MakazhanAlpamys/Soup/issues/2609), step 1). Four arms of the same
`soup train` run (Qwen2.5-7B-Instruct, NF4, LoRA r16, batch 8 x 512) differ only in how PEFT's
`prepare_model_for_kbit_training` is applied. No file under `src/soup_cli` was changed; arms B and C
patch the PEFT function inside the training process, and arm D is one yaml line.

| arm | what differs |
|---|---|
| A | soup-cli 0.77.2 from PyPI, nothing patched |
| B | A, then `embed_tokens`, `lm_head` and every norm weight cast from float32 back to bfloat16 |
| C | `prepare_model_for_kbit_training` replaced by: freeze the base, `enable_input_require_grads()`, gradient checkpointing; no dtype cast |
| D | A plus `training.use_liger: true` |

## Files

The scripts are the files that ran on the rented machine, except that comments, docstrings and two
help strings that named internal working labels, the rental provider or a local path were replaced by
neutral wording (no code changed; see the last section of `harness-diff.txt`), and `p6_post.py`, where
two example paths in the docstring were replaced by placeholders. The hashes below are of the published
files. `ruff.toml` relaxes six lint rules for
this directory only, so the files did not have to be reformatted (which would change their hashes).

| file | role | sha256 |
|---|---|---|
| `job_p6.sh` | the job: set-up, model download, data window, 20 cells in the order A B C D per round, throttle sampler | `179643314ee6a6cb8d31d3443f53ec3e0b88e5c43467174b7c0e446351ba5075` |
| `p6_arms.py` | arms B and C (patches `peft.prepare_model_for_kbit_training` in the worker), the dtype census written to `arm_info.json` | `1bc6335e61636a26ca27a56a28395b9d32991173de6d0ffd4fd1a8b9a1a17bf6` |
| `requirements-tco.txt` | pinned stack | `27bb12191e95f2b0179835a9691629fd51cc4bd32832564cb9e298f5b75d2531` |
| `run_tco.sh` | set-up of the virtual environment (identical to the base harness) | `1fd672f6c7ceaa9449dfc2713d0eadbce0826c72bc0a03d07b17d89f9d34ca6b` |
| `tco_common.py`, `tco_meter.py`, `tco_profile.py` | step meter, GPU poller, profiler wrapper (as in the base harness; `tco_meter.py` and `tco_profile.py` differ from it in one comment or docstring line each) | `28086dac...`, `03e004aa...`, `0d973664...` |
| `tco_run.py` | cell runner; the base harness (`ace99f3e...`) plus the arm D yaml line and the `p6_arms` hook, see `harness-diff.txt` | `83ce37e1d5f28cd7a93c549696f957c4045f478067dd90d691853a781e1a9111` |
| `analyze.py` | joins every GPU event of a profiler trace to its CPU launch and aten op, buckets by kernel and shape | `716be31a0004989356d85054f61bab568104962d2c854cae75e77673b357827a` |
| `p6_post.py` | timing, loss and profile tables from the per-cell files | `7c2073ae307b521d6a1f1d58b6dd7f297068bb1700876c5076c812aa63f98fc5` |
| `throttle_summary.py` | throttle-reason share per timing cell from the 2-second `nvidia-smi` sampler | `f009a787a05b7882bffbec1cbdbff2331f3a6e49e3e20bcdcc38e54acefb2f15` |
| `harness-diff.txt` | what changed against the base harness | |

`tco_common.py` and `run_tco.sh` are byte-identical to the base harness (the hashes above equal the
base harness's); `tco_meter.py` and `tco_profile.py` differ from it in one comment or docstring line
each; `tco_run.py` is the only file whose code changed.

## Re-running

One Linux machine with one RTX 4090 and a driver that supports CUDA 12.8 wheels. The harness
pre-flight asks for 48 GB of host RAM and 80 GB of free disk; the recorded machine had 61.9 GB and
66.1 GB, the pre-flight reported the disk check as failed, and every cell ran (the largest write is
the 15 GB model cache). From a directory holding these files:

```bash
VK_OUT=/some/output/dir bash job_p6.sh
```

`VK_OUT` is the output directory (the job runner that started the original run provided it; any
writable directory works). `/root/vk/deadline` is read if it exists and ignored otherwise. The
environment variables `P6_ARMS` (default `A B C D`), `P6_REPS` (default `1 2 3`) and `P6_PROF_REPS`
(default `1 2`) narrow a run. The job installs the pinned stack and liger-kernel (arm D) into a fresh
virtual environment, downloads the model, builds the 840-row window (its sha256 is checked against
`eba8a5bc6074eced5adb255f3f8872d942779978171421e13f9c670494479c8a`) and then runs each cell. A
full run took 60 minutes on the recorded machine.

## Producing the tables

```bash
# timing, loss, profile and tables.md from the per-cell files (this is what produced ../../results/kbit_fp32_ab/)
python -I p6_post.py --runs <out/runs> --analysis <analysis.json> --out <results dir>

# analysis.json from the profiler traces; one --arm per profile cell
python analyze.py --arm A-r1-P=<out/runs>/A-r1-P/prof --arm A-r2-P=<out/runs>/A-r2-P/prof ... --out analysis.json

# throttle reasons per timing cell
python throttle_summary.py <box dir holding smi_throttle.csv and cell_times.tsv>
```

The Chrome traces (`prof/trace.json.gz`, eight of them) and `analysis.json` (475,886 bytes, sha256
`edd5b2a42a7e95414e85e257160aff79ca2e7c4b4591f6046d9f5ca5edb170cc`) are not in the repository. With
`analysis.json` and the per-cell files in `../../results/kbit_fp32_ab/runs/`, `p6_post.py` reproduces
`timing.json`, `loss.json`, `profile.json` and `tables.md` byte for byte, except that the repository's end-of-file hook added a
final newline to each JSON file.
