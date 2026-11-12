"""Record the #2104 resident-parity fixtures: K2's own pinned remote code vs Soup's route.

This is the recorder half of the resident parity gate (``benchmarks/gate-kimi-k2-resident.md``).
It:

1. builds the shared seeded tiny K2 checkpoints (``tests/kimi_k2_tiny.py`` — the exact
   builder the streaming suite's G2 tests use);
2. creates (once) an isolated venv under the system temp dir and installs a PINNED old
   ``transformers`` — starting at ``4.57.6`` and walking the 4.x line until one imports the
   pinned ``modeling_deepseek.py`` — plus CPU torch, numpy and safetensors;
3. downloads ``modeling_deepseek.py`` and ``configuration_deepseek.py`` from
   moonshotai/Kimi-K2-Base at the revision the admission gate pinned, and refuses to run
   unless the full sha256 of ``modeling_deepseek.py`` equals the gate's pinned value;
4. runs K2's own code (``trust_remote_code=True`` in the isolated venv ONLY) at fp32 on
   CPU with three seeded fixed inputs (lengths 8, 16, 33), twice: on the tiny bf16
   checkpoint (``logits_i``/``loss_i``) and on the dequantised FP8 round trip — the exact
   weights the gate's FP8 cell materialises, which K2's code reads as an ordinary
   checkpoint (``logits_fp8_i``/``loss_fp8_i``). The FP8 reference exists because the
   e4m3 round trip moves the final logits by ~1e-2 on this tiny model, three orders of
   magnitude over the gate tolerance: the pristine-bf16 fixtures are recorded (and their
   agreement with transformers' own native class on the current stack is reported in
   ``benchmarks/gate-kimi-k2-resident.md``), but the FP8 cell references the round-trip
   fixtures so the gate measures the load path, not the quantisation error — the round
   trip itself is gated by ``tests/test_fp8_block_streaming.py``;
5. writes ``tests/fixtures/kimi_k2/resident_parity_fixtures.npz`` +
   ``tests/fixtures/kimi_k2/manifest.json``: sha256 of the fixtures, of the pinned
   ``modeling_deepseek.py``, of ``configuration_deepseek.py`` and of the checkpoint
   ``config.json`` used, the validated transformers pin, the venv's torch/numpy/python
   versions, the exact input token ids and the recording command line.

The gate test (``tests/test_kimi_k2_resident_parity.py``) compares Soup's native resident
load of the FP8 tiny checkpoint against these fixtures; the FP8 quantize->dequant round
trip itself is gated by ``tests/test_fp8_block_streaming.py``.

``--check`` verifies the committed fixtures against the manifest (fixtures sha256, pinned
modeling sha256, manifest/npz input-id agreement); it needs no venv and no network.

Run from the repo root: ``python benchmarks/harness/kimi_k2_resident_parity_record.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TESTS_DIR = REPO_ROOT / "tests"
FIXTURES_DIR = TESTS_DIR / "fixtures" / "kimi_k2"
FIXTURES_NPZ = FIXTURES_DIR / "resident_parity_fixtures.npz"
MANIFEST = FIXTURES_DIR / "manifest.json"

#: moonshotai/Kimi-K2-Base revision the admission gate pinned for config.json; the gate's
#: table records modeling_deepseek.py "as above", i.e. this same revision.
K2_REVISION = "ce72df012259dcc55d945e890f815fe7ef69159c"
K2_REPO_ID = "moonshotai/Kimi-K2-Base"
#: Full sha256 of modeling_deepseek.py at K2_REVISION, pinned by
#: benchmarks/gate-kimi-k2-admission.md's table; the recorder refuses to run on any other file.
MODELING_SHA256 = "ce753fc9076ce13ce9557e680e2f5bd68ae8929a8225772968a9a22a1d17b7d4"

#: transformers pins tried in order until one imports the pinned modeling file: the stack
#: recorded in benchmarks/README.md's history first, then the 4.x line walked downward
#: (older DeepSeek-V3-era releases) and only then upward. The validated pin is recorded in
#: the manifest; the gate test does not re-resolve it.
TRANSFORMERS_CANDIDATES = [
    "4.57.6",
    *[f"4.57.{p}" for p in range(5, -1, -1)],
    "4.56.2",
    "4.56.1",
    "4.56.0",
    "4.55.4",
    "4.55.3",
    "4.55.2",
    "4.55.1",
    "4.55.0",
    "4.54.1",
    "4.54.0",
    "4.53.3",
    "4.53.2",
    "4.53.1",
    "4.53.0",
    *[f"4.58.{p}" for p in range(0, 4)],
    *[f"4.59.{p}" for p in range(0, 3)],
]

VENV_DIR = Path(tempfile.gettempdir()) / "kimi-k2-resident-parity-venv"
_VENV_BIN = "Scripts" if os.name == "nt" else "bin"
VENV_PYTHON = VENV_DIR / _VENV_BIN / ("python.exe" if os.name == "nt" else "python")

#: Fixed gate inputs: three token-id sequences of lengths 8, 16 and 33, drawn from a seeded
#: generator over the tiny vocab. The gate test reads the ids back from the npz/manifest.
INPUT_SEED = 977
INPUT_LENGTHS = (8, 16, 33)
INPUT_VOCAB = 128

_CHILD = r'''
"""The isolated-venv half of the recorder: run K2's own code, save the fixtures npz.

argv: out_path, seed, vocab, then a JSON list of {folder, prefix} runs. Every run uses the
same seeded inputs (the same generator seed); the first run also stores the ids. Prefix
"" names the bf16-source reference (logits_i); prefix "fp8_" names the reference recorded
on the dequantised FP8 round trip (logits_fp8_i), the exact weights the gate's FP8 cell
materialises — the pristine-bf16 reference would carry the quantisation error the gate is
not about (benchmarks/gate-kimi-k2-resident.md, section 3).
"""
import json
import platform
import sys

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

out_path, seed, vocab = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
runs, lengths = json.loads(sys.argv[4]), json.loads(sys.argv[5])
torch.manual_seed(20261112)
torch.use_deterministic_algorithms(True)
payload = {}
for run_index, run in enumerate(runs):
    config = AutoConfig.from_pretrained(run["folder"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        run["folder"], config=config, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    prefix = run["prefix"]
    for index, length in enumerate(lengths):
        ids = torch.randint(0, vocab, (1, length), generator=generator)
        with torch.no_grad():
            output = model(input_ids=ids, labels=ids)
        if run_index == 0:
            payload[f"ids_{index}"] = ids[0].to(torch.int64).numpy()
        payload[f"logits_{prefix}{index}"] = output.logits[0, -1].to(torch.float32).numpy()
        payload[f"loss_{prefix}{index}"] = np.float32(float(output.loss))
    del model
np.savez(out_path, **payload)
print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": __import__("transformers").__version__,
    "numpy": np.__version__,
}))
'''


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_helper():
    """tests/kimi_k2_tiny.py by file location (the harness pattern; no package install)."""
    import importlib.util

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("kimi_k2_tiny", TESTS_DIR / "kimi_k2_tiny.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(command, **kwargs):
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True, **kwargs)


def _ensure_venv() -> Path:
    if not VENV_PYTHON.exists():
        print(f"creating the isolated venv at {VENV_DIR}")
        created = subprocess.run(
            [sys.executable, "-m", "venv", VENV_DIR], capture_output=True, text=True
        )
        if created.returncode != 0:
            # e.g. Debian/Ubuntu's python3 without ensurepip: build without pip and
            # bootstrap pip into the venv from get-pip.py.
            print("python -m venv failed (ensurepip missing?); retrying --without-pip + get-pip.py")
            shutil.rmtree(VENV_DIR, ignore_errors=True)
            _run([sys.executable, "-m", "venv", "--without-pip", VENV_DIR])
            get_pip = Path(tempfile.mkdtemp(prefix="get-pip-")) / "get-pip.py"
            with urllib.request.urlopen("https://bootstrap.pypa.io/get-pip.py", timeout=120) as r:
                get_pip.write_bytes(r.read())
            _run([VENV_PYTHON, get_pip])
        _run([VENV_PYTHON, "-m", "pip", "install", "--upgrade", "pip"])
        _run(
            [
                VENV_PYTHON,
                "-m",
                "pip",
                "install",
                "torch",
                "numpy",
                "safetensors",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
                "--extra-index-url",
                "https://pypi.org/simple",
            ]
        )
    return VENV_PYTHON


def _validates_pin(python: Path, code_dir: Path) -> str | None:
    """The venv's transformers version if it imports the pinned modeling file, else None.

    The pinned file does a relative ``from .configuration_deepseek import ...``, so the
    probe imports it as the package ``k2code`` (the directory holds both pinned files and
    an ``__init__.py``), the same shape transformers' dynamic-module loader builds.
    """
    probe = (
        f"import sys; sys.path.insert(0, {str(code_dir.parent)!r});"
        " import k2code.modeling_deepseek as module;"
        " assert hasattr(module, 'DeepseekV3ForCausalLM'), 'no DeepseekV3ForCausalLM';"
        " import transformers;"
        " print(transformers.__version__)"
    )
    result = subprocess.run([str(python), "-c", probe], capture_output=True, text=True, timeout=300)
    return result.stdout.strip() if result.returncode == 0 else None


def _installed_transformers(python: Path) -> str | None:
    result = subprocess.run(
        [str(python), "-c", "import transformers; print(transformers.__version__)"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _select_transformers_pin(python: Path, code_dir: Path, run_child) -> str:
    """Walk the candidate pins until one not only imports but RUNS the pinned remote code.

    Importing the pinned file is necessary but not sufficient (transformers 4.57.6 imports
    it, then fails at runtime: K2's code calls ``DynamicCache.get_usable_length``, which
    newer caches no longer have), so each import-validated candidate must produce the
    fixtures before its pin is recorded. A venv already holding a working candidate keeps
    it (first in the order), so re-runs do not churn pins.
    """
    candidates = list(TRANSFORMERS_CANDIDATES)
    installed = _installed_transformers(python)
    if installed in candidates:
        candidates = [installed] + [pin for pin in candidates if pin != installed]
    for pin in candidates:
        if _installed_transformers(python) != pin:
            print(f"trying transformers=={pin}")
            _run([python, "-m", "pip", "install", f"transformers=={pin}"])
        version = _validates_pin(python, code_dir)
        if not version:
            print(f"  transformers {pin}: does not import the pinned modeling file; next")
            continue
        print(f"  transformers {version}: import-validated; running K2's own code on it")
        try:
            run_child()
        except subprocess.CalledProcessError:
            print(f"  transformers {version}: the pinned remote code fails on it; next pin")
            continue
        print(f"validated pin: transformers {version} runs the pinned modeling_deepseek.py")
        return version
    raise SystemExit(
        "no candidate transformers pin runs the pinned modeling_deepseek.py; tried "
        + ", ".join(candidates)
    )


def _download_pinned_code(destination: Path) -> dict:
    """Fetch the two K2 code files at the pinned revision; assert the pinned sha256."""
    destination.mkdir(parents=True, exist_ok=True)
    shas = {}
    for name in ("modeling_deepseek.py", "configuration_deepseek.py"):
        url = f"https://huggingface.co/{K2_REPO_ID}/raw/{K2_REVISION}/{name}"
        target = destination / name
        print(f"downloading {url}")
        with urllib.request.urlopen(url, timeout=120) as response:
            target.write_bytes(response.read())
        shas[name] = _sha256(target)
    # The pinned modeling file imports its config sibling relatively; give the probe (and
    # only the probe — the gate's own run uses transformers' dynamic-module loader) a
    # package to import them as.
    (destination / "__init__.py").write_text("", encoding="utf-8")
    if shas["modeling_deepseek.py"] != MODELING_SHA256:
        raise SystemExit(
            f"modeling_deepseek.py at {K2_REVISION} has sha256 "
            f"{shas['modeling_deepseek.py']}, not the admission gate's pinned "
            f"{MODELING_SHA256}; refusing to record fixtures against different vendor code"
        )
    return shas


def _record() -> None:
    helper = _load_helper()
    work = Path(tempfile.mkdtemp(prefix="kimi-k2-resident-parity-"))
    checkpoints = helper.build_tiny_k2_checkpoints(work / "checkpoints")
    k2_bf16 = Path(checkpoints["bf16"][0])
    config_sha256 = _sha256(k2_bf16 / "config.json")

    code_dir = work / "k2code"
    code_shas = _download_pinned_code(code_dir)
    python = _ensure_venv()

    # The two folders K2's own code runs from, each the tiny checkpoint plus the two
    # pinned code files, sha-verified above: the pristine bf16 source, and the
    # dequantised FP8 round trip (the fp8-ref weights the gate's FP8 cell materialises,
    # under the same K2-spelled config.json). trust_remote_code is used in this isolated
    # venv only; nothing in Soup's tree or test env ever executes these files.
    run_folder = work / "run"
    shutil.copytree(k2_bf16, run_folder)
    run_fp8_folder = work / "run-fp8"
    run_fp8_folder.mkdir()
    fp8_ref = Path(checkpoints["fp8"][1])
    shutil.copyfile(fp8_ref / "model.safetensors", run_fp8_folder / "model.safetensors")
    shutil.copyfile(k2_bf16 / "config.json", run_fp8_folder / "config.json")
    for folder in (run_folder, run_fp8_folder):
        for name in ("modeling_deepseek.py", "configuration_deepseek.py"):
            shutil.copyfile(code_dir / name, folder / name)
    runs = [
        {"folder": str(run_folder), "prefix": ""},
        {"folder": str(run_fp8_folder), "prefix": "fp8_"},
    ]

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    child = work / "_resident_record_child.py"
    child.write_text(_CHILD, encoding="utf-8")

    def run_child():
        _run(
            [
                python,
                child,
                FIXTURES_NPZ,
                str(INPUT_SEED),
                str(INPUT_VOCAB),
                json.dumps(runs),
                json.dumps(list(INPUT_LENGTHS)),
            ]
        )

    transformers_pin = _select_transformers_pin(python, code_dir, run_child)
    versions = json.loads(
        subprocess.run(
            [
                python,
                "-c",
                "import json, platform, torch, transformers, numpy; print(json.dumps("
                "{'python': platform.python_version(), 'torch': torch.__version__, "
                "'transformers': transformers.__version__, 'numpy': numpy.__version__}))",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )

    manifest = {
        "gate": "benchmarks/gate-kimi-k2-resident.md (#2104)",
        "fixtures": {
            "path": "tests/fixtures/kimi_k2/resident_parity_fixtures.npz",
            "sha256": _sha256(FIXTURES_NPZ),
            "entries": [
                *(f"ids_{i}" for i in range(len(INPUT_LENGTHS))),
                *(f"logits_{i}" for i in range(len(INPUT_LENGTHS))),
                *(f"loss_{i}" for i in range(len(INPUT_LENGTHS))),
                *(f"logits_fp8_{i}" for i in range(len(INPUT_LENGTHS))),
                *(f"loss_fp8_{i}" for i in range(len(INPUT_LENGTHS))),
            ],
            "reference_runs": [
                "logits_/loss_: K2's own code on the pristine bf16 tiny checkpoint",
                "logits_fp8_/loss_fp8_: K2's own code on the dequantised FP8 round trip "
                "(the exact weights the gate's FP8 cell materialises); the gate's FP8 cell "
                "references these",
            ],
        },
        "modeling_deepseek.py": {
            "repo_id": K2_REPO_ID,
            "revision": K2_REVISION,
            "sha256": code_shas["modeling_deepseek.py"],
        },
        "configuration_deepseek.py": {
            "repo_id": K2_REPO_ID,
            "revision": K2_REVISION,
            "sha256": code_shas["configuration_deepseek.py"],
        },
        "checkpoint_config_json": {
            "path": "<tiny k2-bf16 folder>/config.json (tests/kimi_k2_tiny.py _tiny_k2_raw())",
            "sha256": config_sha256,
        },
        "recording_env": versions,
        "transformers_pin": transformers_pin,
        "inputs": {
            "seed": INPUT_SEED,
            "vocab_size": INPUT_VOCAB,
            "lengths": list(INPUT_LENGTHS),
            "note": "exact ids are stored in the npz as ids_0..ids_2 and mirrored below",
        },
        "recording_command": " ".join(sys.argv),
        "recorded_with_python": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
    }
    # Mirror the exact ids into the manifest so a reader needs nothing else.
    import numpy as np

    with np.load(FIXTURES_NPZ) as fixtures:
        manifest["inputs"]["ids"] = [
            fixtures[f"ids_{i}"].tolist() for i in range(len(INPUT_LENGTHS))
        ]
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {FIXTURES_NPZ} and {MANIFEST}")


def _check() -> None:
    if not MANIFEST.exists() or not FIXTURES_NPZ.exists():
        raise SystemExit(
            f"missing {MANIFEST.name} or {FIXTURES_NPZ.name}; run the recorder: "
            "python benchmarks/harness/kimi_k2_resident_parity_record.py"
        )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = []
    fixtures_sha = _sha256(FIXTURES_NPZ)
    if manifest["fixtures"]["sha256"] != fixtures_sha:
        failures.append(
            f"fixtures sha256 {fixtures_sha} != manifest {manifest['fixtures']['sha256']}"
        )
    if manifest["modeling_deepseek.py"]["sha256"] != MODELING_SHA256:
        failures.append("manifest's modeling_deepseek.py sha256 is not the admission gate's pin")
    if manifest["modeling_deepseek.py"]["revision"] != K2_REVISION:
        failures.append("manifest's modeling_deepseek.py revision is not the admission gate's pin")
    import numpy as np

    with np.load(FIXTURES_NPZ) as fixtures:
        if sorted(fixtures.files) != sorted(manifest["fixtures"]["entries"]):
            failures.append("npz entries disagree with the manifest's entries list")
        for index, ids in enumerate(manifest["inputs"]["ids"]):
            if fixtures[f"ids_{index}"].tolist() != ids:
                failures.append(f"npz ids_{index} disagrees with the manifest")
        for index in range(len(manifest["inputs"]["ids"])):
            for prefix in ("", "fp8_"):
                for stem in ("logits", "loss"):
                    name = f"{stem}_{prefix}{index}"
                    if name not in fixtures:
                        failures.append(f"npz is missing {name}")
        # The FP8 reference must carry the round trip: identical arrays would mean the
        # second run never happened.
        if all(
            np.array_equal(fixtures[f"logits_{index}"], fixtures[f"logits_fp8_{index}"])
            for index in range(len(manifest["inputs"]["ids"]))
        ):
            failures.append("bf16-source and fp8 reference logits are identical")
    if failures:
        raise SystemExit("fixture check failed:\n- " + "\n- ".join(failures))
    print(
        f"fixtures OK: {FIXTURES_NPZ.name} sha256 matches the manifest; "
        f"{len(manifest['inputs']['ids'])} recorded inputs agree"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed fixtures against the manifest (no venv, no network)",
    )
    args = parser.parse_args()
    if args.check:
        _check()
    else:
        _record()


if __name__ == "__main__":
    main()
