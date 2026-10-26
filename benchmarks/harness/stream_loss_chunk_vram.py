#!/usr/bin/env python3
"""Peak VRAM of a streamed sft step with and without ``training.stream_loss_chunk``.

The measurement #1693 asks for: peak VRAM at 512 / 1024 / 2048 tokens, with and
without the option, same model and stack, beside the pre-flight's prediction for
the same run, so that "the pre-flight does not under-predict on those rows" can
be checked rather than assumed.

``--self-test`` runs the same protocol on a tiny local Llama and never reads a
peak: it checks the plumbing (the arms, the realised length, the hook, the
pre-flight's inputs and arithmetic, the verdicts), not memory. It runs on the
device ``TrainingArguments`` picks by itself, so CPU, or MPS on Apple Silicon.

Requirements
------------
- CUDA GPU for the measurement. The unchunked arm is expected to run out of
  memory at the longer lengths on a small card; that is a row, not a crash.
- The base model, by Hub id or local path (``--base``).
- ``--self-test`` needs neither: no GPU, no download, a few seconds.

Usage
-----
    python benchmarks/harness/stream_loss_chunk_vram.py --base <id-or-path>
    python benchmarks/harness/stream_loss_chunk_vram.py --base <id-or-path> \\
        --quantization 4bit --seqs 512,1024,2048 --chunk 128
    python benchmarks/harness/stream_loss_chunk_vram.py --self-test

Protocol
--------
The real ``soup train`` setup (``SFTTrainerWrapper.setup``) and one training
step per arm, batch 1, LoRA r=8 on ``q_proj`` / ``v_proj``, ``stream_buffers: 2``,
the rewind log on (its default), the off arm first at each length. Each arm runs
in a fresh process, as each ``soup train`` does: in one process only the first arm
got the RAM tier and the rest fell back to the disk tier (#1693), since the tier is
chosen from the free RAM left behind. This process never initialises CUDA, so no
arm loses VRAM to a second context. Each arm sets the allocator hint as ``soup train``
does (``configure_cuda_alloc_hint``) before it reads the device.

- "real peak" is ``torch.cuda.max_memory_allocated()`` over the step, after
  ``reset_peak_memory_stats()``.
- "predicted" is the number the pre-flight itself computed during that setup,
  read from ``estimate_stream_peak_vram`` as it was called, so the two columns
  are about the same run.
- "free" is the ``torch.cuda.mem_get_info()`` reading the pre-flight compared
  against. ``stream_vram_override`` is set high so the pre-flight never refuses
  an arm this harness is trying to measure; "free" says what it would have done.

As in ``issue395_second_stack_vram.py``, ``data.max_length`` truncates, it does
not pad, so the rows overshoot 3x and the realised length is read off the
collated batch. A row whose realised length is not the requested one is void.

Verdicts and exit status
------------------------
- ``ok``: predicted >= real peak.
- ``UNDER-PREDICTS``: predicted < real peak, or the arm ran out of memory
  although the prediction fit in "free" (the pre-flight would have let it run).
- ``OOM, refused``: the arm ran out of memory and the pre-flight would have
  refused it. Expected for the unchunked arm at long lengths on a small card.
- ``VOID``: the realised length differs from the requested one.

Exit 0 when every row is ``ok`` or ``OOM, refused``; 1 when a row
under-predicts; 3 when a row is void; 2 for bad arguments or no CUDA device.
``--self-test`` exits 0 when every check passes and 1 otherwise.

Reading the table
-----------------
``pred/real`` below 1.0 is the direction the pre-flight must not err in. The
chunked rows are the ones nobody has measured: their logits term is the shipped
14 bytes per element charged on ``min(seq, chunk)`` tokens, plus two
``[batch, seq, hidden]`` tensors (``estimate_loss_chunk_bytes``).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import inspect
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import yaml

#: Bytes the pre-flight is told are free, so that it measures instead of refusing.
ASSUME_FREE_BYTES = 1 << 40

#: Rows per arm; all identical, so the step does not depend on the shuffle.
ROWS_PER_ARM = 4

#: Overshoot factor of each row over the requested length; see the docstring.
OVERSHOOT = 3

VERDICT_OK = "ok"
VERDICT_UNDER = "UNDER-PREDICTS"
VERDICT_REFUSED = "OOM, refused"
VERDICT_VOID = "VOID"
VERDICT_NOT_MEASURED = "no peak read"

EXIT_OK = 0
EXIT_UNDER = 1
EXIT_USAGE = 2
EXIT_VOID = 3


@dataclass(frozen=True)
class Arm:
    """One arm of one length: what the pre-flight predicted and what happened."""

    seq: int
    chunk: Optional[int]  # None is the unchunked arm
    realised: int
    predicted: int
    preflight: dict  # the keyword arguments the pre-flight's estimate was called with
    free: Optional[int]  # the free VRAM the pre-flight compared against; None off CUDA
    peak: Optional[int]  # None when not measured (off CUDA) or out of memory
    oom: bool
    hooked: bool  # the chunked head and loss was on the trainer
    loss: Optional[float]  # the step's training loss; None when it ran out of memory


def rows(seq: int) -> list[dict]:
    """``ROWS_PER_ARM`` sft rows that tokenize past ``seq``, so truncation lands on it."""
    body = "word " * (seq * OVERSHOOT)
    return [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": body}]}
        for _ in range(ROWS_PER_ARM)
    ]


def config_doc(
    base: str, seq: int, chunk: Optional[int], quantization: str, out_dir: pathlib.Path
) -> dict:
    """One arm's ``soup.yaml`` as a dict; ``chunk=None`` is the unchunked arm."""
    training: dict[str, Any] = {
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "epochs": 1,
        "quantization": quantization,
        "stream_layers": True,
        "stream_buffers": 2,
        "stream_vram_override": ASSUME_FREE_BYTES,
        "logging_steps": 1,
        "save_steps": 100000,
        "lora": {"r": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]},
    }
    if chunk is not None:
        training["stream_loss_chunk"] = chunk
    return {
        "base": base,
        "task": "sft",
        "backend": "transformers",
        "modality": "text",
        "data": {
            "train": str(out_dir / "d.jsonl"),
            "max_length": seq,
            "chat_template": "chatml",
        },
        "output": str(out_dir / "out"),
        "training": training,
    }


def verdict(arm: Arm) -> str:
    """The row's verdict; see the module docstring."""
    if arm.realised != arm.seq:
        return VERDICT_VOID
    if arm.oom:
        if arm.free is not None and arm.predicted <= arm.free:
            return VERDICT_UNDER
        return VERDICT_REFUSED
    if arm.peak is None:
        return VERDICT_NOT_MEASURED
    return VERDICT_OK if arm.predicted >= arm.peak else VERDICT_UNDER


def exit_status(arms: list[Arm]) -> int:
    """``EXIT_VOID`` beats ``EXIT_UNDER``: a void row says nothing either way."""
    verdicts = [verdict(arm) for arm in arms]
    if VERDICT_VOID in verdicts:
        return EXIT_VOID
    if VERDICT_UNDER in verdicts:
        return EXIT_UNDER
    return EXIT_OK


HEADER = (
    f"{'seq':>6} {'realised':>8} {'chunk':>6} | {'predicted':>11} {'free':>11} "
    f"{'real peak':>11} {'pred/real':>9} | verdict"
)


def _gb(value: Optional[int]) -> str:
    return "-" if value is None else f"{value / 1e9:.3f}GB"


def table_row(arm: Arm) -> str:
    """One printed line of the table."""
    if arm.oom:
        peak, ratio = "OOM", "-"
    elif arm.peak is None:
        peak, ratio = "-", "-"
    else:
        peak, ratio = _gb(arm.peak), f"{arm.predicted / arm.peak:.4f}"
    chunk = "off" if arm.chunk is None else str(arm.chunk)
    return (
        f"{arm.seq:>6} {arm.realised:>8} {chunk:>6} | {_gb(arm.predicted):>11} "
        f"{_gb(arm.free):>11} {peak:>11} {ratio:>9} | {verdict(arm)}"
    )


@contextlib.contextmanager
def recording_preflight() -> Iterator[dict]:
    """Record what the pre-flight computes, without changing it.

    ``stream_setup`` imports both functions from ``layer_stream`` when the
    pre-flight runs, so replacing the module attributes for the duration of one
    setup is enough; they are put back however the setup ends.
    """
    from soup_cli.utils import layer_stream

    estimate = layer_stream.estimate_stream_peak_vram
    resolve = layer_stream.resolve_available_vram_bytes
    seen: dict = {}

    def recording_estimate(**kwargs):
        seen["preflight"] = dict(kwargs)
        seen["predicted"] = estimate(**kwargs)
        return seen["predicted"]

    def recording_resolve(**kwargs):
        seen["free"] = kwargs["measured_bytes"]
        return resolve(**kwargs)

    layer_stream.estimate_stream_peak_vram = recording_estimate
    layer_stream.resolve_available_vram_bytes = recording_resolve
    try:
        yield seen
    finally:
        layer_stream.estimate_stream_peak_vram = estimate
        layer_stream.resolve_available_vram_bytes = resolve


def run_one(
    base: str,
    seq: int,
    chunk: Optional[int],
    quantization: str,
    device: str,
    work_dir: pathlib.Path,
    measure: bool = True,
) -> Arm:
    """Build one arm through the real setup and run one training step.

    The peak is read only when ``measure`` is set and ``device`` is CUDA.
    """
    import torch

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils.stream_loss_chunk import HOOK_ATTR

    on_cuda = device == "cuda"
    measuring = measure and on_cuda
    data = rows(seq)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "d.jsonl").write_text("\n".join(json.dumps(r) for r in data), encoding="utf-8")
    (work_dir / "out").mkdir(exist_ok=True)

    gc.collect()
    if on_cuda:
        torch.cuda.empty_cache()
    wrapper = None
    try:
        with recording_preflight() as seen:
            cfg = load_config_from_string(
                yaml.safe_dump(config_doc(base, seq, chunk, quantization, work_dir))
            )
            wrapper = SFTTrainerWrapper(cfg, device=device)
            wrapper.setup({"train": data})
        if "predicted" not in seen:
            raise RuntimeError(
                "the pre-flight did not run (it could not read the model's shape), "
                "so there is no prediction to compare against"
            )
        trainer = wrapper.trainer
        hooked = getattr(trainer, HOOK_ATTR, None) is not None
        batch = next(iter(trainer.get_train_dataloader()))
        realised = int(batch["input_ids"].shape[-1])
        trainer.args.max_steps = 1
        peak: Optional[int] = None
        loss: Optional[float] = None
        oom = False
        if measuring:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                loss = float(trainer.train().training_loss)
                peak = int(torch.cuda.max_memory_allocated())
            except torch.cuda.OutOfMemoryError:
                oom = True
        else:
            loss = float(trainer.train().training_loss)
    finally:
        if wrapper is not None:
            # `wrapper.train()` does both on its way out; this harness calls the
            # trainer directly, as issue395_second_stack_vram.py does.
            for teardown in (wrapper._remove_stream_loss_chunk_hook, wrapper._close_stream_runtime):
                try:
                    teardown()
                except Exception:  # noqa: BLE001 - teardown must not mask the measurement
                    pass
        del wrapper
        gc.collect()
        if on_cuda:
            torch.cuda.empty_cache()
    return Arm(
        seq=seq,
        chunk=chunk,
        realised=realised,
        predicted=int(seen["predicted"]),
        preflight=seen["preflight"],
        free=seen.get("free"),
        peak=peak,
        oom=oom,
        hooked=hooked,
        loss=loss,
    )


def _arm_name(chunk: Optional[int]) -> str:
    return "off" if chunk is None else str(chunk)


def parse_arm(value: str) -> tuple[int, Optional[int]]:
    """``--arm SEQ:CHUNK``, CHUNK an integer or ``off``."""
    seq, _, chunk = value.partition(":")
    try:
        return int(seq), None if chunk == "off" else int(chunk)
    except ValueError:
        raise ValueError(f"--arm must be SEQ:CHUNK or SEQ:off; got {value!r}") from None


def run_arm_process(
    base: str, seq: int, chunk: Optional[int], quantization: str, work_dir: pathlib.Path
) -> tuple[Arm, str]:
    """Run one arm in a fresh interpreter; ``(arm, the device it ran on)``."""
    work_dir.mkdir(parents=True, exist_ok=True)
    result = work_dir / "arm.json"
    argv = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--base", base,
        "--quantization", quantization,
        "--arm", f"{seq}:{_arm_name(chunk)}",
        "--arm-out", str(result),
    ]  # fmt: skip
    # The child writes to the same stdout; without this, a redirected run's header
    # lines would land after the first arm's output.
    sys.stdout.flush()
    status = subprocess.run(argv, check=False).returncode
    if status != 0 or not result.exists():
        raise RuntimeError(
            f"seq {seq}, chunk {_arm_name(chunk)}: the arm's process exited {status} "
            "without a result; its output is above"
        )
    doc = json.loads(result.read_text(encoding="utf-8"))
    return Arm(**doc["arm"]), doc["device"]


def run_arm(base: str, seq: int, chunk: Optional[int], quantization: str, out: str) -> int:
    """The child side of :func:`run_arm_process`: one arm, its result written to ``out``."""
    import torch

    from soup_cli.utils.gpu import configure_cuda_alloc_hint

    # As `soup train` does before anything starts CUDA (#2149); every arm streams.
    configure_cuda_alloc_hint(stream_layers=True)
    device = trainer_device()
    out_path = pathlib.Path(out)
    arm = run_one(base, seq, chunk, quantization, device, out_path.parent)
    if device == "cuda":
        # After the arm, so that this process does not initialise CUDA before the setup does.
        props = torch.cuda.get_device_properties(0)
        device = f"{props.name} | {props.total_memory / 1e9:.3f} GB"
    doc = {"arm": dataclasses.asdict(arm), "device": device}
    out_path.write_text(json.dumps(doc), encoding="utf-8")
    return EXIT_OK


def sweep(base: str, seqs: list[int], chunk: int, quantization: str) -> int:
    import torch

    import soup_cli

    print(f"# soup_cli {soup_cli.__version__} from {soup_cli.__file__}")
    print(f"# torch {torch.__version__} (CUDA {torch.version.cuda}) | one process per arm")
    print(f"# base {base} | quantization {quantization} | chunk {chunk}")
    arms = []
    with tempfile.TemporaryDirectory(prefix="stream_loss_chunk_vram_") as tmp:
        for seq in seqs:
            for arm_chunk in (None, chunk):
                work_dir = pathlib.Path(tmp) / f"s{seq}-{_arm_name(arm_chunk)}"
                arm, device = run_arm_process(base, seq, arm_chunk, quantization, work_dir)
                if not arms:
                    print(f"# device {device}")
                    print(HEADER)
                print(table_row(arm), flush=True)
                arms.append(arm)
    return exit_status(arms)


# --- --self-test: the same protocol on a tiny local Llama, no peak read ----

#: Small enough to run in seconds, with a vocabulary large enough that the
#: logits term the option removes dominates the two hidden-sized tensors it adds.
_TINY = {"vocab": 512, "hidden": 64, "layers": 2}
_SELF_TEST_SEQ = 64


def _tiny_llama(directory: pathlib.Path) -> str:
    """A real (tiny, untied) Llama checkpoint and an offline tokenizer on disk."""
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=_TINY["vocab"],
        hidden_size=_TINY["hidden"],
        intermediate_size=_TINY["hidden"] * 2,
        num_hidden_layers=_TINY["layers"],
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=False,
        max_position_embeddings=4 * _SELF_TEST_SEQ,
    )
    model = LlamaForCausalLM(config).to(torch.float32)
    directory.mkdir(parents=True, exist_ok=True)
    state = {name: value.contiguous() for name, value in model.state_dict().items()}
    save_file(state, str(directory / "model.safetensors"))
    config.save_pretrained(str(directory))

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3, "hi": 4, "word": 5}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "model_max_length": 4 * _SELF_TEST_SEQ,
                "clean_up_tokenization_spaces": False,
            }
        ),
        encoding="utf-8",
    )
    return str(directory)


def trainer_device() -> str:
    """The device ``TrainingArguments`` picks by itself: cuda, else mps, else cpu.

    A streamed model built anywhere else fails its first step on a device
    mismatch, so the self-test builds there too.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps is not None and mps.is_available() else "cpu"


@contextlib.contextmanager
def _environment(**values: str) -> Iterator[None]:
    before = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _self_test_checks(off: Arm, chunked: Arm, chunk: int) -> list[tuple[str, bool]]:
    from soup_cli.utils.layer_stream import (
        estimate_logits_bytes,
        estimate_loss_chunk_bytes,
        estimate_stream_peak_vram,
    )

    seq = off.seq
    kwargs = off.preflight
    # The pre-flight does not pass a dtype; the estimate then charges its default.
    dtype = kwargs.get(
        "dtype", inspect.signature(estimate_stream_peak_vram).parameters["dtype"].default
    )
    full = estimate_logits_bytes(
        vocab_size=kwargs["vocab_size"],
        seq_len=seq,
        batch_size=kwargs["batch_size"],
        bytes_per_element=kwargs["logits_bytes_per_element"],
    )
    one_chunk = estimate_logits_bytes(
        vocab_size=kwargs["vocab_size"],
        seq_len=min(seq, chunk),
        batch_size=kwargs["batch_size"],
        bytes_per_element=kwargs["logits_bytes_per_element"],
    )
    kept = estimate_loss_chunk_bytes(
        hidden_size=kwargs["hidden_size"],
        seq_len=seq,
        batch_size=kwargs["batch_size"],
        dtype=dtype,
    )
    same_loss = (
        off.loss is not None
        and chunked.loss is not None
        and math.isfinite(off.loss)
        and abs(chunked.loss - off.loss) <= 1e-5 * abs(off.loss)
    )
    rest_equal = {k: v for k, v in off.preflight.items() if k != "loss_chunk"} == {
        k: v for k, v in chunked.preflight.items() if k != "loss_chunk"
    }

    def synthetic(**fields: Any) -> Arm:
        values = dict(off.__dict__)
        values.update(fields)
        return Arm(**values)

    return [
        ("both arms realise the requested length", off.realised == chunked.realised == seq),
        ("the pre-flight saw seq_len == the requested length", kwargs["seq_len"] == seq),
        (
            "the pre-flight saw loss_chunk None / the chunk",
            off.preflight["loss_chunk"] is None and chunked.preflight["loss_chunk"] == chunk,
        ),
        ("the arms' pre-flight inputs differ only in loss_chunk", rest_equal),
        ("the chunked head and loss is on the chunked arm only", chunked.hooked and not off.hooked),
        (
            "predicted(off) - predicted(chunk) == full logits - one chunk - two hidden tensors",
            off.predicted - chunked.predicted == full - one_chunk - kept,
        ),
        ("the step losses agree within 1e-5 (LoRA B starts at zero)", same_loss),
        ("no peak is read and no row claims one", off.peak is None and not off.oom),
        (
            "a measured row at or above its peak is ok",
            verdict(synthetic(peak=off.predicted)) == VERDICT_OK,
        ),
        (
            "a row below its peak under-predicts and exits 1",
            exit_status([synthetic(peak=off.predicted + 1)]) == EXIT_UNDER,
        ),
        (
            "an OOM the pre-flight would refuse is a row, exit 0",
            exit_status([synthetic(oom=True, free=off.predicted - 1)]) == EXIT_OK
            and "OOM" in table_row(synthetic(oom=True, free=off.predicted - 1)),
        ),
        (
            "an OOM inside the predicted fit under-predicts",
            verdict(synthetic(oom=True, free=off.predicted)) == VERDICT_UNDER,
        ),
        (
            "a row at another length is void and exits 3",
            exit_status([synthetic(realised=seq - 1), synthetic(peak=off.predicted + 1)])
            == EXIT_VOID,
        ),
    ]


def run_self_test(chunk: int = 16) -> int:
    """The protocol end to end without a peak: both arms, one length, every check printed."""
    with tempfile.TemporaryDirectory(prefix="stream_loss_chunk_vram_self_test_") as tmp:
        root = pathlib.Path(tmp)
        base = _tiny_llama(root / "model")
        device = trainer_device()
        with _environment(SOUP_LAYER_STREAM_CACHE_DIR=str(root / "cache")):
            arms = [
                run_one(
                    base,
                    _SELF_TEST_SEQ,
                    arm,
                    "none",
                    device,
                    root / f"arm-{arm or 'off'}",
                    measure=False,
                )
                for arm in (None, chunk)
            ]
    print(f"# self-test on {device}; no peak is read")
    print(HEADER)
    for arm in arms:
        print(table_row(arm))
    for arm in arms:
        name = "off" if arm.chunk is None else f"chunk {arm.chunk}"
        print(f"# {name}: predicted {arm.predicted} bytes, step loss {arm.loss!r}")
    checks = _self_test_checks(arms[0], arms[1], chunk)
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    failed = sum(not passed for _, passed in checks)
    print(f"self-test: {len(checks) - failed} passed, {failed} failed (no peak measured)")
    return EXIT_OK if failed == 0 else EXIT_UNDER


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base", help="model id or local path (required unless --self-test)")
    parser.add_argument("--seqs", default="512,1024,2048", help="comma-separated token counts")
    parser.add_argument("--chunk", type=int, default=128, help="training.stream_loss_chunk")
    parser.add_argument("--quantization", default="none", choices=["none", "4bit"])
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the protocol on a tiny local Llama, reading no peak (no GPU, no download)",
    )
    # One arm, run by the sweep in a process of its own.
    parser.add_argument("--arm", help=argparse.SUPPRESS)
    parser.add_argument("--arm-out", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    from soup_cli.utils.config_bounds import MAX_STREAM_LOSS_CHUNK, MIN_STREAM_LOSS_CHUNK

    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    if not args.base:
        print("ERROR: --base is required unless --self-test is used")
        return EXIT_USAGE
    if args.arm is not None:
        if args.arm_out is None:
            print("ERROR: --arm needs --arm-out")
            return EXIT_USAGE
        try:
            seq, chunk = parse_arm(args.arm)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return EXIT_USAGE
        return run_arm(args.base, seq, chunk, args.quantization, args.arm_out)
    if not MIN_STREAM_LOSS_CHUNK <= args.chunk <= MAX_STREAM_LOSS_CHUNK:
        print(
            f"ERROR: --chunk must be between {MIN_STREAM_LOSS_CHUNK} and "
            f"{MAX_STREAM_LOSS_CHUNK}, the bounds of training.stream_loss_chunk"
        )
        return EXIT_USAGE
    try:
        seqs = [int(part) for part in args.seqs.split(",")]
    except ValueError:
        print(f"ERROR: --seqs must be comma-separated integers; got {args.seqs!r}")
        return EXIT_USAGE
    if not seqs or any(seq <= 0 for seq in seqs):
        print("ERROR: every --seqs entry must be positive")
        return EXIT_USAGE

    import torch

    if not torch.cuda.is_available():
        print("ERROR: CUDA required: every figure in this table is a device peak (try --self-test)")
        return EXIT_USAGE
    base = os.path.abspath(args.base) if os.path.exists(args.base) else args.base
    return sweep(base, seqs, args.chunk, args.quantization)


if __name__ == "__main__":
    sys.exit(main())
