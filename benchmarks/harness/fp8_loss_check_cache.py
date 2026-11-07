"""Will ``soup train`` reuse a layer cache, or re-shard? Asked the way ``soup train`` asks it.

Part of ``benchmarks/probe-rtx5070-moonlight-fp8-source.md``. Read-only and CPU-only: it never
creates, writes or deletes anything (``resolve_shard_dir`` makes the folder, so this computes
the same path without calling it). Run it from the run's cwd with the run's
``SOUP_LAYER_STREAM_CACHE_DIR``, ``PYTHONPATH`` pointing at the tree under test, and
``CUDA_VISIBLE_DEVICES=-1``:

    python benchmarks/harness/fp8_loss_check_cache.py --base <base dir> \
        [--expect reuse|reshard] [--json <report.json>]

What ``soup train`` decides, restated from ``trainer/stream_setup.py`` and
``utils/layer_shard.py`` at the commit under test, for a ``quantization: 4bit``, fused-layout,
unstriped run on a CUDA card with bf16:

- cache root: ``SOUP_LAYER_STREAM_CACHE_DIR`` when it is free of control characters and lies
  under ``$HOME``, the cwd or ``$TMPDIR``; otherwise, SILENTLY, ``~/.soup/layer-stream``
  (``resolve_cache_root``). This script exits 2 if the variable is unset or was dropped, since
  that run would write ~30 GB on the system drive.
- shard dir: ``<root>/<model_slug(base)>``, the slug made from the ``base`` string as the
  config spells it (``D:/soup-bench/models/X`` -> ``D___soup-bench__models__X``).
- reuse: ``inspect_shard_cache`` with dtype ``bfloat16``, quant ``nf4``, double quant on,
  quantisation device ``cuda``, external mode ``""``, expert layout ``""`` (fused), no
  stripe roots, and the source identity ``soup train`` computes for ``deepseek_v3``: name,
  size and mtime of every checkpoint file the index names, plus ``config.json``.

The dtype and device are the RTX 5070's (bf16-capable CUDA card); they are fixed here because
this script must not touch the GPU. ``--expect`` turns the answer into an exit code: 0 when it
matches, 1 when it does not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

GB = 1e9
CACHE_ENV = "SOUP_LAYER_STREAM_CACHE_DIR"
STRIPE_ENV = "SOUP_LAYER_STREAM_STRIPE_DIRS"
STREAM_DTYPE = "bfloat16"
QUANT_DEVICE_KIND = "cuda"
ARCH = "deepseek_v3"


def _listing_digest(folder: str) -> tuple[int, int, str]:
    """(files, bytes, sha256 of 'name|size|mtime_ns' lines): re-run after a run to show the
    folder was only read."""
    digest = hashlib.sha256()
    count = total = 0
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        stat = os.stat(path)
        digest.update(f"{name}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
        count += 1
        total += stat.st_size
    return count, total, digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--base", required=True, help="the config's `base`, spelled as there")
    parser.add_argument("--expect", choices=("reuse", "reshard"), default=None)
    parser.add_argument("--json", default=None, help="also write the answer here")
    args = parser.parse_args(argv)

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        print("refusing: set CUDA_VISIBLE_DEVICES=-1 (this check must not touch the GPU)")
        return 2
    if os.environ.get(STRIPE_ENV):
        print(f"refusing: {STRIPE_ENV} is set; this check covers unstriped runs only")
        return 2

    import soup_cli
    from soup_cli.utils.layer_shard import (
        QUANT_NF4,
        _discover_safetensors,
        _fingerprint_components,
        _source_file_components,
        checkpoint_source_components,
        default_layer_stream_cache_dir,
        estimate_shard_cache_bytes,
        inspect_shard_cache,
        model_slug,
        read_shard_index,
        resolve_cache_root,
        source_identity_includes_config,
    )

    env_root = os.environ.get(CACHE_ENV)
    root = resolve_cache_root()
    report: dict = {
        "soup_cli": soup_cli.__file__,
        "cwd": os.getcwd(),
        "env": {CACHE_ENV: env_root},
        "cache_root": root,
        "base": args.base,
    }
    if not env_root or os.path.normcase(root) == os.path.normcase(
        os.path.realpath(default_layer_stream_cache_dir())
    ):
        print(
            f"refusing: {CACHE_ENV} is unset or was rejected (it must lie under $HOME, the cwd "
            f"or $TMPDIR); soup train would shard into {root}"
        )
        return 2

    shard_dir = os.path.join(root, model_slug(args.base))
    shards = _discover_safetensors(args.base)
    components = checkpoint_source_components(
        args.base,
        _source_file_components(shards),
        include_config=source_identity_includes_config(ARCH),
    )
    fingerprint = _fingerprint_components(components)
    report.update(
        shard_dir=shard_dir,
        shard_dir_exists=os.path.isdir(shard_dir),
        source_files=len(components),
        source_fingerprint=fingerprint,
    )

    reusable = False
    reason = "shard folder does not exist"
    if os.path.isdir(shard_dir):
        index, reason = inspect_shard_cache(
            shard_dir,
            STREAM_DTYPE,
            fingerprint,
            components,
            QUANT_NF4,
            True,
            QUANT_DEVICE_KIND,
            "",
            stripe_roots=(),
            expert_layout="",
        )
        reusable = index is not None
        files, size, listing = _listing_digest(shard_dir)
        report.update(shard_files=files, shard_bytes=size, shard_listing_sha256=listing)
        if index is not None:
            report.update(
                index_soup_version=index.soup_version,
                index_quant_device=index.quant_device,
                index_format_version=index.format_version,
            )
    report.update(reusable=reusable, reason=reason)

    # What a re-shard would write: the sharder's own estimate from the source headers. The
    # NF4 suffix set is the one a fused NF4 Moonlight cache records (the meta probe's 10
    # decoder linears): this cache's index when it has one, else the same 10 names spelled out.
    if not reusable:
        suffixes = None
        try:
            suffixes = tuple(read_shard_index(shard_dir).quant_specs)
        except (OSError, ValueError, KeyError, TypeError):
            suffixes = None
        if not suffixes:
            suffixes = (
                "self_attn.q_proj.weight",
                "self_attn.kv_a_proj_with_mqa.weight",
                "self_attn.kv_b_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
                "mlp.shared_experts.gate_proj.weight",
                "mlp.shared_experts.up_proj.weight",
                "mlp.shared_experts.down_proj.weight",
            )
        with open(os.path.join(args.base, "config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        estimate = estimate_shard_cache_bytes(
            shards,
            dtype=STREAM_DTYPE,
            arch=ARCH,
            quant=QUANT_NF4,
            quant_suffixes=suffixes,
            double_quant=True,
            expert_layout="",
            num_hidden_layers=int(config["num_hidden_layers"]),
            quantization_config=config.get("quantization_config"),
        )
        report["reshard_estimate_bytes"] = estimate.total_bytes if estimate else None
    existing = root
    while not os.path.isdir(existing) and os.path.dirname(existing) != existing:
        existing = os.path.dirname(existing)
    report["cache_volume_free_bytes"] = shutil.disk_usage(existing).free

    print(json.dumps(report, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    if args.expect is None:
        return 0
    return 0 if reusable == (args.expect == "reuse") else 1


if __name__ == "__main__":
    sys.exit(main())
