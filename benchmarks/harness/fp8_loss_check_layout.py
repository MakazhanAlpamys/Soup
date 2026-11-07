"""Compare safetensors layouts of two shard caches (headers only; read-only).

Section 6 of ``benchmarks/probe-rtx5070-moonlight-fp8-source.md`` (the T6 diagnosis, run after
the verdict): arm A's cache against arm B's, per layer file: header length, data start, size,
metadata, tensor order, every tensor's dtype, shape, length and absolute offset (the large
tensors' offsets mod 4096 and mod 512 are printed for layer 10), then ``index.json`` field by
field. The paths are the run's. Output: ``results/probe-rtx5070/fp8-loss-check/t6-layout.txt``.
"""
import json
import os
import struct
import sys

A = "D:/soup-bench/soup-cache/D___soup-bench__models__Moonlight-16B-A3B"
B = "D:/soup-bench/fp8-loss-check/soup-cache/D___soup-bench__models__Moonlight-16B-A3B-fp8block"


def header(path):
    with open(path, "rb") as handle:
        (n,) = struct.unpack("<Q", handle.read(8))
        meta = json.loads(handle.read(n))
    meta_info = meta.pop("__metadata__", None)
    base = 8 + n
    order = sorted(meta.items(), key=lambda kv: kv[1]["data_offsets"][0])
    tensors = [
        (name, info["dtype"], tuple(info["shape"]), base + info["data_offsets"][0],
         info["data_offsets"][1] - info["data_offsets"][0])
        for name, info in order
    ]
    return n, base, meta_info, tensors, os.path.getsize(path)


def main():
    names = sorted(f for f in os.listdir(A) if f.endswith(".safetensors"))
    names_b = sorted(f for f in os.listdir(B) if f.endswith(".safetensors"))
    assert names == names_b, "file sets differ"
    summary = []
    for name in names:
        ha, hb = header(os.path.join(A, name)), header(os.path.join(B, name))
        diffs = []
        if ha[0] != hb[0]:
            diffs.append(f"header_len {ha[0]} vs {hb[0]}")
        if ha[2] != hb[2]:
            diffs.append(f"metadata {ha[2]} vs {hb[2]}")
        if ha[4] != hb[4]:
            diffs.append(f"size {ha[4]} vs {hb[4]}")
        if [t[0] for t in ha[3]] != [t[0] for t in hb[3]]:
            diffs.append("tensor order differs")
        for ta, tb in zip(ha[3], hb[3]):
            if ta[1:3] != tb[1:3] or ta[4] != tb[4]:
                diffs.append(f"{ta[0]}: dtype/shape/len {ta[1:3]},{ta[4]} vs {tb[1:3]},{tb[4]}")
            if ta[3] != tb[3]:
                diffs.append(f"{ta[0]}: offset {ta[3]} vs {tb[3]}")
        big_a = [(t[0], t[3] % 4096, t[3] % 512, t[4]) for t in ha[3] if t[4] >= 1 << 20]
        big_b = [(t[0], t[3] % 4096, t[3] % 512, t[4]) for t in hb[3] if t[4] >= 1 << 20]
        summary.append({
            "file": name,
            "header_len": [ha[0], hb[0]],
            "data_start": [ha[1], hb[1]],
            "data_start_mod4096": [ha[1] % 4096, hb[1] % 4096],
            "size": [ha[4], hb[4]],
            "metadata": [ha[2], hb[2]],
            "n_tensors": [len(ha[3]), len(hb[3])],
            "diffs": diffs[:8],
            "big_tensors_mod": (
                {"A": big_a, "B": big_b} if name == "layer_010.safetensors" else None
            ),
        })
    for row in summary:
        if row["diffs"] or row["file"] in ("layer_000.safetensors", "layer_010.safetensors"):
            print(json.dumps(row, indent=1))
    print("files with any difference:", [r["file"] for r in summary if r["diffs"]])
    ia = json.load(open(os.path.join(A, "index.json")))
    ib = json.load(open(os.path.join(B, "index.json")))
    for key in sorted(set(ia) | set(ib)):
        if key in ("source_files", "source_fingerprint", "quant_specs", "layer_keys"):
            if ia.get(key) != ib.get(key):
                note = "(expected for source_*)" if key.startswith("source") else ""
                print("index differs:", key, note)
            continue
        if ia.get(key) != ib.get(key):
            print("index differs:", key, ia.get(key), "vs", ib.get(key))
    print(
        "quant_specs equal:", ia["quant_specs"] == ib["quant_specs"],
        "layer_keys equal:", ia["layer_keys"] == ib["layer_keys"],
    )


if __name__ == "__main__":
    sys.exit(main())
