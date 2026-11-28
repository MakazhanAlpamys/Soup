"""Independent bounded-memory replay of the unchanged elementwise capture gate."""

from __future__ import annotations

from pathlib import Path


def compare(reference: Path, candidate: Path) -> dict:
    import torch
    from safetensors import safe_open

    groups, losses = {}, []
    equal = True
    with safe_open(reference, framework="pt", device="cpu") as first, safe_open(
        candidate, framework="pt", device="cpu",
    ) as second:
        names = first.keys()
        if set(names) != set(second.keys()):
            raise ValueError("trace keys differ")
        if "loss" not in names:
            raise KeyError("loss")
        for name in names:
            group = name.split("/")[0]
            stats = groups.setdefault(group, {"tensors": 0, "failed_tensors": 0,
                                             "max_abs": 0.0, "failures": []})
            stats["tensors"] += 1
            left_slice, right_slice = first.get_slice(name), second.get_slice(name)
            shape = left_slice.get_shape()
            if shape != right_slice.get_shape():
                raise ValueError("trace shapes differ")
            if len(shape) in {2, 3} and shape[-1] > 10000:
                ranges = [(start, min(start + 16, shape[-2]))
                          for start in range(0, shape[-2], 16)]
            else:
                ranges = [None]
            failed = False
            for bounds in ranges:
                if bounds is None:
                    value, other = first.get_tensor(name), second.get_tensor(name)
                elif len(shape) == 3:
                    value = left_slice[:, bounds[0]:bounds[1], :]
                    other = right_slice[:, bounds[0]:bounds[1], :]
                else:
                    value = left_slice[bounds[0]:bounds[1], :]
                    other = right_slice[bounds[0]:bounds[1], :]
                exact = torch.equal(value, other)
                equal &= exact
                finite = bool(torch.isfinite(value).all() and torch.isfinite(other).all())
                close = exact if group in {"batch", "initial"} else torch.allclose(
                    value, other, rtol=1e-2, atol=1e-5,
                )
                failed |= not (finite and close)
                stats["max_abs"] = max(
                    stats["max_abs"], float((value.double() - other.double()).abs().max()),
                )
                if name == "loss":
                    losses = [value.tolist(), other.tolist()]
            if failed:
                stats["failed_tensors"] += 1
                stats["failures"].append(name)
        loss_match = all(f"{a:.3f}" == f"{b:.3f}"
                         for a, b in zip(*losses, strict=True))
    return {"passed": loss_match and all(row["failed_tensors"] == 0 for row in groups.values()),
            "loss_three_decimals": loss_match, "groups": groups, "rtol": 1e-2,
            "atol": 1e-5, "bitwise_equal": bool(equal), "tensor_count": len(names),
            "replay": "elementwise equivalent chunked replay; primary remote comparator unchanged"}
