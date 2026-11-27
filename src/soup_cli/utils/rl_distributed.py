"""Ordered, bounded exchanges for replicated RL callbacks (#3245)."""

from __future__ import annotations

import json

_TAGS = ("checkpoint.prepare", "checkpoint.saved", "pid.prepare", "pid.decision", "pid.apply")
_MAX_PAYLOAD_BYTES = 64 * 1024


def peft_policy(model):
    """Find the policy in DDP and TRL PPO's policy/value wrapper."""
    from peft import PeftModel
    from torch.nn.parallel import DistributedDataParallel

    if isinstance(model, DistributedDataParallel):
        model = model.module
    if isinstance(model, PeftModel):
        return model
    policy = getattr(model, "policy", None)
    return policy if isinstance(policy, PeftModel) else None


def active_group(enabled: bool):
    """Environment rank variables alone never authorize a collective."""
    if not enabled:
        return None
    import torch.distributed as distributed

    if (distributed.is_available() and distributed.is_initialized()
            and distributed.get_world_size() > 1):
        return distributed
    return None


def exchange(distributed, step: int, tag: str, payload: dict) -> list[dict]:
    """Validate step/phase before exchanging JSON; never deserialize peer pickle."""
    import torch

    try:
        encoded = json.dumps(payload, allow_nan=False).encode("utf-8")
        if len(encoded) > _MAX_PAYLOAD_BYTES:
            raise ValueError("RL exchange payload exceeds 65536 bytes")
    except (TypeError, ValueError) as exc:
        encoded = json.dumps({
            "error": f"invalid RL exchange payload: {type(exc).__name__}"
        }).encode()
    device = torch.device("cuda", torch.cuda.current_device()) if (
        distributed.get_backend() == "nccl"
    ) else torch.device("cpu")
    header = torch.tensor([step, _TAGS.index(tag), len(encoded)], dtype=torch.int64, device=device)
    headers = [torch.empty_like(header) for _rank in range(distributed.get_world_size())]
    distributed.all_gather(headers, header)
    values = [entry.tolist() for entry in headers]
    if any(entry[:2] != values[0][:2] for entry in values):
        phases = ", ".join(f"rank {rank}: step {entry[0]} {_TAGS[entry[1]]}"
                           for rank, entry in enumerate(values))
        raise RuntimeError(f"RL collective exchange mismatch ({phases}); stopping all ranks")
    size = max(entry[2] for entry in values)
    if size < 1 or size > _MAX_PAYLOAD_BYTES:
        raise RuntimeError("RL collective payload size is invalid; stopping all ranks")
    buffer = torch.zeros(size, dtype=torch.uint8, device=device)
    buffer[:len(encoded)] = torch.tensor(list(encoded), dtype=torch.uint8, device=device)
    buffers = [torch.empty_like(buffer) for _rank in values]
    distributed.all_gather(buffers, buffer)
    return [json.loads(bytes(entry[:header_values[2]].tolist()))
            for entry, header_values in zip(buffers, values, strict=True)]


def first_error(payloads: list[dict]) -> str | None:
    """Every rank reports the same first failure."""
    return next((f"rank {rank}: {entry['error']}" for rank, entry in enumerate(payloads)
                 if entry.get("error")), None)
