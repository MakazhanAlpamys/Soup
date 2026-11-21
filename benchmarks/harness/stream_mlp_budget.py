"""Fixed B allocation and repaired observation; original cache arithmetic reused."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from stream_mlp_retention import ProjectionObserver as OriginalObserver
from stream_mlp_retention import candidate_plan as validate_original_census


def candidate_plan(native: dict, *, include_mlp: bool = True) -> dict:
    from soup_cli.utils.layer_stream import retained_layer_charge

    validate_original_census(native)
    q_raw, m_raw = 41943040, 58720256
    charge = lambda raw: retained_layer_charge(  # noqa: E731
        raw, native["peak_reserved"], native["peak_allocated"]
    )
    outputs = 22 if include_mlp else 0
    residual = native["budget_bytes"] - 6 * native["full_layer_charge"]
    charged = 26 * charge(q_raw) + outputs * charge(m_raw)
    if residual - charged < 64 * 2**20:
        raise ValueError("candidate B census leaves insufficient charged slack")
    return dict(native_k=11, effective_k=6, retained_indices=list(range(26, 32)),
                qkvo_indices=list(range(26)), mlp_indices=list(range(15, 26)) if outputs else [],
                raw_bytes=26 * q_raw + outputs * m_raw, charged_bytes=charged,
                residual_bytes=residual, slack_bytes=residual - charged,
                cache_tensors=104 + outputs)


class ProjectionObserver(OriginalObserver):
    """A single active phase is visible to CUDA autograd worker threads."""

    _shared_phase = "outside"

    @contextmanager
    def phase(self, phase: str) -> Any:
        if phase not in ("forward", "backward") or self._shared_phase != "outside":
            raise ValueError("invalid or concurrent model phase")
        self._shared_phase = phase
        try:
            yield
        finally:
            self._shared_phase = "outside"

    def _wrap(self, original: Any, index: int, name: str) -> Any:
        def forward(instance: Any, *args: Any, **kwargs: Any) -> Any:
            from torch.profiler import record_function

            phase = self._shared_phase
            if phase not in ("forward", "backward"):
                raise RuntimeError("projection outside explicit model phase")
            with record_function(f"soup2162|{phase}|{index}|{name}"):
                return original(*args, **kwargs)

        return forward


def preserve_events(events: list[Any], path: Path) -> None:
    records = []
    for event in events:
        if event.name not in ("aten::mm", "aten::addmm", "bitsandbytes::gemm_4bit"):
            continue
        ancestors = []
        parent = event.cpu_parent
        while parent is not None:
            ancestors.append(parent.name)
            parent = parent.cpu_parent
        records.append(dict(name=event.name, ancestors=ancestors,
                            children=[child.name for child in event.cpu_children],
                            device_us=event.device_time_total,
                            self_cpu_us=event.self_cpu_time_total))
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(events=records), handle)
