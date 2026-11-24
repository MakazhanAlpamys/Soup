# SPDX-License-Identifier: Apache-2.0
#
# Origin: Soup Kernel Challenge (Decentrathon 6.0), the submission of team Luck
# (BAITC-Hacks/hack-e79529fc-luck@b3f1225f); the int8 gate/up packs follow ideas seen in the
# solutions of teams Null Pointers, 21, DarkDaur and OrbiTeam.
# Modified for Soup: the tables are parked after the model has loaded (the load-time hook that
# read them straight into host memory is not ported), the two module singletons became a
# :class:`HostTables` that belongs to one run, and pinning goes through the run's PinnedPool.
"""The embedding and lm_head weights in pinned host memory instead of on the GPU, from the first
step the memory plan asks for it.

They are the largest tensors the run keeps on the GPU besides the 4-bit layers (445 MiB each for
Qwen2.5-1.5B, tied into one tensor; 1.0 GiB each for Qwen2.5-7B, two tensors), and the training
step reads only a little of them at a time: the embedding lookup needs one row per token, and the
head walks the whole lm_head matrix once per step, slab by slab (``triton/head.py``), which a copy
stream feeds from host memory while the previous slab is computed. So:

- ``copy_to_host``: a page-locked host copy of the (GPU, frozen) weight;
- ``activate``: the parameter's storage becomes that host copy and the GPU copy is freed (the
  parameter keeps its shape, dtype and identity; only its device changes). A sample of rows is
  compared first: any mismatch keeps the weight on the GPU;
- each step gathers the rows it needs (the tokens from the embedding, the labelled tokens'
  targets from the lm_head) on the CPU and copies them over in one transfer each (one for both
  when the two are the same tied tensor);
- anything else that needs the weight on the GPU (the stock forward for eval and generation, a
  checkpoint save) puts it back first: ``restore``.

Same values everywhere; the head's sums run in a different order (slabs), within the checks'
tolerances.
"""

from __future__ import annotations

import torch

#: The host-parked head copies the lm_head (445 MiB .. 1 GiB) host -> GPU a step, so it wants a
#: fast link (healthy PCIe gen3 x16 ~12 GB/s).
MIN_LINK_BYTES_PER_S = 8e9


class HostWeight:
    """One frozen weight [rows, hidden] (the embedding table, or the lm_head) and its host copy."""

    def __init__(self, pool):
        self.pool = pool
        self.host = None  # pinned copy of the weight (compute dtype)
        self.staging = None  # pinned rows gathered for one step
        self.active = False  # the parameter's storage is host

    def copy_to_host(self, weight):
        """A page-locked host copy of the (GPU, frozen) weight. False if it cannot be made."""
        if self.host is not None:
            return True
        if weight.dtype not in (torch.float16, torch.bfloat16) or not weight.is_cuda:
            return False
        self._copy_from_gpu(weight)
        return True

    def _copy_from_gpu(self, weight):  # pragma: no cover - needs a CUDA device (gpu tests)
        host = torch.empty(weight.shape, dtype=weight.dtype)
        if not self.pool.register(host):
            host = torch.empty(weight.shape, dtype=weight.dtype, pin_memory=True)
        stream = torch.cuda.Stream(weight.device)
        stream.wait_stream(torch.cuda.current_stream(weight.device))  # the load's writes first
        with torch.cuda.stream(stream):
            host.copy_(weight.detach(), non_blocking=True)
        stream.synchronize()
        self.host = host

    def activate(self, param):
        """The parameter's storage becomes the host copy; its GPU copy is freed. A sample of rows
        is compared first: any mismatch keeps the weight on the GPU. True if it is on the host."""
        if self.active:
            return True
        if self.host is None or param.shape != self.host.shape or param.dtype != self.host.dtype:
            return False
        with torch.no_grad():
            for rows in (slice(0, 64), slice(None, None, 997)):
                if not torch.equal(self.host[rows].to(param.device), param[rows]):
                    return False
        param.data = self.host
        self.active = True
        return True

    def restore(self, param, device):
        """The weight back on the GPU, for code that reads it there (the stock forward, a save)."""
        if self.active:
            param.data = self.host.to(device)
            self.active = False

    def drop_host(self):
        """After ``restore`` for good (the plan keeps the weight on the GPU): unlock and free the
        host copy."""
        if self.host is not None and not self.active:
            self.pool.unregister(self.host.data_ptr())
            self.host = None

    def gather(self, ids_cpu, device):
        """Rows of the weight for these token ids (CPU int64), on the GPU, in one copy."""
        n, hidden = ids_cpu.numel(), self.host.shape[1]
        if self.staging is None or self.staging.shape[0] < n:
            self.staging = torch.empty(max(n, 4096), hidden, dtype=self.host.dtype,
                                       pin_memory=True)
        rows = self.staging[:n]
        torch.index_select(self.host, 0, ids_cpu, out=rows)
        # the previous step's copy out of the staging buffer finished before this step's host sync
        return rows.to(device, non_blocking=True)

    def close(self):
        """Unlock the host copy (the weight must be back on the GPU first)."""
        self.drop_host()
        self.staging = None


class HostTables:
    """The embedding and the lm_head of one run; the same object twice where they are one tied
    tensor."""

    def __init__(self, pool, tied):
        self.embed = HostWeight(pool)
        self.head = self.embed if tied else HostWeight(pool)
        self.tied = tied

    def weights(self):
        """The distinct host weights (one or two)."""
        return [self.embed] if self.head is self.embed else [self.embed, self.head]

    def active_embed(self):
        """Whether the embedding lives in host memory now."""
        return self.embed.active

    def active_head(self):
        """Whether the lm_head lives in host memory now."""
        return self.head.active

    def any_active(self):
        """Whether either table lives in host memory now."""
        return self.embed.active or self.head.active

    def park(self, embed_param, head_param, embed=True, head=True):
        """Move the tables the plan keeps off the GPU: host copy first, then the parameter's
        storage. True if every one that was asked for is on the host now; a table that could not
        move stays on the GPU."""
        ok = True
        if embed:
            ok = self.embed.copy_to_host(embed_param) and self.embed.activate(embed_param) and ok
        if head and self.head is not self.embed:
            ok = self.head.copy_to_host(head_param) and self.head.activate(head_param) and ok
        return ok

    def restore(self, embed_param, head_param, device):
        """Both tables back on the GPU (the host copies stay for a later ``park``)."""
        self.embed.restore(embed_param, device)
        if self.head is not self.embed:
            self.head.restore(head_param, device)

    def gather(self, ids_cpu, device):
        """Rows of the embedding for these token ids (CPU int64), on the GPU, in one copy."""
        return self.embed.gather(ids_cpu, device)

    def gather_pair(self, ids_cpu, target_ids_cpu, device):
        """(embedding rows of ids_cpu, lm_head rows of target_ids_cpu) on the GPU: one copy where
        the two weights are one tied tensor, one copy each otherwise."""
        if self.head is self.embed:
            both = self.embed.gather(torch.cat([ids_cpu, target_ids_cpu]), device)
            return both[:ids_cpu.numel()], both[ids_cpu.numel():]
        return self.embed.gather(ids_cpu, device), self.head.gather(target_ids_cpu, device)

    def close(self):
        """Release the host copies (restore the tables first)."""
        for weight in self.weights():
            weight.close()
