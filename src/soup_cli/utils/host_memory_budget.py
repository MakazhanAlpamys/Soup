"""Proposed C3/A4 allocation contract; accounting, not an OS/process memory limit.

A consumer reserves before allocating and releases only after asynchronous users
finish and owned storage is dropped. Fixed reservations cover untracked consumers.
"""

from __future__ import annotations

import threading


def _check_bytes(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("memory amounts must be non-negative integer bytes")


class HostMemoryBudget:
    def __init__(self, limit_bytes: int, reserved_bytes: int = 0) -> None:
        _check_bytes(limit_bytes)
        _check_bytes(reserved_bytes)
        if not limit_bytes or reserved_bytes > limit_bytes:
            raise ValueError("positive host limit must cover fixed reservations")
        self._limit_bytes = limit_bytes
        self._reserved_bytes = reserved_bytes
        self._leases: dict[str, MemoryLease] = {}
        self._lock = threading.RLock()

    @property
    def limit_bytes(self) -> int:
        return self._limit_bytes

    @property
    def reserved_bytes(self) -> int:
        return self._reserved_bytes

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self.reserved_bytes + sum(item.nbytes for item in self._leases.values())

    @property
    def available_bytes(self) -> int:
        with self._lock:
            return self.limit_bytes - self.used_bytes

    def reserve(self, owner: str, nbytes: int) -> MemoryLease:
        _check_bytes(nbytes)
        if not isinstance(owner, str) or not owner:
            raise ValueError("a nonempty reservation owner is required")
        with self._lock:
            if owner in self._leases:
                raise ValueError(f"duplicate host reservation owner {owner!r}")
            if nbytes > self.available_bytes:
                raise MemoryError(
                    f"{owner}: needs {nbytes} bytes; {self.available_bytes} available"
                )
            lease = MemoryLease(self, owner, nbytes)
            self._leases[owner] = lease
            return lease


class MemoryLease:
    def __init__(self, budget: HostMemoryBudget, owner: str, nbytes: int) -> None:
        self._budget = budget
        self._owner = owner
        self._nbytes = nbytes
        self._closed = False

    @property
    def owner(self) -> str:
        return self._owner

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def resize(self, nbytes: int) -> None:
        _check_bytes(nbytes)
        with self._budget._lock:
            if self._closed:
                raise RuntimeError("memory lease is closed")
            if nbytes - self.nbytes > self._budget.available_bytes:
                raise MemoryError(f"{self.owner}: growing the reservation exceeds the host budget")
            self._nbytes = nbytes

    def close(self) -> None:
        with self._budget._lock:
            if self._closed:
                return
            del self._budget._leases[self.owner]
            self._closed = True
