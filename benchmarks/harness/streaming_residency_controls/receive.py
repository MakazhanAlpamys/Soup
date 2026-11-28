"""Resume authenticated immutable archives after disconnects; accept only exact SHA."""

import hashlib
from pathlib import Path


def receive(target, size, sha256, fetch, *, attempts=12):
    """fetch(offset, append_sink) returns a transport code; bytes are authoritative."""
    target = Path(target)
    if type(size) is not int or size < 0 or attempts < 1 or target.is_symlink():
        raise ValueError("invalid immutable transfer")
    target.parent.mkdir(parents=True, exist_ok=True)
    offsets, codes = [], []
    for _attempt in range(attempts + 1):
        offset = target.stat().st_size if target.exists() else 0
        if offset > size:
            raise ValueError("archive exceeds authenticated size")
        if offset == size:
            hashed = hashlib.sha256()
            with target.open("rb") as source:
                for chunk in iter(lambda: source.read(4 * 2**20), b""):
                    hashed.update(chunk)
            if hashed.hexdigest() != sha256:
                raise ValueError("complete archive SHA mismatch; do not ACK")
            return {"sha256_verified": True, "bytes": size, "sha256": sha256,
                    "attempt_offsets": offsets, "transport_codes": codes}
        if len(offsets) == attempts:
            break
        offsets.append(offset)
        with target.open("ab") as sink:
            try:
                codes.append(fetch(offset, sink))
            except (ConnectionError, TimeoutError):
                codes.append("connection_or_timeout")
    raise RuntimeError("resumable transfer exhausted attempts; partial bytes preserved")
