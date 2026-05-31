"""Local Model Registry / Provenance Vault (v0.26.0 Part A).

Tracks fine-tuning artifacts with lineage, making every run discoverable,
diff-able, and promotable. Built on the same SQLite infrastructure as
:mod:`soup_cli.experiment.tracker` but with a separate database file so the
two lifecycles can evolve independently.
"""

from __future__ import annotations

from soup_cli.registry.hashing import hash_config, hash_entry, hash_file
from soup_cli.registry.store import (
    RegistryStore,
    validate_name,
    validate_tag,
)

__all__ = [
    "RegistryStore",
    "hash_config",
    "hash_entry",
    "hash_file",
    "validate_name",
    "validate_tag",
]
