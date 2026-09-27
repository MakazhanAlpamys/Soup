#!/usr/bin/env python3
"""Regenerate tests/fixtures/recipe_base_architectures.json (#1070).

Records, for EVERY base a shipped recipe names, the ``model_type`` and the
routed-expert count read off that base's ``config.json`` on the Hub.

Why every base and not only MoE ones: the first version of #1070's ratchet found
MoE recipes by their MoE *flags*, then patched in a hand-kept list of the
flagless ones it knew about. That is the same flag-blindness one level down --
a new flagless MoE recipe passed straight through (#1102 review, F3). A record
of every base, checked for completeness against the catalogue, makes a new base
fail the ratchet until someone classifies it, whatever flags its recipe sets.

It reads the raw JSON with ``hf_hub_download`` and never instantiates anything,
so it executes no remote code -- which is also why it can read a base that
needs ``trust_remote_code`` (DeepSeek-OCR) that a model build would refuse.

A base whose ``config.json`` cannot be read is recorded with the reason rather
than dropped. Anonymously that includes every gated base, so run with
``HF_TOKEN`` set to classify those; the record says which ones it could not.

Deliberately a separate, explicit step: run it and review the diff.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "recipe_base_architectures.json"

#: Keys that count routed experts, across the architectures in the catalogue.
#: ``num_experts_per_tok`` is deliberately absent: it is the ACTIVE count, and a
#: dense model can carry it at 1 without having experts at all.
_EXPERT_KEYS = ("n_routed_experts", "num_local_experts", "num_experts", "moe_num_experts")


def routed_experts(config: dict) -> int:
    """The largest routed-expert count anywhere in the config, 0 if none.

    Searched recursively, because a vision-language wrapper keeps the language
    tower's MoE settings in ``text_config`` / ``language_config`` / ``llm_config``
    (DeepSeek-OCR's 64 experts are one level down).
    """
    best = 0
    for key, value in config.items():
        if key in _EXPERT_KEYS and isinstance(value, int) and not isinstance(value, bool):
            best = max(best, value)
        elif isinstance(value, dict):
            best = max(best, routed_experts(value))
    return best


def classify(base: str) -> dict:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (
        EntryNotFoundError,
        GatedRepoError,
        RepositoryNotFoundError,
    )

    try:
        path = hf_hub_download(base, "config.json")
    except GatedRepoError:  # before RepositoryNotFoundError: it subclasses it (#677)
        return {"unresolvable": "gated; set HF_TOKEN to classify"}
    except RepositoryNotFoundError:
        return {"unresolvable": "repo not found or not publicly readable (see #677)"}
    except EntryNotFoundError:
        return {"unresolvable": "the repo has no config.json"}
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"model_type": config.get("model_type"), "routed_experts": routed_experts(config)}


def main() -> int:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    import yaml

    from soup_cli.recipes.catalog import list_recipes

    bases = sorted({yaml.safe_load(r.yaml_str).get("base") for r in list_recipes()})
    record = {}
    for base in bases:
        record[base] = classify(base)
        print(f"{base:55} {record[base]}", file=sys.stderr)
    _FIXTURE.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    moe = sum(1 for v in record.values() if v.get("routed_experts", 0) > 1)
    unresolvable = sum(1 for v in record.values() if "unresolvable" in v)
    print(f"{len(record)} bases, {moe} MoE, {unresolvable} unresolvable", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
