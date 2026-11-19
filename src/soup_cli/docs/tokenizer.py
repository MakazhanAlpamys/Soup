"""Deterministic, lightweight domain-aware technical regex tokenizer.

Tokenizes technical text preserving CLI flags, code identifiers, file paths,
version numbers, and sub-words without introducing external dependencies.
"""

from __future__ import annotations

import re

VERSION_RE = re.compile(r"^v?\d+(?:\.\d+)+$")
STRIP_PUNCT = "\"'`()[]{}<>,;?!~*#%^&|=+"


def tokenize(text: str) -> list[str]:
    """Tokenize technical text into terms for lexical retrieval."""
    if not text:
        return []

    text = text.lower()
    raw_tokens = text.split()
    tokens: list[str] = []

    for raw in raw_tokens:
        t = raw.strip(STRIP_PUNCT)
        while t and t[-1] in ".:,/":
            if t.endswith(".") and not VERSION_RE.match(t):
                t = t[:-1]
            elif t.endswith(":") or t.endswith(",") or t.endswith("/"):
                t = t[:-1]
            else:
                break

        if not t or not any(c.isalnum() for c in t):
            continue

        if t.startswith("--") and "=" in t:
            candidate_tokens = list(t.split("=", 1))
        else:
            candidate_tokens = [t]

        for cand in candidate_tokens:
            cand = cand.strip(STRIP_PUNCT)
            if not cand or not any(c.isalnum() for c in cand):
                continue

            if VERSION_RE.match(cand):
                tokens.append(cand)
                if cand.startswith("v") and len(cand) > 1:
                    tokens.append(cand[1:])
                continue

            if cand.startswith("--"):
                clean_flag = cand.rstrip(".,;:")
                tokens.append(clean_flag)
                parts = clean_flag[2:].split("-")
                for p in parts:
                    if len(p) > 1 and any(c.isalnum() for c in p):
                        tokens.append(p)
                continue

            if "/" in cand:
                tokens.append(cand)
                segments = cand.split("/")
                for seg in segments:
                    if seg and any(c.isalnum() for c in seg) and seg != cand:
                        tokens.append(seg)
                        if any(s in seg for s in ("_", "-", ".")):
                            for sp in re.split(r"[_.-]", seg):
                                if sp and any(c.isalnum() for c in sp) and sp != seg:
                                    tokens.append(sp)
                continue

            if any(sep in cand for sep in ("_", "-", ".")):
                tokens.append(cand)
                sub_parts = re.split(r"[_.-]", cand)
                for sp in sub_parts:
                    if sp and any(c.isalnum() for c in sp) and sp != cand:
                        tokens.append(sp)
            else:
                tokens.append(cand)

    return tokens
