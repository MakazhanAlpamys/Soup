# Security Policy

## Supported Versions

Security fixes are shipped for the latest published release. We recommend always
running the most recent version from PyPI (`pip install -U soup-cli`).

| Version | Supported          |
| ------- | ------------------ |
| 0.70.x  | :white_check_mark: |
| < 0.70  | :x:                |

## Reporting a Vulnerability

Please report security issues **privately** — do not open a public GitHub issue
for anything security-sensitive.

- Preferred: open a private report via
  [GitHub Security Advisories](https://github.com/MakazhanAlpamys/Soup/security/advisories/new).

We aim to acknowledge reports within 5 business days and to ship a fix or
mitigation for confirmed, in-scope issues as promptly as is practical. When
reporting, please include:

- the affected version(s) and platform,
- a minimal reproduction or proof of concept,
- the impact you observed.

## Scope

Soup is a local-first CLI for fine-tuning LLMs. The threat model assumes the
operator runs Soup on their own machine with their own data. Representative
in-scope issues:

- path traversal or arbitrary file read/write from user-supplied config,
  dataset, or artifact paths;
- SSRF in the synthetic-data providers, inference server, or hub/endpoint
  validators;
- command, Modelfile, or Jinja chat-template injection;
- secret leakage in logs, crash bundles, or generated artifacts;
- sandbox escape in the RLVR code-execution reward path.

Out of scope: vulnerabilities in third-party model weights or datasets you
choose to load, and issues that require an already-compromised host.

## Disclosure

We practice coordinated disclosure. Once a fix is released we credit the
reporter in the release notes, unless anonymity is requested.

> A detailed, per-version log of historical security hardening previously lived
> in this file. It now lives in the project's git history and in the
> [GitHub Releases](https://github.com/MakazhanAlpamys/Soup/releases) notes.
