# Security Policy

## Supported Versions

Security fixes are shipped for the latest published release. 0.76.x, the last line
that runs on Python 3.10 and exports AWQ / GPTQ, also receives security fixes only,
until 0.78.0 is released. We recommend always running the most recent version from
PyPI (`pip install -U soup-cli`).

| Version | Supported                                     |
| ------- | --------------------------------------------- |
| 0.77.x  | :white_check_mark:                            |
| 0.76.x  | :white_check_mark: security fixes only, until 0.78.0 |
| < 0.76  | :x:                                           |

## Reporting a Vulnerability

Please report security issues **privately** — do not open a public GitHub issue
for anything security-sensitive.

- Preferred: open a private report via
  [GitHub Security Advisories](https://github.com/MakazhanAlpamys/Soup/security/advisories/new).
- If you cannot use GitHub, email **team@trysoup.dev** (the project address), or
  **makazanalpamys@gmail.com** (the maintainer's personal address) if that bounces.
  Please do not report security issues in Discord — it is a public channel.

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
- command, Modelfile, Jinja chat-template, or systemd/launchd unit injection;
- secret leakage in logs, crash bundles, or generated artifacts;
- sandbox escape in the RLVR code-execution reward path.

Out of scope:

- vulnerabilities in third-party model weights or datasets you choose to load;
- issues that require an already-compromised host;
- DNS and email configuration of `trysoup.dev` — a missing or permissive
  DMARC / SPF / DKIM record, and anything else established by a public DNS
  query. These are worth fixing and we do fix them, but they are not findings
  in Soup and they are not eligible for anything.

### Telemetry and Network Egress

Soup is local-first. Telemetry is opt-in and off by default: it is considered only when `SOUP_TELEMETRY` is `1`, `true`, `yes` or `on`, and `soup --no-telemetry <command>` turns it off for one invocation. **As released, no telemetry is sent:** the bundled PostHog key is a placeholder, so an opted-in command prints `Notice: telemetry is not yet live.` and makes no network request. Only an operator who sets their own `SOUP_POSTHOG_KEY` (and optionally `SOUP_POSTHOG_ENDPOINT`, which must be HTTPS and is refused for loopback, private and internal hosts) gets one HTTPS POST at command exit. The payload carries `soup_version`, `command` (the top-level command name, checked against the registered commands, otherwise `(unknown)`), `python` (major.minor), `os`, `arch`, `duration_seconds` (the command's duration) and `distinct_id`, a random UUID kept in `~/.soup/telemetry_id` (created on opt-in, even while the key is the placeholder). It never carries model weights, dataset contents, file paths, configurations or secrets. See the [Privacy Policy](docs/backends-and-ops.md#privacy-policy) for details.

**There is no bug bounty and no monetary reward.** We credit reporters by name
in the release notes, which is the whole of what we offer. Reports that open
with a request for payment get this paragraph as the reply.

## Disclosure

We practice coordinated disclosure. Once a fix is released we credit the
reporter in the release notes, unless anonymity is requested.

> A detailed, per-version log of historical security hardening previously lived
> in this file. It now lives in the project's git history and in the
> [GitHub Releases](https://github.com/MakazhanAlpamys/Soup/releases) notes.
