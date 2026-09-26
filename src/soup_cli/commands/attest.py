"""soup attest — in-toto + SLSA-3 attestation CLI (v0.59.0 Part B; ed25519 v0.71.2)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape

from soup_cli.utils.attest import (
    AttestationStatement,
    render_attestation,
    sign_attestation,
    verify_attestation,
    verify_sigstore_attestation,
    write_attestation,
)
from soup_cli.utils.terminal import for_terminal

console = Console()

_SIGNATURE_SUFFIX = ".sig"
_MAX_SIGNATURE_SIDECAR_BYTES = 16 * 1024 * 1024
_EXIT_ATTACH_FAILED = 1
_EXIT_USAGE = 2

app = typer.Typer(
    no_args_is_help=True,
    help="In-toto + SLSA-3 attestations per Soup Can stage (v0.59.0).",
)


@app.command("emit")
def emit_cmd(
    stage: str = typer.Option(
        ..., "--stage",
        help="Stage: extract / train / eval / export / publish.",
    ),
    subject_name: str = typer.Option(..., "--subject", help="Artefact name."),
    subject_sha: str = typer.Option(..., "--sha", help="64-hex SHA-256 of the artefact."),
    builder_id: str = typer.Option(
        "soup-cli", "--builder",
        help="Builder identity (default: soup-cli).",
    ),
    invocation: Optional[str] = typer.Option(
        None, "--invocation",
        help="Free-form invocation marker (e.g. command line).",
    ),
    sign_backend: str = typer.Option(
        "unsigned", "--sign",
        help="Signature backend: unsigned (default) | ed25519 | sigstore "
             "(keyless OIDC + Fulcio/Rekor).",
    ),
    key: Optional[str] = typer.Option(
        None, "--key",
        help="ed25519 private-key PEM path (or set SOUP_SIGNING_KEY).",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path (cwd-contained).",
    ),
    attach_to_registry: Optional[str] = typer.Option(
        None, "--attach-to-registry",
        help="Attach the emitted attestation statement to a registry entry id (needs --output).",
    ),
    interactive_oidc: bool = typer.Option(
        False, "--interactive-oidc",
        help="Allow Sigstore browser OIDC when no ambient credential exists. "
             "Off by default so headless runners fail instead of hanging.",
    ),
) -> None:
    """Emit a per-stage in-toto/SLSA-3 attestation.

    With ``--sign ed25519 --key <priv.pem>`` the rendered Statement is
    signed with a detached ed25519 signature. ``--sign sigstore`` uses
    keyless OIDC + Fulcio/Rekor. With ``--output``, both backends write the
    portable signature material to ``<output>.sig``; verify it with
    ``soup attest verify``.
    """
    try:
        st = AttestationStatement(
            stage=stage,
            subject_name=subject_name,
            subject_sha256=subject_sha,
            builder_id=builder_id,
            invocation={"command": invocation or ""},
            materials=(),
            created_at=datetime.now(tz=timezone.utc).isoformat(),
        )
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Invalid attestation: {escape(str(exc))}[/]")
        raise typer.Exit(2)

    # Validate every local argument/path before a Sigstore request can create
    # a permanent Rekor entry carrying the signer's identity.
    normalized_backend = sign_backend.lower() if isinstance(sign_backend, str) else ""
    if attach_to_registry is not None and output is None:
        console.print(
            "[red]--attach-to-registry needs --output "
            "(nothing written to attach).[/]"
        )
        raise typer.Exit(_EXIT_USAGE)
    if normalized_backend == "sigstore" and output is None:
        console.print(
            "[red]--sign sigstore requires --output so the Sigstore bundle is "
            "persisted and can be verified later.[/]"
        )
        raise typer.Exit(_EXIT_USAGE)
    if output is not None:
        from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

        try:
            enforce_under_cwd_and_no_symlink(output, "--output")
            if os.path.isdir(output):
                raise ValueError("--output must name a file, not a directory")
            if normalized_backend in {"ed25519", "sigstore"}:
                sidecar_path = output + _SIGNATURE_SUFFIX
                enforce_under_cwd_and_no_symlink(sidecar_path, "signature sidecar")
                if os.path.isdir(sidecar_path):
                    raise ValueError(
                        "signature sidecar path must name a file, not a directory"
                    )
        except (TypeError, ValueError) as exc:
            console.print(f"[red]Invalid output: {for_terminal(exc)}[/]")
            raise typer.Exit(_EXIT_USAGE) from exc

    text = render_attestation(st)

    try:
        sig = sign_attestation(
            text.encode("utf-8"),
            backend=sign_backend,
            key_path=key,
            sigstore_interactive=interactive_oidc,
        )
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Sign failed: {for_terminal(exc)}[/]")
        raise typer.Exit(2)
    except RuntimeError as exc:
        # Explicit signing requests must never degrade to unsigned.
        console.print(f"[red]Sign failed: {for_terminal(exc)}[/]")
        raise typer.Exit(1)

    if output is None:
        console.print(text)
        console.print(f"[dim]signature backend: {escape(sig['backend'])}[/]")
        if sig.get("signature"):
            console.print(f"[dim]signature: {escape(sig['signature'][:32])}...[/]")
        return

    try:
        written = write_attestation(st, output)
        written_paths = [written]
        if sig.get("backend") in {"ed25519", "sigstore"} and (
            sig.get("signature") or sig.get("sigstore_bundle")
        ):
            written_paths.append(_write_sig_sidecar(output, sig))
    except (TypeError, ValueError, OSError) as exc:
        console.print(f"[red]Write failed: {for_terminal(exc)}[/]")
        raise typer.Exit(2)
    console.print(
        f"[green]Wrote attestation[/] -> {escape(written)} "
        f"[dim](signature: {escape(sig['backend'])})[/]"
    )
    if attach_to_registry is not None:
        _attach_attestation(attach_to_registry, written_paths)


def _attach_attestation(registry_id: str, paths: list[str]) -> None:
    """Attach an emitted attestation statement and signature to a registry entry.

    A requested registry attachment is part of command success: lookup or
    attachment failures exit non-zero after leaving the statement on disk.
    """
    try:
        from soup_cli.registry.attach import attach_artifact
    except ImportError as exc:
        console.print(
            f"[red]Error:[/] could not import registry attach helper: {escape(str(exc))}"
        )
        raise typer.Exit(_EXIT_ATTACH_FAILED) from exc
    for path in paths:
        try:
            attach_artifact(registry_id, path=path, kind="attestation")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Error:[/] could not attach to registry: {escape(str(exc))}")
            raise typer.Exit(_EXIT_ATTACH_FAILED) from exc
        else:
            console.print(
                f"[green]Attached[/] attestation to registry entry "
                f"[bold]{escape(registry_id)}[/]"
            )


def _write_sig_sidecar(output: str, sig: dict) -> str:
    """Atomic write of the ``<output>.sig`` JSON sidecar (cwd-contained)."""
    from soup_cli.utils.paths import atomic_write_text

    payload = {
        "backend": sig.get("backend", ""),
        "signature": sig.get("signature", ""),
        "public_key": sig.get("public_key", ""),
    }
    if sig.get("backend") == "sigstore":
        bundle = sig.get("sigstore_bundle", "")
        if not isinstance(bundle, str) or not bundle.strip():
            raise ValueError("Sigstore sidecar requires a non-empty bundle")
        payload["sigstore_bundle"] = bundle
    body = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    )
    return atomic_write_text(
        body,
        output + _SIGNATURE_SUFFIX,
        prefix=".attest-sig.",
        suffix=".json.tmp",
    )


@app.command("verify")
def verify_cmd(
    statement: str = typer.Argument(..., help="Path to the in-toto Statement JSON."),
    signature: str = typer.Option(
        ..., "--signature", "-s",
        help="Path to the .sig JSON sidecar written by `attest emit --sign`.",
    ),
    public_key: Optional[str] = typer.Option(
        None, "--public-key",
        help="Trusted ed25519 public-key PEM. When set, the embedded key must "
             "match it (genuine authentication).",
    ),
    cert_identity: Optional[str] = typer.Option(
        None, "--cert-identity",
        help="Trusted Sigstore certificate identity (SAN). Requires "
             "--cert-oidc-issuer and is required for sigstore-signed attestations.",
    ),
    cert_oidc_issuer: Optional[str] = typer.Option(
        None, "--cert-oidc-issuer",
        help="Trusted OIDC issuer. Required together with --cert-identity.",
    ),
) -> None:
    """Verify an ed25519 or Sigstore-signed attestation.

    Exit codes: 0 = signature valid; 1 = verifier unavailable; 2 = usage/input
    error; 3 = invalid signature / policy mismatch.
    """
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    try:
        enforce_under_cwd_and_no_symlink(statement, "statement")
        enforce_under_cwd_and_no_symlink(signature, "signature")
        if os.lstat(signature).st_size > _MAX_SIGNATURE_SIDECAR_BYTES:
            raise ValueError(
                f"signature exceeds {_MAX_SIGNATURE_SIDECAR_BYTES} bytes"
            )
    except (OSError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2)

    try:
        with open(statement, encoding="utf-8") as fh:
            raw_text = fh.read()
        with open(signature, encoding="utf-8") as fh:
            sig_doc = json.load(fh)
    except (OSError, ValueError) as exc:
        console.print(f"[red]Could not read inputs: {escape(str(exc))}[/]")
        raise typer.Exit(2)

    # `emit` signs the canonical in-toto JSON (json.dumps sort_keys, indent=2).
    # Re-canonicalise the on-disk statement so verification is independent of
    # platform newline translation (Windows CRLF) and incidental whitespace.
    try:
        payload = json.dumps(
            json.loads(raw_text), indent=2, sort_keys=True
        ).encode("utf-8")
    except (ValueError, TypeError):
        console.print("[red]Statement is not valid JSON[/]")
        raise typer.Exit(3)

    if not isinstance(sig_doc, dict):
        console.print("[red]Signature sidecar must be a JSON object[/]")
        raise typer.Exit(2)
    backend = str(sig_doc.get("backend", ""))
    sig_hex = str(sig_doc.get("signature", ""))
    pub = str(sig_doc.get("public_key", ""))
    sigstore_bundle = str(sig_doc.get("sigstore_bundle", ""))

    if cert_identity is not None and not cert_oidc_issuer:
        console.print("[red]--cert-identity requires --cert-oidc-issuer[/]")
        raise typer.Exit(2)
    if cert_oidc_issuer is not None and not cert_identity:
        console.print("[red]--cert-oidc-issuer requires --cert-identity[/]")
        raise typer.Exit(2)

    if backend == "sigstore":
        if public_key is not None:
            console.print("[red]--public-key applies only to ed25519 signatures[/]")
            raise typer.Exit(2)
        if not sigstore_bundle:
            console.print("[red]Sigstore sidecar has no bundle — cannot verify[/]")
            raise typer.Exit(3)
        if not cert_identity or not cert_oidc_issuer:
            console.print(
                "[red]Sigstore verification requires trusted --cert-identity and "
                "--cert-oidc-issuer values supplied out of band[/]"
            )
            raise typer.Exit(3)
        try:
            verify_sigstore_attestation(
                payload,
                sigstore_bundle,
                identity=cert_identity,
                issuer=cert_oidc_issuer,
            )
        except RuntimeError as exc:
            console.print(
                f"[red]Sigstore verification unavailable: {for_terminal(exc)}[/]"
            )
            raise typer.Exit(1)
        except ValueError as exc:
            console.print(
                f"[red]Sigstore attestation INVALID: {for_terminal(exc)}[/]"
            )
            raise typer.Exit(3)
        console.print(
            f"[green]Attestation Sigstore signature valid[/] "
            f"[dim]({escape(os.path.basename(statement))})[/]"
        )
        console.print(
            "[dim]Note: the subject digest is asserted by the signer, not "
            "re-verified against an artifact.[/]"
        )
        return

    if cert_identity is not None:
        console.print("[red]--cert-identity applies only to Sigstore signatures[/]")
        raise typer.Exit(2)

    if backend != "ed25519" or not sig_hex:
        console.print(
            f"[yellow]Signature backend {escape(backend or 'unsigned')!r} "
            "is not cryptographically verifiable.[/]"
        )
        raise typer.Exit(3)

    # Explicit fail-closed guard: an ed25519 sidecar with no public key (and no
    # --public-key supplied out of band) cannot be verified (code-review H1).
    if not pub and public_key is None:
        console.print(
            "[red]Signature sidecar has no public key and no --public-key was "
            "supplied — cannot verify.[/]"
        )
        raise typer.Exit(3)

    if public_key is not None:
        from soup_cli.utils.signing import read_public_key_file

        try:
            trusted = read_public_key_file(public_key)
        except (OSError, ValueError) as exc:
            console.print(f"[red]Could not read --public-key: {escape(str(exc))}[/]")
            raise typer.Exit(2)
        if "".join(trusted.split()) != "".join(pub.split()):
            console.print(
                "[red]Signed by an untrusted key (embedded key does not match "
                "--public-key).[/]"
            )
            raise typer.Exit(3)
        pub = trusted

    if verify_attestation(payload, sig_hex, pub):
        console.print(
            f"[green]Attestation signature valid[/] "
            f"[dim]({escape(os.path.basename(statement))})[/]"
        )
        # Make the trust boundary explicit: a valid signature proves the
        # signer asserted this statement — it does NOT re-verify the subject
        # digest against any on-disk artifact.
        console.print(
            "[dim]Note: the subject digest is asserted by the signer, not "
            "re-verified against an artifact.[/]"
        )
        return
    console.print("[red]Attestation signature INVALID — tampered or wrong key.[/]")
    raise typer.Exit(3)
