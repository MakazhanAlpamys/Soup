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
    write_attestation,
)

console = Console()

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
             "(OIDC-via-GitHub + Fulcio/Rekor).",
    ),
    key: Optional[str] = typer.Option(
        None, "--key",
        help="ed25519 private-key PEM path (or set SOUP_SIGNING_KEY).",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path (cwd-contained).",
    ),
) -> None:
    """Emit a per-stage in-toto/SLSA-3 attestation.

    With ``--sign ed25519 --key <priv.pem>`` (v0.71.2 #179) the rendered
    Statement is signed with a real detached ed25519 signature; when
    ``--output`` is set a ``<output>.sig`` JSON sidecar (backend / signature /
    public_key) is written next to it. Verify with ``soup attest verify``.
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

    text = render_attestation(st)

    try:
        sig = sign_attestation(text.encode("utf-8"), backend=sign_backend, key_path=key)
    except ImportError as exc:
        console.print(f"[red]Signing unavailable: {escape(str(exc))}[/]")
        raise typer.Exit(1)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Sign failed: {escape(str(exc))}[/]")
        raise typer.Exit(2)

    if output is None:
        console.print(text)
        console.print(f"[dim]signature backend: {escape(sig['backend'])}[/]")
        if sig.get("signature"):
            console.print(f"[dim]signature: {escape(sig['signature'][:32])}...[/]")
        return

    try:
        written = write_attestation(st, output)
        if sig.get("backend") in ("ed25519", "sigstore") and sig.get("signature"):
            _write_sig_sidecar(output, sig)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Write failed: {escape(str(exc))}[/]")
        raise typer.Exit(2)
    console.print(
        f"[green]Wrote attestation[/] -> {escape(written)} "
        f"[dim](signature: {escape(sig['backend'])})[/]"
    )


def _write_sig_sidecar(output: str, sig: dict) -> None:
    """Atomic write of the ``<output>.sig`` JSON sidecar (cwd-contained)."""
    from soup_cli.utils.paths import atomic_write_text

    body = json.dumps(
        {
            "backend": sig.get("backend", ""),
            "signature": sig.get("signature", ""),
            "public_key": sig.get("public_key", ""),
            "certificate": sig.get("certificate", ""),
            "bundle": sig.get("bundle", ""),
            "identity_token_issuer": sig.get("identity_token_issuer", ""),
        },
        indent=2,
        sort_keys=True,
    )
    atomic_write_text(body, output + ".sig", prefix=".attest-sig.", suffix=".json.tmp")


@app.command("verify")
def verify_cmd(
    file: str = typer.Argument(
        ...,
        help="Path to the attestation JSON or its .sig sidecar. "
             "The companion file is resolved automatically "
             "(e.g. ``foo.json`` ↔ ``foo.json.sig``).",
    ),
    public_key: Optional[str] = typer.Option(
        None, "--public-key",
        help="Trusted ed25519 public-key PEM. When set, the embedded key must "
             "match it (genuine authentication).",
    ),
) -> None:
    """Verify an ed25519- or sigstore-signed attestation.

    Accepts a single file argument — the attestation JSON or its ``.sig``
    sidecar — and automatically resolves the companion file.

    Exit codes: 0 = signature valid; 1 = mismatch / failure.
    """
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    # Automatic .sig sidecar resolution.
    if file.endswith(".sig"):
        statement_path = file[:-4]
        sig_path = file
    else:
        statement_path = file
        sig_path = file + ".sig"

    try:
        enforce_under_cwd_and_no_symlink(statement_path, "statement")
        enforce_under_cwd_and_no_symlink(sig_path, "signature")
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1)

    try:
        with open(statement_path, encoding="utf-8") as fh:
            raw_text = fh.read()
        with open(sig_path, encoding="utf-8") as fh:
            sig_doc = json.load(fh)
    except (OSError, ValueError) as exc:
        console.print(f"[red]Could not read inputs: {escape(str(exc))}[/]")
        raise typer.Exit(1)

    # `emit` signs the canonical in-toto JSON (json.dumps sort_keys, indent=2).
    # Re-canonicalise the on-disk statement so verification is independent of
    # platform newline translation (Windows CRLF) and incidental whitespace.
    try:
        payload = json.dumps(
            json.loads(raw_text), indent=2, sort_keys=True
        ).encode("utf-8")
    except (ValueError, TypeError):
        console.print("[red]Statement is not valid JSON[/]")
        raise typer.Exit(1)

    if not isinstance(sig_doc, dict):
        console.print("[red]Signature sidecar must be a JSON object[/]")
        raise typer.Exit(1)
    backend = str(sig_doc.get("backend", ""))
    sig_hex = str(sig_doc.get("signature", ""))
    pub = str(sig_doc.get("public_key", ""))

    if backend == "sigstore":
        cert_pem = str(sig_doc.get("certificate", ""))
        bundle_str = str(sig_doc.get("bundle", ""))
        _identity_issuer = str(sig_doc.get("identity_token_issuer", ""))

        if not cert_pem or not bundle_str:
            console.print(
                "[red]Sigstore signature sidecar is missing certificate or "
                "bundle data — cannot verify.[/]"
            )
            raise typer.Exit(1)

        try:
            import sigstore.verify
            from sigstore.oidc import Issuer as OidcIssuer
        except ImportError as exc:
            console.print(
                f"[red]Sigstore verification unavailable: {escape(str(exc))}[/]"
            )
            raise typer.Exit(1)

        try:
            issuer = OidcIssuer.prod()
            identity = issuer.identity()
        except (AttributeError, ValueError) as exc:
            console.print(
                f"[red]Sigstore verification unavailable: {escape(str(exc))}[/]"
            )
            raise typer.Exit(1)

        verifier = sigstore.verify.Verifier.production()
        with verifier:
            try:
                sigstore.verify.verify(
                    raw_text,
                    identity=identity,
                    certificate=cert_pem.encode("utf-8"),
                    bundle=bundle_str.encode("utf-8"),
                )
            except Exception as exc:
                console.print(
                    f"[red]Sigstore verification failed: {escape(str(exc))}[/]"
                )
                raise typer.Exit(1)

        console.print(
            f"[green]Attestation signature valid[/] "
            f"[dim]({escape(os.path.basename(statement_path))})[/]"
        )
        console.print(
            "[dim]Note: the subject digest is asserted by the signer, not "
            "re-verified against an artifact.[/]"
        )
        raise typer.Exit(0)

    if backend != "ed25519" or not sig_hex:
        console.print(
            f"[yellow]Signature backend {escape(backend or 'unsigned')!r} "
            "is not ed25519 — nothing to cryptographically verify.[/]"
        )
        raise typer.Exit(1)

    # Explicit fail-closed guard: an ed25519 sidecar with no public key (and no
    # --public-key supplied out of band) cannot be verified (code-review H1).
    if not pub and public_key is None:
        console.print(
            "[red]Signature sidecar has no public key and no --public-key was "
            "supplied — cannot verify.[/]"
        )
        raise typer.Exit(1)

    if public_key is not None:
        from soup_cli.utils.signing import read_public_key_file

        try:
            trusted = read_public_key_file(public_key)
        except (OSError, ValueError) as exc:
            console.print(f"[red]Could not read --public-key: {escape(str(exc))}[/]")
            raise typer.Exit(1)
        if "".join(trusted.split()) != "".join(pub.split()):
            console.print(
                "[red]Signed by an untrusted key (embedded key does not match "
                "--public-key).[/]"
            )
            raise typer.Exit(1)
        pub = trusted

    if verify_attestation(payload, sig_hex, pub):
        console.print(
            f"[green]Attestation signature valid[/] "
            f"[dim]({escape(os.path.basename(statement_path))})[/]"
        )
        # Make the trust boundary explicit: a valid signature proves the
        # signer asserted this statement — it does NOT re-verify the subject
        # digest against any on-disk artifact.
        console.print(
            "[dim]Note: the subject digest is asserted by the signer, not "
            "re-verified against an artifact.[/]"
        )
        raise typer.Exit(0)
    console.print("[red]Attestation signature INVALID — tampered or wrong key.[/]")
    raise typer.Exit(1)
