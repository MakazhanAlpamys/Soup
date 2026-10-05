"""CycloneDX ML-BOM + SPDX 2.3 AI-profile emitter (v0.59.0 Part A).

Pure-stdlib, no heavy imports — reads from a small ``BomEntry`` frozen
dataclass that the caller derives from a v0.26.0 ``RegistryEntry``. Two
output formats:

- **CycloneDX 1.6 + ML-BOM extension** (``bomFormat: CycloneDX``,
  ``specVersion: 1.6``, root component ``type=machine-learning-model``).
- **SPDX 2.3 + AI profile** (``spdxVersion: SPDX-2.3``, ``dataLicense:
  CC0-1.0``, package with ``primaryPackagePurpose: AI-MODEL``).

Atomic write via ``tempfile.mkstemp + os.replace`` under cwd containment
+ ``os.lstat + S_ISLNK`` rejection (TOCTOU defence — mirrors v0.33.0 #22
/ v0.43.0 Part C / v0.46.0 Part A / v0.56.0 / v0.57.0 / v0.58.0 policy).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import uuid
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Mapping, Optional, Tuple

from soup_cli.utils.paths import atomic_write_text
from soup_cli.utils.spdx_license_ids import canonical_spdx_id

if TYPE_CHECKING:
    from soup_cli.utils.energy import EnergyMeasurement

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_MAX_NAME = 256
_MAX_VERSION = 64
_MAX_LICENSE = 64
_MAX_TASK = 64
_VALID_FORMATS = ("cyclonedx", "spdx", "both")


def _check_str(
    value: object, *, field_name: str, max_len: int, allow_none: bool = False,
) -> Optional[str]:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} must not be None")
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be str, got {type(value).__name__}")
    if "\x00" in value:
        raise ValueError(f"{field_name} must not contain null bytes")
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    if len(value) > max_len:
        raise ValueError(f"{field_name} length {len(value)} exceeds {max_len}")
    return value


def _check_sha256(value: object, *, field_name: str, allow_none: bool = False) -> Optional[str]:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} must not be None")
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be str, got {type(value).__name__}")
    if not _SHA256_RE.match(value):
        raise ValueError(f"{field_name} must be 64 hex chars (got {len(value)})")
    return value


@dataclass(frozen=True)
class BomEntry:
    """Per-run BOM input.

    The shape mirrors what we can read from a v0.26.0 ``RegistryEntry``:
    ``base_model`` + ``base_sha`` + ``config_sha`` + ``data_sha`` + ``task`` +
    ``parents`` (tuple of parent entry ids for SLSA materials) + ``artifacts``
    (tuple of dicts with ``kind`` + ``sha256`` + ``size_bytes``).
    """

    name: str
    version: str
    base_model: str
    base_sha: str
    config_sha: str
    data_sha: Optional[str]
    task: str
    license: Optional[str]
    parents: Tuple[str, ...]
    artifacts: Tuple[Mapping[str, Any], ...]
    created_at: str
    # Optional energy attachments (Part F) — see attach_energy().
    energy_kwh: Optional[float] = None
    co2_kg: Optional[float] = None
    pue: Optional[float] = None
    grid_intensity_g_per_kwh: Optional[float] = None
    energy_source: Optional[str] = None

    def __post_init__(self) -> None:
        _check_str(self.name, field_name="name", max_len=_MAX_NAME)
        _check_str(self.version, field_name="version", max_len=_MAX_VERSION)
        _check_str(self.base_model, field_name="base_model", max_len=_MAX_NAME)
        _check_sha256(self.base_sha, field_name="base_sha")
        _check_sha256(self.config_sha, field_name="config_sha")
        _check_sha256(self.data_sha, field_name="data_sha", allow_none=True)
        _check_str(self.task, field_name="task", max_len=_MAX_TASK)
        if self.license is not None:
            _check_str(self.license, field_name="license", max_len=_MAX_LICENSE)
        _check_str(self.created_at, field_name="created_at", max_len=64)
        if not isinstance(self.parents, tuple):
            raise ValueError("parents must be a tuple")
        for p in self.parents:
            _check_str(p, field_name="parents[*]", max_len=_MAX_NAME)
        if not isinstance(self.artifacts, tuple):
            raise ValueError("artifacts must be a tuple")
        for value, name in (
            (self.energy_kwh, "energy_kwh"),
            (self.co2_kg, "co2_kg"),
            (self.pue, "pue"),
            (self.grid_intensity_g_per_kwh, "grid_intensity_g_per_kwh"),
        ):
            if value is None:
                continue
            if isinstance(value, bool):
                raise ValueError(f"{name} must not be bool")
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if float(value) < 0:
                raise ValueError(f"{name} must be >= 0")


def attach_energy(entry: BomEntry, measurement: EnergyMeasurement) -> BomEntry:
    """Return a new BomEntry with energy/CO2 fields populated from a v0.59 EnergyMeasurement.

    Caller is responsible for ensuring the measurement is finite + non-negative
    (the EnergyMeasurement dataclass already enforces this).
    """
    return replace(
        entry,
        energy_kwh=float(measurement.energy_kwh),
        co2_kg=float(measurement.co2_kg),
        pue=float(measurement.pue),
        grid_intensity_g_per_kwh=float(measurement.grid_intensity_g_per_kwh),
        energy_source=str(measurement.source),
    )


def _energy_properties(entry: BomEntry) -> list[dict]:
    props: list[dict] = []
    if entry.energy_kwh is not None:
        props.append({"name": "soup:energy_kwh", "value": str(entry.energy_kwh)})
    if entry.co2_kg is not None:
        props.append({"name": "soup:co2_kg", "value": str(entry.co2_kg)})
    if entry.pue is not None:
        props.append({"name": "soup:pue", "value": str(entry.pue)})
    if entry.grid_intensity_g_per_kwh is not None:
        props.append({
            "name": "soup:grid_intensity_g_per_kwh",
            "value": str(entry.grid_intensity_g_per_kwh),
        })
    if entry.energy_source is not None:
        props.append({"name": "soup:energy_source", "value": entry.energy_source})
    return props


def _energy_annotations(entry: BomEntry, annotation_date: str) -> list[dict]:
    """SPDX has no native properties — surface energy as OTHER annotations.

    Mirrors :func:`_energy_properties` (same ``soup:<field>=<value>`` naming) so
    energy data lands in the SPDX output as well as CycloneDX (the #244 contract
    is "both outputs").
    """
    return [
        {
            "annotator": "Tool: soup-cli",
            "annotationDate": annotation_date,
            "annotationType": "OTHER",
            "annotationComment": f"{prop['name']}={prop['value']}",
        }
        for prop in _energy_properties(entry)
    ]


_SPDX_OPERATORS = frozenset({"AND", "OR", "WITH"})
_SPDX_TOKEN_RE = re.compile(r"[()]|[^\s()]+")
# A ``DocumentRef-`` qualified ref is not accepted: it points into another SPDX document
# and this builder emits no ``externalDocumentRefs``, so the value stays a name.
_LICENSE_REF_RE = re.compile(r"LicenseRef-[A-Za-z0-9.-]+")


def _spdx_expression(value: str) -> Optional[str]:
    """``value`` as an SPDX expression with canonical ids, or ``None`` when it is not one.

    SPDX 2.3 Annex D: operators are upper-case and sit between operands, parentheses
    balance, and every operand is a listed id (optionally with ``+``) or a
    ``LicenseRef-``. So ``Gemma Terms of Use and Prohibited Use Policy`` or
    ``Apache 2.0 with Commons Clause`` is a name, not an expression (#1569 review).
    """
    tokens = _SPDX_TOKEN_RE.findall(value)
    # A lone ``LicenseRef-`` or a lone ``id+`` that is not itself a listed id
    # (``Apache-2.0+``; ``GPL-2.0+`` is listed) is a simple expression; a lone
    # listed id stays on the ``license.id`` path.
    simple = len(tokens) == 1 and (
        _LICENSE_REF_RE.fullmatch(tokens[0]) is not None
        or (tokens[0].endswith("+") and canonical_spdx_id(tokens[0]) is None)
    )
    if not simple and not any(token in _SPDX_OPERATORS for token in tokens):
        return None
    out: list[str] = []
    depth, want_operand = 0, True
    for token in tokens:
        if token == "(" and want_operand:
            depth += 1
        elif token == ")" and not want_operand and depth:
            depth -= 1
        elif token in _SPDX_OPERATORS and not want_operand:
            want_operand = True
        elif not want_operand or token in "()" or token in _SPDX_OPERATORS:
            return None
        else:
            want_operand = False
            if not _LICENSE_REF_RE.fullmatch(token):
                plus = "+" if token.endswith("+") else ""
                canonical = canonical_spdx_id(token[: len(token) - len(plus)])
                if canonical is None:
                    return None
                token = canonical + plus
        out.append(token)
    if want_operand or depth:
        return None
    return re.sub(r"\( | \)", lambda m: m.group(0).strip(), " ".join(out))


def _license_choice(value: Optional[str]) -> list[dict]:
    """CycloneDX 1.6 ``licenses``: a canonical SPDX id in ``license.id``, an SPDX
    expression (``A OR B``) in ``expression``, anything else in ``license.name`` (#1446)."""
    if not value:
        return []
    expression = _spdx_expression(value)
    if expression is not None:
        return [{"expression": expression}]
    canonical = canonical_spdx_id(value)
    if canonical is not None:
        return [{"license": {"id": canonical}}]
    return [{"license": {"name": value}}]


_LICENSE_REF_SAFE_RE = re.compile(r"[^A-Za-z0-9.-]+")


def _spdx_license_expression(value: Optional[str]) -> tuple[str, list[dict]]:
    """SPDX 2.3 ``licenseConcluded`` / ``licenseDeclared``: a canonical id, an expression
    with its ids canonicalised, or a ``LicenseRef-`` with its ``hasExtractedLicensingInfos``
    entry (#1446). Every ``LicenseRef-`` the field uses, including an operand inside an
    expression, gets an entry: SPDX requires each one to be defined in the document."""
    if not value:
        return "NOASSERTION", []
    expression = _spdx_expression(value)
    if expression is not None:
        refs = list(dict.fromkeys(_LICENSE_REF_RE.findall(expression)))
        return expression, [{"licenseId": ref, "name": ref, "extractedText": ref} for ref in refs]
    canonical = canonical_spdx_id(value)
    if canonical is not None:
        return canonical, []
    suffix = _LICENSE_REF_SAFE_RE.sub("-", value).strip("-")
    if not suffix:  # nothing idstring-safe survived: name it by content instead
        suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    ref = f"LicenseRef-{suffix}"
    return ref, [{"licenseId": ref, "name": value, "extractedText": value}]


def build_cyclonedx_bom(entry: BomEntry) -> dict:
    """Render a CycloneDX 1.6 ML-BOM dict (in-memory)."""
    if not isinstance(entry, BomEntry):
        raise TypeError(f"entry must be BomEntry, got {type(entry).__name__}")
    licenses = _license_choice(entry.license)

    components: list[dict] = [
        {
            "type": "machine-learning-model",
            "name": entry.base_model,
            "bom-ref": f"base:{entry.base_sha}",
            "hashes": [{"alg": "SHA-256", "content": entry.base_sha}],
            "mime-type": "application/x-machine-learning-model",
        }
    ]
    for parent in entry.parents:
        components.append({
            "type": "machine-learning-model",
            "name": parent,
            "bom-ref": f"parent:{parent}",
        })
    for index, art in enumerate(entry.artifacts):
        kind = str(art.get("kind", "artifact"))
        digest = str(art.get("sha256", "")).lower()
        raw_size = art.get("size_bytes", 0)
        if isinstance(raw_size, bool):
            raise ValueError(f"artifact[{index}].size_bytes must not be bool")
        try:
            size = int(raw_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"artifact[{index}].size_bytes must be int-like: {exc}"
            ) from exc
        comp = {
            "type": "file",
            "name": f"{entry.name}/{kind}",
            "bom-ref": f"artifact:{kind}:{digest[:12]}",
        }
        if _SHA256_RE.match(digest):
            comp["hashes"] = [{"alg": "SHA-256", "content": digest}]
        if size > 0:
            comp["properties"] = [{"name": "size_bytes", "value": str(size)}]
        components.append(comp)

    properties = [
        {"name": "soup:task", "value": entry.task},
        {"name": "soup:config_sha256", "value": entry.config_sha},
    ]
    if entry.data_sha:
        properties.append({"name": "soup:data_sha256", "value": entry.data_sha})
    properties.extend(_energy_properties(entry))

    doc = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": uuid.uuid4().urn,  # #1446: RFC 4122, as the 1.6 schema requires
        "version": 1,
        "metadata": {
            "timestamp": entry.created_at,
            "tools": [{"name": "soup-cli", "version": _read_soup_version()}],
            "component": {
                "type": "machine-learning-model",
                "name": entry.name,
                "version": entry.version,
                "bom-ref": f"soup:{entry.name}@{entry.version}",
                **({"licenses": licenses} if licenses else {}),
                "properties": [
                    {"name": "soup:base_model", "value": entry.base_model},
                    {"name": "soup:task", "value": entry.task},
                ],
            },
            "properties": properties,
        },
        "components": components,
    }
    return doc


def build_spdx_bom(entry: BomEntry) -> dict:
    """Render an SPDX 2.3 + AI-profile dict (in-memory)."""
    if not isinstance(entry, BomEntry):
        raise TypeError(f"entry must be BomEntry, got {type(entry).__name__}")
    spdx_id_main = "SPDXRef-Model"
    license_expression, extracted_licenses = _spdx_license_expression(entry.license)
    pkg = {
        "SPDXID": spdx_id_main,
        "name": entry.name,
        "versionInfo": entry.version,
        "downloadLocation": "NOASSERTION",
        "filesAnalyzed": False,
        "licenseConcluded": license_expression,
        "licenseDeclared": license_expression,
        "copyrightText": "NOASSERTION",
        "primaryPackagePurpose": "AI-MODEL",
        "annotations": [
            {
                "annotator": "Tool: soup-cli",
                "annotationDate": entry.created_at,
                "annotationType": "OTHER",
                "annotationComment": f"task={entry.task} base={entry.base_model}",
            },
            *_energy_annotations(entry, entry.created_at),
        ],
        "checksums": [{"algorithm": "SHA256", "checksumValue": entry.config_sha}],
    }
    pkg_base = {
        "SPDXID": "SPDXRef-Base",
        "name": entry.base_model,
        "downloadLocation": "NOASSERTION",
        "filesAnalyzed": False,
        "licenseConcluded": "NOASSERTION",
        "licenseDeclared": "NOASSERTION",
        "copyrightText": "NOASSERTION",
        "primaryPackagePurpose": "AI-MODEL",
        "checksums": [{"algorithm": "SHA256", "checksumValue": entry.base_sha}],
    }
    relationships = [
        {
            "spdxElementId": spdx_id_main,
            "relatedSpdxElement": "SPDXRef-Base",
            "relationshipType": "DERIVED_FROM",
        }
    ]

    doc = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": entry.name,
        "documentNamespace": f"https://soup.local/bom/{entry.name}-{secrets.token_hex(8)}",
        "creationInfo": {
            "created": entry.created_at,
            "creators": [f"Tool: soup-cli-{_read_soup_version()}"],
        },
        "packages": [pkg, pkg_base],
        "relationships": relationships,
    }
    if extracted_licenses:
        doc["hasExtractedLicensingInfos"] = extracted_licenses
    if entry.data_sha:
        doc["packages"].append({
            "SPDXID": "SPDXRef-Data",
            "name": "training-data",
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
            "copyrightText": "NOASSERTION",
            "primaryPackagePurpose": "SOURCE",
            "checksums": [{"algorithm": "SHA256", "checksumValue": entry.data_sha}],
        })
        # #1446: SPDX 2.3 reads `A BUILD_DEPENDENCY_OF B` as "A is a build dependency
        # of B": the training data is the dependency, the model the dependent.
        relationships.append({
            "spdxElementId": "SPDXRef-Data",
            "relatedSpdxElement": spdx_id_main,
            "relationshipType": "BUILD_DEPENDENCY_OF",
        })
    return doc


def render_bom(entry: BomEntry, fmt: str) -> str:
    """Return JSON-serialised BOM (CycloneDX or SPDX) for the given entry."""
    if not isinstance(fmt, str):
        raise ValueError("fmt must be str")
    fmt_lc = fmt.lower()
    if fmt_lc == "cyclonedx":
        return json.dumps(build_cyclonedx_bom(entry), indent=2, sort_keys=True)
    if fmt_lc == "spdx":
        return json.dumps(build_spdx_bom(entry), indent=2, sort_keys=True)
    raise ValueError(f"Unsupported BOM format: {fmt!r} (use one of {_VALID_FORMATS})")


def write_bom(entry: BomEntry, fmt: str, output_path: str) -> str:
    """Atomically write a BOM to ``output_path`` (must stay under cwd)."""
    text = render_bom(entry, fmt)
    return atomic_write_text(text, output_path, prefix=".bom.", suffix=".json.tmp")


def _read_soup_version() -> str:
    try:
        from soup_cli import __version__
        return __version__
    except ImportError:
        return "unknown"
