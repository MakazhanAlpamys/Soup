# CycloneDX 1.6 schemas (test fixtures)

Vendored unmodified from the CycloneDX specification repository
(https://github.com/CycloneDX/specification) at tag `1.6`, `schema/` directory, so the tests can
validate the documents `soup bom` writes without network access:

| file | upstream file | what it is |
|---|---|---|
| `bom-1.6.schema.json` | `schema/bom-1.6.schema.json` | the CycloneDX 1.6 BOM schema |
| `spdx.schema.json` | `schema/spdx.schema.json` | the SPDX licence id enum the 1.6 schema references (`$comment: v1.0-3.23`) |
| `jsf-0.82.schema.json` | `schema/jsf-0.82.schema.json` | the JSON Signature Format schema the 1.6 `signature` property references |

The CycloneDX specification repository is licensed under the Apache License 2.0; the two schemas
that carry a licence `$comment` upstream keep it here verbatim. `src/soup_cli/utils/spdx_license_ids.py`
is generated from `spdx.schema.json` (a test keeps the two in step). Where notices for vendored
material live repository-wide is being settled in #1677; this file records the provenance of these
three until then.
