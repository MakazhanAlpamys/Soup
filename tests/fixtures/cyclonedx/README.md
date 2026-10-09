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
that carry a licence `$comment` upstream keep it here verbatim. The upstream repository has no
`NOTICE` file at tag `1.6`, so there is nothing to carry over into this project's `NOTICE`.
`src/soup_cli/utils/spdx_license_ids.py` is generated from `spdx.schema.json` (a test keeps the
two in step).

Third-party notices for vendored files live here, next to the files, and not in `NOTICE` or a
separate notices file (#1677). The files are byte-identical to upstream; the sha256 of each, as
committed (LF line endings, the same as upstream), lets that be checked without the network:

| file | sha256 |
|---|---|
| `bom-1.6.schema.json` | `3e92dddbc30cf7f6a02b80f0942b1a4cfd4fb1c26f1dfc4310afa9d613cafb93` |
| `jsf-0.82.schema.json` | `8bae002c25e723db7ee1f26afde680ae1a2b1a8f6b4b4b0fd65dc3becb090aae` |
| `spdx.schema.json` | `baa9d3bd1ed57b6751b0887edead6b5063ff53ff7429cf85d476c6c94af0166e` |

Check one with `git show HEAD:tests/fixtures/cyclonedx/bom-1.6.schema.json | sha256sum`; a
Windows checkout with `core.autocrlf=true` rewrites the working-tree copy to CRLF, which hashes
differently.
