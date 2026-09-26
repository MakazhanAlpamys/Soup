# Changelog fragments

User-visible pull requests add one Markdown fragment instead of editing the shared
`CHANGELOG.md` file:

```text
changelog.d/<latest-release>/<number>.<category>.md
```

`<latest-release>` is the newest released version heading in `CHANGELOG.md`. `<number>`
should be the pull-request number rather than the issue it closes; this keeps two pull
requests for the same issue distinct and traceable. The validator rejects a second
category with the same number. `<category>` is one of `added`, `changed`, `deprecated`,
`removed`, `fixed`, or `security`.

Write the complete changelog list item in the fragment and mention the same `#<number>` in
its text. Long entries may contain paragraphs, tables, and fenced code blocks; assembly
copies their Markdown verbatim. For example:

```text
changelog.d/0.73.3/490.changed.md
```

Before a release, run:

```bash
python scripts/assemble_changelog.py
```

The command validates every file, inserts entries under `[Unreleased]`, and consumes the
fragments. A fragment based on an older release fails validation after a release, forcing
its pull request to rebase and move the file. Tag publication independently refuses to
continue while any fragment remains.
