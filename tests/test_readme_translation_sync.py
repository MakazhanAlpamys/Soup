"""Repo-wide ratchet: every README translation must still be a translation of README.md.

Translations were accepted on one condition (#769): something must make rot loud.
Git conflicts loudly on Python but auto-merges Markdown silently, so a translation
keeps teaching commands that no longer exist and nothing fails -- the drift
``tests/test_recipe_count_is_synced.py`` already guards for recipe counts.

Four locks, each answering a different question:

1. **Stamp present.** Every ``README.<lang>.md`` carries
   ``<!-- synced-from: README.md sha256:<hex> -->``. An absent stamp fails; it is
   not a pass, or deleting the line would be the loophole.
2. **Stamp current.** The hex equals ``sha256(README.md)`` over LF-normalised
   bytes. A Windows checkout with ``core.autocrlf=true`` sees CRLF and a different
   raw hash; one stamp has to satisfy all nine CI cells, so line endings are not
   content.
3. **Structure mirrors README.md.** A stamp proves *when* a file was synced, never
   *what* it was synced from (#774 review) -- a different document can carry a
   valid one. So every ``## `` section is compared with README.md's section at the
   same position, on facts translation does not change: the sequence of
   fenced-code languages, the set of external URLs, and the set of inline-code
   spans, in both directions (nothing dropped, nothing invented). Commands and
   flags are not translated; a lost or altered one is lost content, and it is the
   error machine translation is most likely to make. Prose is not policed.
4. **The banner links exactly the READMEs that exist.** The language banner above
   the logo in every README links every other README and nothing else, so a new
   language cannot land unlinked and no banner can point at a missing file.

Files are found with ``glob("README.*.md")``, never a hand-written list: a
``README.de.md`` added without a stamp must fail, not be skipped.

**After changing README.md:** update each translation, then replace its stamp with
the line the stale-stamp failure prints.
"""

from __future__ import annotations

import hashlib
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

_STAMP_RE = re.compile(r"<!--\s*synced-from:\s*README\.md\s+sha256:([0-9a-f]{64})\s*-->")
_URL_RE = re.compile(r"https?://[^\s)\"'<>\]]+")
_CODE_SPAN_RE = re.compile(r"(`+)(.+?)\1")
_README_HREF_RE = re.compile(r'href="(README(?:\.[\w-]+)?\.md)"')
_LOGO = '<img src="soup.png"'


def translations(root: pathlib.Path = ROOT) -> list[pathlib.Path]:
    """Every ``README.<lang>.md``, enumerated so that a new language cannot hide."""
    return sorted(root.glob("README.*.md"))


def readme_sha256(root: pathlib.Path = ROOT) -> str:
    data = (root / "README.md").read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def stamp_line(root: pathlib.Path = ROOT) -> str:
    return f"<!-- synced-from: README.md sha256:{readme_sha256(root)} -->"


def stamp_of(path: pathlib.Path) -> str | None:
    found = _STAMP_RE.search(path.read_text(encoding="utf-8"))
    return found.group(1) if found else None


def missing_stamps(root: pathlib.Path = ROOT) -> list[str]:
    return [path.name for path in translations(root) if stamp_of(path) is None]


def stale_stamps(root: pathlib.Path = ROOT) -> list[str]:
    """Translations whose stamp is absent or names a different README.md."""
    expected = readme_sha256(root)
    stale = []
    for path in translations(root):
        stamp = stamp_of(path)
        if stamp != expected:
            shown = f"{stamp[:12]}…" if stamp else "no stamp"
            stale.append(f"{path.name}: {shown} != README.md {expected[:12]}…")
    return stale


def _sections(text: str) -> list[dict]:
    """Split at ``## `` headings outside code fences; keep translation-invariant facts."""
    sections: list[dict] = [{"title": "(before the first heading)", "fences": [], "prose": []}]
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            if not in_fence:
                sections[-1]["fences"].append(line[3:].strip())
            in_fence = not in_fence
        elif in_fence:
            continue
        elif line.startswith("## "):
            sections.append({"title": line[3:].strip(), "fences": [], "prose": []})
        else:
            # Blockquote markers go so a code span wrapped across two `> ` lines reads
            # as one span -- README.md has one, in the PEP 668 note.
            sections[-1]["prose"].append(re.sub(r"^\s*>\s?", "", line))
    for section in sections:
        prose = " ".join(section.pop("prose"))
        section["urls"] = {
            url.replace("&amp;", "&").rstrip(".,;:") for url in _URL_RE.findall(prose)
        }
        section["code"] = {" ".join(m.group(2).split()) for m in _CODE_SPAN_RE.finditer(prose)}
    return sections


def structure_problems(root: pathlib.Path = ROOT) -> list[str]:
    reference = _sections((root / "README.md").read_text(encoding="utf-8"))
    problems: list[str] = []
    for path in translations(root):
        got = _sections(path.read_text(encoding="utf-8"))
        if len(got) != len(reference):
            problems.append(
                f"{path.name}: {len(got) - 1} `## ` sections, README.md has {len(reference) - 1}"
            )
            continue
        for ref, sec in zip(reference, got):
            where = f"{path.name} § {ref['title']!r}"
            if sec["fences"] != ref["fences"]:
                problems.append(f"{where}: code blocks {sec['fences']} != {ref['fences']}")
            for kind, label in (("urls", "URL"), ("code", "inline code")):
                if missing := sorted(ref[kind] - sec[kind]):
                    problems.append(f"{where}: {label} missing {missing}")
                if invented := sorted(sec[kind] - ref[kind]):
                    problems.append(f"{where}: {label} not in README.md {invented}")
    return problems


def banner_problems(root: pathlib.Path = ROOT) -> list[str]:
    readmes = ["README.md", *(path.name for path in translations(root))]
    problems: list[str] = []
    for name in readmes:
        head = (root / name).read_text(encoding="utf-8").split(_LOGO, 1)[0]
        linked = set(_README_HREF_RE.findall(head))
        expected = set(readmes) - {name}
        if unlinked := sorted(expected - linked):
            problems.append(f"{name}: banner does not link {unlinked}")
        if extra := sorted(linked - expected):
            problems.append(f"{name}: banner links {extra}, which is itself or does not exist")
    return problems


def all_problems(root: pathlib.Path = ROOT) -> list[str]:
    return (
        missing_stamps(root)
        + stale_stamps(root)
        + structure_problems(root)
        + banner_problems(root)
    )


class TestReadmeTranslationsAreSynced:
    def test_every_translation_carries_a_stamp(self):
        missing = missing_stamps()
        assert not missing, (
            f"Translations without a sync stamp: {missing}. Put this line at the top of each:"
            f"\n  {stamp_line()}"
        )

    def test_every_stamp_matches_the_current_readme(self):
        stale = stale_stamps()
        assert not stale, (
            "README.md changed after these translations were synced. Update each one to the "
            "current README.md, then replace its stamp with:\n  "
            + stamp_line()
            + "\n  "
            + "\n  ".join(stale)
        )

    def test_every_translation_mirrors_readme_structure(self):
        problems = structure_problems()
        assert not problems, (
            "A translation no longer carries README.md's content. Code blocks, URLs and "
            "inline code are not translated, so each must survive section by section:\n  "
            + "\n  ".join(problems)
        )

    def test_the_banner_links_exactly_the_readmes_that_exist(self):
        problems = banner_problems()
        assert not problems, "Language banner out of date:\n  " + "\n  ".join(problems)


# ---------------------------------------------------------------------------
# The ratchet must be shown able to fail, not merely observed green.
# ---------------------------------------------------------------------------

_EN = """<p align="center">🌍 <strong>English</strong> | <a href="README.tr.md">Türkçe</a></p>

<p align="center"><img src="soup.png" alt="Soup"></p>

Install with `pip install soup-cli`; the site is https://trysoup.dev.

```bash
soup train
```

## Contact

Join the [Discord](https://discord.gg/x) or read `docs/commands.md`.
"""

_TR = """<p align="center">🌍 <a href="README.md">English</a> | <strong>Türkçe</strong></p>

<p align="center"><img src="soup.png" alt="Soup"></p>

`pip install soup-cli` ile kurun; site https://trysoup.dev adresinde.

```bash
soup train
```

## İletişim

[Discord](https://discord.gg/x)'a katılın ya da `docs/commands.md` okuyun.
"""


@pytest.fixture
def tree(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "README.md").write_text(_EN, encoding="utf-8")
    (tmp_path / "README.tr.md").write_text(f"{stamp_line(tmp_path)}\n{_TR}", encoding="utf-8")
    assert all_problems(tmp_path) == []  # CONTROL: the untouched tree is clean
    return tmp_path


def _edit(path: pathlib.Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, old
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


class TestTheRatchetCanActuallyFail:
    """Every lock is green today, which means nothing unless each is shown to go red.
    Each test starts from a clean two-language tree and breaks exactly one thing."""

    def test_a_deleted_stamp_fails_both_stamp_checks(self, tree):
        tr = tree / "README.tr.md"
        tr.write_text(tr.read_text(encoding="utf-8").split("\n", 1)[1], encoding="utf-8")
        assert missing_stamps(tree) == ["README.tr.md"]
        [stale] = stale_stamps(tree)
        assert stale.startswith("README.tr.md: no stamp")

    def test_one_byte_in_readme_fails_and_names_both_hashes(self, tree):
        before = readme_sha256(tree)
        with (tree / "README.md").open("a", encoding="utf-8") as fh:
            fh.write("x")
        [stale] = stale_stamps(tree)
        assert stale.startswith("README.tr.md: ")
        assert before[:12] in stale and readme_sha256(tree)[:12] in stale

    def test_a_well_formed_but_wrong_stamp_fails(self, tree):
        _edit(tree / "README.tr.md", readme_sha256(tree), "0" * 64)
        assert missing_stamps(tree) == []
        assert len(stale_stamps(tree)) == 1

    def test_crlf_line_endings_do_not_change_the_stamp(self, tree):
        readme = tree / "README.md"
        lf = readme.read_bytes().replace(b"\r\n", b"\n")
        readme.write_bytes(lf.replace(b"\n", b"\r\n"))
        assert hashlib.sha256(readme.read_bytes()).digest() != hashlib.sha256(lf).digest()
        assert stale_stamps(tree) == []

    def test_a_new_language_cannot_hide(self, tree):
        """The hole a hand-written file list had: README.de.md was never visited."""
        (tree / "README.de.md").write_text(_TR, encoding="utf-8")
        assert missing_stamps(tree) == ["README.de.md"]
        assert any("README.de.md" in problem for problem in banner_problems(tree))

    def test_a_dropped_code_block_fails(self, tree):
        _edit(tree / "README.tr.md", "```bash\nsoup train\n```\n", "")
        assert any("code blocks" in problem for problem in structure_problems(tree))

    def test_a_changed_url_fails_in_both_directions(self, tree):
        _edit(tree / "README.tr.md", "https://discord.gg/x", "https://discord.gg/y")
        problems = structure_problems(tree)
        assert any("URL missing ['https://discord.gg/x']" in p for p in problems), problems
        assert any("URL not in README.md ['https://discord.gg/y']" in p for p in problems)

    def test_a_translated_command_fails_in_both_directions(self, tree):
        """The error machine translation is most likely to make."""
        _edit(tree / "README.tr.md", "`docs/commands.md`", "`docs/komutlar.md`")
        problems = structure_problems(tree)
        assert any("missing ['docs/commands.md']" in p for p in problems), problems
        assert any("not in README.md ['docs/komutlar.md']" in p for p in problems)

    def test_a_dropped_section_fails(self, tree):
        _edit(tree / "README.tr.md", "## İletişim\n", "")
        assert structure_problems(tree) == ["README.tr.md: 0 `## ` sections, README.md has 1"]

    def test_a_banner_pointing_at_a_missing_file_fails(self, tree):
        _edit(
            tree / "README.md",
            '<a href="README.tr.md">Türkçe</a>',
            '<a href="README.tr.md">Türkçe</a> | <a href="README.ja.md">日本語</a>',
        )
        assert any("README.ja.md" in problem for problem in banner_problems(tree))

    def test_a_readme_without_a_banner_link_fails(self, tree):
        _edit(tree / "README.md", '<a href="README.tr.md">Türkçe</a>', "Türkçe")
        assert banner_problems(tree) == ["README.md: banner does not link ['README.tr.md']"]

    def test_rewording_prose_is_not_policed(self, tree):
        """CONTROL. Only translation-invariant facts are compared; wording is free."""
        _edit(tree / "README.tr.md", "okuyun", "inceleyin")
        assert structure_problems(tree) == []
