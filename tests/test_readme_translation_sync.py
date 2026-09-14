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
   valid one, and a re-stamp without a re-translation passes the stamp (#852
   review). So every ``## `` section is compared with README.md's section at the
   same position, on facts translation does not change, in both directions
   (nothing dropped, nothing invented):

   - fenced code blocks: the same languages in the same order, and the same
     contents once ``#`` comments are dropped -- comments are the only part of a
     code block a translation may change;
   - external URLs, and relative link targets (``docs/models.md#optional-extras``);
   - inline-code spans;
   - numbers in prose. Measured figures keep README.md's digits: ``119.6`` stays
     ``119.6``, never ``119,6`` or ``١١٩٫٦`` -- a reader in a mixed locale takes
     ``48.241 MiB`` for roughly 48 (#852 review). Numbers written as words in
     README.md stay words.

   In-page anchors are translated with their headings, so instead of being
   compared they must resolve to a heading in the same file. Prose is not policed.
4. **The banner links exactly the READMEs that exist.** The language banner above
   the logo in every README links every other README and nothing else, so a new
   language cannot land unlinked and no banner can point at a missing file.

Files are found with ``glob("README.*.md")``, never a hand-written list: a
``README.de.md`` added without a stamp must fail, not be skipped. Every translation
also names who maintains it in ``MAINTAINERS``; a red ``main`` says whom to ping,
and a language nobody maintains is removed rather than left to lie (#769).

**Maintainers:** tr — @Ercaner1988.

**After changing README.md:** update each translation, then replace its stamp with
the line the stale-stamp failure prints.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
import unicodedata
from urllib.parse import unquote

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Who re-syncs each translation when README.md changes.
MAINTAINERS = {"README.tr.md": "@Ercaner1988"}

_STAMP_RE = re.compile(r"<!--\s*synced-from:\s*README\.md\s+sha256:([0-9a-f]{64})\s*-->")
_URL_RE = re.compile(r"https?://[^\s)\"'<>\]]+")
_CODE_SPAN_RE = re.compile(r"(`+)(.+?)\1")
_README_HREF_RE = re.compile(r'href="(README(?:\.[\w-]+)?\.md)"')
_LINK_TARGET_RE = re.compile(r'\]\(([^)\s]+)\)|(?:href|src)="([^"]+)"')
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
_NOT_PROSE_RE = re.compile(r"https?://\S+|\]\([^)]*\)|(`+).+?\1|<[^>]+>")
_NUMBER_RE = re.compile(r"[0-9]+(?:[.,][0-9]+)*")
_HASH_COMMENT_LANGS = {"bash", "sh", "shell", "yaml", "yml", "toml", "python"}
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


def unmaintained(root: pathlib.Path = ROOT) -> list[str]:
    return [path.name for path in translations(root) if path.name not in MAINTAINERS]


def _ping(root: pathlib.Path = ROOT) -> str:
    names = [f"{p.name} → {MAINTAINERS.get(p.name, 'nobody')}" for p in translations(root)]
    return "\n  Maintainers: " + ", ".join(names)


def _code_body(lang: str, lines: list[str]) -> tuple[str, ...]:
    """A fence body with ``#`` comments dropped where the language has them."""
    body = []
    for line in lines:
        if lang in _HASH_COMMENT_LANGS:
            line = re.sub(r"(^|\s)#.*$", "", line)
        if line := line.rstrip():
            body.append(line)
    return tuple(body)


def _link_targets(prose: str) -> set[str]:
    """Relative link targets; URLs, in-page anchors and the banner are checked elsewhere."""
    targets = set()
    for match in _LINK_TARGET_RE.finditer(prose):
        target = (match.group(1) or match.group(2)).replace("&amp;", "&")
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        if _README_HREF_RE.fullmatch(f'href="{target}"'):
            continue
        targets.add(target)
    return targets


def _sections(text: str) -> list[dict]:
    """Split at ``## `` headings outside code fences; keep translation-invariant facts."""
    sections: list[dict] = [{"title": "(before the first heading)", "fences": [], "prose": []}]
    fence: tuple[str, list[str]] | None = None
    for line in text.splitlines():
        if line.startswith("```"):
            if fence is None:
                fence = (line[3:].strip(), [])
            else:
                sections[-1]["fences"].append((fence[0], _code_body(*fence)))
                fence = None
        elif fence is not None:
            fence[1].append(line)
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
        section["links"] = _link_targets(prose)
        section["numbers"] = set(_NUMBER_RE.findall(_NOT_PROSE_RE.sub(" ", prose)))
    return sections


def _slug(heading: str) -> str:
    """GitHub's anchor for a heading: lowercased, punctuation and symbols dropped."""
    kept = (c for c in heading.lower() if c in "-_ " or unicodedata.category(c)[0] in "LMN")
    return "".join(kept).replace(" ", "-")


def _unresolved_anchors(text: str) -> list[str]:
    """In-page anchors that name no heading of the same file."""
    slugs, prose, in_fence = set(), [], False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            if heading := _HEADING_RE.match(line):
                slugs.add(_slug(heading.group(1)))
            prose.append(line)
    anchors = re.findall(r'\]\(#([^)\s]+)\)|href="#([^"]+)"', "\n".join(prose))
    return sorted({unquote(a or b) for a, b in anchors} - slugs)


def _swapped(number: str) -> str:
    return number.translate(str.maketrans(".,", ",."))


def structure_problems(root: pathlib.Path = ROOT) -> list[str]:
    reference = _sections((root / "README.md").read_text(encoding="utf-8"))
    problems: list[str] = []
    for path in translations(root):
        text = path.read_text(encoding="utf-8")
        for anchor in _unresolved_anchors(text):
            problems.append(f"{path.name}: anchor #{anchor} does not resolve to a heading")
        got = _sections(text)
        if len(got) != len(reference):
            problems.append(
                f"{path.name}: {len(got) - 1} `## ` sections, README.md has {len(reference) - 1}"
            )
            continue
        for ref, sec in zip(reference, got):
            where = f"{path.name} § {ref['title']!r}"
            ref_langs = [lang for lang, _ in ref["fences"]]
            got_langs = [lang for lang, _ in sec["fences"]]
            if got_langs != ref_langs:
                problems.append(f"{where}: code blocks {got_langs} != {ref_langs}")
            else:
                for n, ((lang, want), (_, have)) in enumerate(zip(ref["fences"], sec["fences"])):
                    if have != want:
                        first = next(
                            (f"{h!r} != {w!r}" for h, w in zip(have, want) if h != w),
                            f"{len(have)} lines != {len(want)}",
                        )
                        problems.append(
                            f"{where}: code block {n + 1} ({lang}) differs outside comments: "
                            f"{first}"
                        )
            for kind, label in (
                ("urls", "URL"),
                ("links", "relative link"),
                ("code", "inline code"),
                ("numbers", "number"),
            ):
                missing = sorted(ref[kind] - sec[kind])
                invented = sorted(sec[kind] - ref[kind])
                hint = ""
                if kind == "numbers" and any(_swapped(m) in invented for m in missing):
                    hint = " -- keep README.md's digits (119.6, not 119,6)"
                if missing:
                    problems.append(f"{where}: {label} missing {missing}{hint}")
                if invented:
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
        + unmaintained(root)
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
            + _ping()
        )

    def test_every_translation_mirrors_readme_structure(self):
        problems = structure_problems()
        assert not problems, (
            "A translation no longer carries README.md's content. Code, URLs, links and "
            "numbers are not translated, so each must survive section by section:\n  "
            + "\n  ".join(problems)
            + _ping()
        )

    def test_the_banner_links_exactly_the_readmes_that_exist(self):
        problems = banner_problems()
        assert not problems, "Language banner out of date:\n  " + "\n  ".join(problems)

    def test_every_translation_names_a_maintainer(self):
        missing = unmaintained()
        assert not missing, (
            f"No maintainer named for {missing}. Add one to MAINTAINERS in this file -- a "
            "language nobody maintains is removed rather than left to go stale (#769)."
        )


# ---------------------------------------------------------------------------
# The ratchet must be shown able to fail, not merely observed green.
# ---------------------------------------------------------------------------

_EN = """<p align="center">🌍 <strong>English</strong> | <a href="README.tr.md">Türkçe</a></p>

<p align="center"><img src="soup.png" alt="Soup"></p>

Install with `pip install soup-cli`; the site is https://trysoup.dev. Peak 3.32 GB --
read [the guide](docs/guide.md) or skip to [Support](#support).

```bash
soup train  # start training
```

## Support

Join the [Discord](https://discord.gg/x) or read `docs/commands.md`.
"""

_TR = """<p align="center">🌍 <a href="README.md">English</a> | <strong>Türkçe</strong></p>

<p align="center"><img src="soup.png" alt="Soup"></p>

`pip install soup-cli` ile kurun; site https://trysoup.dev adresinde. Tepe 3.32 GB --
[rehberi](docs/guide.md) okuyun ya da [Destek](#destek) bölümüne geçin.

```bash
soup train  # eğitimi başlat
```

## Destek

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
        assert unmaintained(tree) == ["README.de.md"]
        assert any("README.de.md" in problem for problem in banner_problems(tree))

    def test_a_dropped_code_block_fails(self, tree):
        _edit(tree / "README.tr.md", "```bash\nsoup train  # eğitimi başlat\n```\n", "")
        assert any("code blocks" in problem for problem in structure_problems(tree))

    def test_a_changed_command_inside_a_code_block_fails(self, tree):
        """The #852 gap: a re-stamp without a re-translation used to pass this."""
        _edit(tree / "README.tr.md", "soup train  #", "soup trainx  #")
        problems = structure_problems(tree)
        assert any("code block 1 (bash) differs" in p for p in problems), problems

    def test_a_translated_code_comment_is_accepted(self, tree):
        """CONTROL. Comments are the one part of a code block a translation may change."""
        _edit(tree / "README.tr.md", "# eğitimi başlat", "# başka bir açıklama")
        assert structure_problems(tree) == []

    def test_a_changed_url_fails_in_both_directions(self, tree):
        _edit(tree / "README.tr.md", "https://discord.gg/x", "https://discord.gg/y")
        problems = structure_problems(tree)
        assert any("URL missing ['https://discord.gg/x']" in p for p in problems), problems
        assert any("URL not in README.md ['https://discord.gg/y']" in p for p in problems)

    def test_a_changed_relative_link_fails_in_both_directions(self, tree):
        _edit(tree / "README.tr.md", "(docs/guide.md)", "(docs/rehber.md)")
        problems = structure_problems(tree)
        assert any("relative link missing ['docs/guide.md']" in p for p in problems), problems
        assert any("relative link not in README.md ['docs/rehber.md']" in p for p in problems)

    def test_an_anchor_that_resolves_to_no_heading_fails(self, tree):
        """Anchors are translated with their headings, so they must resolve, not match."""
        _edit(tree / "README.tr.md", "(#destek)", "(#support)")
        assert structure_problems(tree) == [
            "README.tr.md: anchor #support does not resolve to a heading"
        ]

    def test_a_translated_command_fails_in_both_directions(self, tree):
        """The error machine translation is most likely to make."""
        _edit(tree / "README.tr.md", "`docs/commands.md`", "`docs/komutlar.md`")
        problems = structure_problems(tree)
        assert any("missing ['docs/commands.md']" in p for p in problems), problems
        assert any("not in README.md ['docs/komutlar.md']" in p for p in problems)

    def test_a_changed_number_fails(self, tree):
        """The #852 gap in prose: 119.6 -> 999.9 with a re-stamp used to pass."""
        _edit(tree / "README.tr.md", "Tepe 3.32 GB", "Tepe 9.99 GB")
        problems = structure_problems(tree)
        assert any("number missing ['3.32']" in p for p in problems), problems
        assert any("number not in README.md ['9.99']" in p for p in problems)

    def test_a_localised_decimal_separator_fails_with_a_hint(self, tree):
        _edit(tree / "README.tr.md", "Tepe 3.32 GB", "Tepe 3,32 GB")
        problems = structure_problems(tree)
        assert any("keep README.md's digits" in p for p in problems), problems

    def test_a_dropped_section_fails(self, tree):
        _edit(tree / "README.tr.md", "## Destek\n", "")
        problems = structure_problems(tree)
        assert "README.tr.md: 0 `## ` sections, README.md has 1" in problems, problems

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
        _edit(tree / "README.tr.md", "okuyun ya da", "inceleyin veya")
        assert structure_problems(tree) == []
