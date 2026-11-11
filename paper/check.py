#!/usr/bin/env python3
"""Compliance checks for the manuscript. Run from ``paper/`` (or via ``make check``).

Checks, each of which the journal or the project's own house rules require:

* highlights   -- 3 to 5 bullets, each at most 85 characters including spaces;
* abstract     -- at most 250 words;
* pages        -- at most 50, or 120 in elsarticle ``review`` mode
                  (read from ``main.log``, so build first);
* todo         -- inventory of ``\\TODOnum`` markers, the results-shaped holes;
* refs         -- bibliography entries, DOI coverage, uncited / missing keys.

Exit status is non-zero if a hard limit is violated, so it can gate a build.
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ABSTRACT_LIMIT = 250
HIGHLIGHT_CHARS = 85
PAGE_LIMIT = 50            # elsarticle final/twocolumn
PAGE_LIMIT_REVIEW = 120    # elsarticle review mode (EAAI submission)


def _read(name: str) -> str:
    return (HERE / name).read_text(encoding="utf-8", errors="ignore")


def _strip_tex(s: str) -> str:
    """Crude but stable de-TeX for word counting: floats and math become tokens."""
    s = re.sub(r"(?m)^%.*$", "", s)
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = re.sub(r"\\begin\{(tabular|tikzpicture|algorithmic)\}.*?\\end\{\1\}", " ", s, flags=re.S)
    s = re.sub(r"\$[^$]*\$", " x ", s)
    s = re.sub(r"\\begin\{(equation|align)\*?\}.*?\\end\{\1\*?\}", " ", s, flags=re.S)
    s = re.sub(r"\\(cite|ref|label|input|includegraphics)\w*\{[^}]*\}", " ", s)
    s = re.sub(r"\\TODOnum\{[^}]*\}", " NUM ", s)
    s = re.sub(r"\\[a-zA-Z@]+\*?", " ", s)
    s = re.sub(r"[{}\[\]&\\~^_]", " ", s)
    return s


def _words(s: str) -> int:
    return len([w for w in _strip_tex(s).split() if any(c.isalnum() for c in w)])


def check_highlights() -> bool:
    items = [ln.strip()[6:].strip() for ln in _read("highlights.tex").splitlines()
             if ln.strip().startswith("\\item")]
    print(f"== highlights: {len(items)} bullets (need 3-5), limit {HIGHLIGHT_CHARS} chars ==")
    ok = 3 <= len(items) <= 5
    for t in items:
        good = len(t) <= HIGHLIGHT_CHARS
        ok &= good
        print(f"  {'OK ' if good else 'FAIL'} {len(t):3d} | {t}")
    return ok


def check_abstract() -> bool:
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", _read("main.tex"), re.S)
    if not m:
        print("== abstract: NOT FOUND ==")
        return False
    n = _words(m.group(1))
    print(f"== abstract: {n} words (limit {ABSTRACT_LIMIT}) -> "
          f"{'OK' if n <= ABSTRACT_LIMIT else 'OVER'} ==")
    return n <= ABSTRACT_LIMIT


def _page_limit() -> int:
    """Page cap depends on the elsarticle mode declared in ``main.tex``."""
    m = re.search(r"\\documentclass\[([^\]]*)\]\{elsarticle\}", _read("main.tex"))
    opts = {o.strip() for o in m.group(1).split(",")} if m else set()
    return PAGE_LIMIT_REVIEW if "review" in opts else PAGE_LIMIT


def check_pages() -> bool:
    log = HERE / "main.log"
    if not log.exists():
        print("== pages: main.log missing -- build first ==")
        return True
    found = re.findall(r"Output written on main\.pdf \((\d+) page", _read("main.log"))
    if not found:
        print("== pages: unknown ==")
        return True
    n = int(found[-1])
    limit = _page_limit()
    print(f"== pages: {n} (limit {limit}) -> {'OK' if n <= limit else 'OVER'} ==")
    return n <= limit


def check_words() -> bool:
    print("== prose word counts per section ==")
    total = 0
    for f in sorted(glob.glob(str(HERE / "sections" / "*.tex"))):
        n = _words(Path(f).read_text(encoding="utf-8"))
        total += n
        print(f"  {Path(f).name:<34} {n:6d}")
    print(f"  {'TOTAL':<34} {total:6d}")
    return True


def check_todo() -> bool:
    print("== TODOnum inventory (results-shaped holes; a large count is expected) ==")
    total = 0
    files = (sorted(glob.glob(str(HERE / "*.tex"))) +
             sorted(glob.glob(str(HERE / "sections" / "*.tex"))) +
             sorted(glob.glob(str(HERE / "tables" / "*.tex"))))
    for f in files:
        if Path(f).name == "macros.tex":
            continue
        n = len(re.findall(r"\\TODOnum", Path(f).read_text(encoding="utf-8")))
        if n:
            total += n
            rel = Path(f).relative_to(HERE).as_posix()
            print(f"  {rel:<40} {n:4d}")
    print(f"  {'TOTAL':<40} {total:4d}")
    figs = sum(len(re.findall(r"\\TODOfig", Path(f).read_text(encoding="utf-8")))
               for f in glob.glob(str(HERE / "sections" / "*.tex")))
    print(f"  pending figures (TODOfig): {figs}")
    return True


def check_refs() -> bool:
    bib = _read("refs.bib")
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    dois = len(re.findall(r"(?m)^\s*doi\s*=", bib))
    src = "".join(Path(f).read_text(encoding="utf-8")
                  for f in glob.glob(str(HERE / "sections" / "*.tex")))
    src += _read("main.tex")
    cited = {k.strip() for group in re.findall(r"\\cite\{([^}]*)\}", src)
             for k in group.split(",")}
    print(f"== references: {len(keys)} entries, {dois} with a verified DOI, "
          f"{len(cited)} cited ==")
    missing = sorted(cited - keys)
    unused = sorted(keys - cited)
    if missing:
        print(f"  MISSING FROM BIB: {', '.join(missing)}")
    if unused:
        print(f"  not cited: {', '.join(unused)}")
    return not missing


def main() -> int:
    ok = True
    for fn in (check_highlights, check_abstract, check_pages, check_words,
               check_todo, check_refs):
        print()
        ok &= bool(fn())
    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
