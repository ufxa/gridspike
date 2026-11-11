"""Filter supplementary.aux down to its \\newlabel lines into supplabels.aux.

main.tex pulls cross-references from the supplementary document through
``\\externaldocument{supplabels}`` (package xr). xr would otherwise import
the supplement's ``\\bibcite`` entries as well, renumbering citations that
appear in both documents, so the bibliography lines are stripped before
main is compiled. Labels keep their S-numbers because the supplement sets
``\\thetable``/``\\thefigure``/``\\thesection``/``\\theequation`` to an
S-prefixed form.

Usage (from the paper/ directory, after compiling supplementary.tex):
    python supplabels.py
"""
import sys


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else "supplementary.aux"
    dst = sys.argv[2] if len(sys.argv) > 2 else "supplabels.aux"
    with open(src, encoding="utf-8", errors="replace") as handle:
        labels = [line for line in handle if line.startswith("\\newlabel")]
    with open(dst, "w", encoding="ascii", newline="\n") as handle:
        handle.writelines(labels)
    print(f"supplabels.aux: {len(labels)} labels from {src}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
