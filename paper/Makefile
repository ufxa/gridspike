# ===========================================================================
# GridSpike manuscript build.
#
#   make            -> main.pdf (anonymized manuscript) + titlepage.pdf
#                      + supplementary.pdf
#   make main       -> main.pdf only
#   make titlepage  -> titlepage.pdf only (separate, non-anonymized)
#   make supplementary -> supplementary.pdf only (companion doc, S-numbered)
#   make check      -> compliance report: highlights, abstract, pages,
#                      word counts, TODOnum inventory, bibliography
#   make clean      -> remove build intermediates
#   make distclean  -> also remove the PDFs
#
# Cross-document order: supplementary.tex and main.tex reference each
# other's labels through xr (supplementary -> main.aux; main -> a filtered
# supplabels.aux so the supplement's \bibcite entries are not imported).
# Build supplementary first, regenerate supplabels.aux, then main; a
# second supplementary pass afterwards picks up main's freshest numbers.
# `make` below runs exactly that sequence.
#
# Requires a TeX distribution with latexmk and the elsarticle class, plus
# python3 for `make check`. MiKTeX auto-installs missing packages on first
# use (-enable-installer); on TeX Live the flag is simply ignored as an
# unknown pdflatex option, so drop it there if pdflatex complains.
# ===========================================================================

LATEXMK  ?= latexmk
PYTHON   ?= python
PDFLATEX ?= pdflatex -enable-installer %O %S
LMKFLAGS ?= -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
            -pdflatex="$(PDFLATEX)"

SECTIONS := $(wildcard sections/*.tex)
TABLES   := $(wildcard tables/*.tex)

.PHONY: all main titlepage supplementary supplabels check clean distclean

all: supplementary main titlepage

main: main.pdf
titlepage: titlepage.pdf
supplementary: supplementary.pdf

supplabels: supplabels.aux

supplementary.pdf: supplementary.tex macros.tex refs.bib $(TABLES)
	$(LATEXMK) $(LMKFLAGS) supplementary.tex

# xr reads only the supplement's \newlabel lines (see supplabels.py);
# the latexmk run behind supplementary.pdf is what produces that aux.
supplabels.aux: supplementary.pdf
	$(PYTHON) supplabels.py

main.pdf: main.tex highlights.tex macros.tex refs.bib $(SECTIONS) $(TABLES) supplabels.aux
	$(LATEXMK) $(LMKFLAGS) main.tex

titlepage.pdf: titlepage.tex macros.tex
	$(LATEXMK) $(LMKFLAGS) titlepage.tex

check:
	$(PYTHON) check.py

clean:
	-$(LATEXMK) -c main.tex
	-$(LATEXMK) -c supplementary.tex
	-$(LATEXMK) -c titlepage.tex
	-rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out *.spl \
	       *.synctex.gz build*.log sections/*.aux supplabels.aux

distclean: clean
	-rm -f main.pdf supplementary.pdf titlepage.pdf
