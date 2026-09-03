.PHONY: all clean verify
all: main.pdf

main.pdf: main.tex references.bib snapshot_manifest.json
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	bibtex main
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex

verify:
	python3 verify_snapshot.py

clean:
	rm -f main.aux main.bbl main.blg main.log main.nav main.out main.snm main.toc main.vrb
