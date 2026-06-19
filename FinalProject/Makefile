.PHONY: all data pdf clean

PYTHON ?= python

all: data pdf

data:
	$(PYTHON) src/run_all.py

pdf:
	xelatex -interaction=nonstopmode -halt-on-error -file-line-error report.tex
	xelatex -interaction=nonstopmode -halt-on-error -file-line-error report.tex

clean:
	powershell -NoProfile -ExecutionPolicy Bypass -Command "Remove-Item -Recurse -Force build,data/generated,src/__pycache__ -ErrorAction SilentlyContinue; Remove-Item -Force *.aux,*.bbl,*.bcf,*.blg,*.fdb_latexmk,*.fls,*.log,*.out,*.run.xml,*.synctex.gz,*.toc,*.xdv,report.pdf -ErrorAction SilentlyContinue"
