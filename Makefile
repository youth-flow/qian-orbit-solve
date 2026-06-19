.PHONY: all data pdf check clean

PYTHON ?= python

all: data pdf

data:
	$(PYTHON) src/run_all.py

pdf:
	xelatex -interaction=nonstopmode -halt-on-error -file-line-error report.tex
	xelatex -interaction=nonstopmode -halt-on-error -file-line-error report.tex

check:
	$(PYTHON) scripts/check.py

clean:
	$(PYTHON) scripts/clean.py
