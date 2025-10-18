.PHONY: all re clean deps

TEX		 := xelatex
TEXFLAGS ?= -interaction=nonstopmode -halt-on-error -shell-escape
BIB		 ?= biber

SRCDIR	 := thesis
JOB		 := thesis
OUTDIR	 := build

# Dependencies
SRC		 := $(SRCDIR)/$(JOB).tex
PDF		 := $(SRCDIR)/$(OUTDIR)/$(JOB).pdf
BIBS	 := $(wildcard $(SRCDIR)/*.bib)
IMAGES	 := $(wildcard $(SRCDIR)/images/**/*) $(wildcard $(SRCDIR)/images/*)

all: $(PDF)

# Compile from thesis/ to match relative pahts (images/, diagrams/, etc.)
$(PDF): $(SRC) $(BIBS) $(IMAGES)
	@mkdir -p $(SRCDIR)/$(OUTDIR)
	cd $(SRCDIR) && \
		$(TEX) -output-directory=$(OUTDIR) $(TEXFLAGS) $(JOB).tex && \
		$(BIB) $(OUTDIR)/$(JOB) && \
		$(TEX) -output-directory=$(OUTDIR) $(TEXFLAGS) $(JOB).tex && \
		$(TEX) -output-directory=$(OUTDIR) $(TEXFLAGS) $(JOB).tex

# Force recompilation (use :make re)
re:
	cd $(SRCDIR) && \
		$(TEX) -output-directory=$(OUTDIR) $(TEXFLAGS) $(JOB).tex && \
		$(BIB) $(OUTDIR)/$(JOB) && \
		$(TEX) -output-directory=$(OUTDIR) $(TEXFLAGS) $(JOB).tex && \
		$(TEX) -output-directory=$(OUTDIR) $(TEXFLAGS) $(JOB).tex

deps:
	sudo tlmgr update --self
	sudo tlmgr install xurl hyperref babel-spanish csquotes natbib url

clean:
	rm -rf $(SRCDIR)/$(OUTDIR)

