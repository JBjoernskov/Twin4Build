# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source

# Check if we're on ReadTheDocs
ifdef READTHEDOCS_OUTPUT
BUILDDIR      = $(READTHEDOCS_OUTPUT)
else
BUILDDIR      = build
endif

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

buildapi:
	rm -rf source/auto
	sphinx-apidoc -f -o source/auto ../twin4build --maxdepth=1 ../twin4build/core ../twin4build/tests ../twin4build/examples
	python clean_sphinx_docs.py
	@echo "Auto-generation of API documentation finished. " \
		"The generated files are in 'source/auto/'"