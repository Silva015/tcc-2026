PROJECT = poster
PDFLATEX = pdflatex -interaction=nonstopmode

.PHONY: all build clean preview
all: build

build: $(PROJECT).tex
	$(PDFLATEX) $(PROJECT)
	$(PDFLATEX) $(PROJECT)

# Low-resolution PNG preview of the A0 page
preview: build
	pdftoppm -r 30 -png -singlefile $(PROJECT).pdf $(PROJECT)-preview

clean:
	rm -f $(PROJECT).aux $(PROJECT).log $(PROJECT).out build.log $(PROJECT)-preview.png
