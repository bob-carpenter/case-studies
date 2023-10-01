prob-corpus.html: prob-corpus.qmd *.stan
	quarto render prob-corpus.qmd --to html

prob-corpus.pdf: prob-corpus.qmd *.stan
	quarto render prob-corpus.qmd --to pdf

clean:
	rm -f prob-corpus.html prob-corpus.ipynb logistic-regression.hpp logistic-regression linear-regression.hpp linear-regression weighted-logistic-regression.hpp weighted-logistic-regression
