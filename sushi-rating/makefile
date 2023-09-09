rating.html: rating.qmd *.stan
	quarto render rating.qmd --to html

rating.pdf: rating.qmd *.stan
	quarto render rating.qmd --to pdf

clean:
	rm -f plackett-luce.hpp plackett-luce rating.html rating.pdf
