rating.html: rating.qmd *.stan
	quarto render rating.qmd --to html

rating.pdf: rating.qmd *.stan
	quarto render rating.qmd --to pdf

clean:
	rm -f rating.ipynb plackett-luce.hpp plackett-luce rating.html rating.pdf ordinal-logit.hpp ordinal-logit ordinal-logit-prior.hpp ordinal-logit-prior rating-ranking.hpp rating-ranking

