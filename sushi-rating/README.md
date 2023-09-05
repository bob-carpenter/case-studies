# Bayesian models of data ranking and rating

## Building the case study

### HTML format

From the command line interface, you can change directory to the
top-level directory of the repository and invoke `make`:

```
$ cd <sushi-rating>
$ make
```

will run the quarto build process and produce an html document in the
file `sushi-rating.html`.  To run the same thing manually:

```
$ quarto render rating.qmd --to html 
```

### PDF format

To produce the pdf, you can first change to the top-level directory,
then run `make`:

```
$ cd <sushi-rating>
$ make sushi-rating.pdf
```

To run the same thing manually, you can do this

```
$ quarto render rating.qmd --to pdf
```
