#!/bin/sh

Rscript -e "rmarkdown::render('"${1}"')"
