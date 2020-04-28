write_comments <- function(path, comments) {
  cat(paste("# ", comments, "\n", sep = ""),
      file = path, append = FALSE, sep = "")
}

space_sep_missing_data <- function(path_in, sep, path_out, citation) {
  y <- read.csv(path_in, sep = sep, header = FALSE)
  I <- dim(y)[1]
  K <- dim(y)[2]
  N <- sum(!is.na(y) && y != -1)
  n <- 1
  coder <- rep(NA, N)
  item <- rep(NA, N)
  response <- rep(NA, N)
  for (i in 1:I) {
    for (k in 1:K) {
      if (!is.na(y[i, k]) && y[i, k] != -1) {
        coder[n] <- k
        item[n] <- i
        response[n] <- as.numeric(y[i, k])
        n <- n + 1
      }
    }
  }
  df <- data.frame(coder, item, response)
  write_comments(path_out, citation)
  write.table(df, file = path_out, append = TRUE,
              sep = ",", row.names = FALSE, quote = FALSE)
}

citation <- c("Andre Rzhetsky, Hagit Shatkay, and W. John Wilbur. 2009.",
  "How to get the most out of your curation effort. PLoS Computational",
  "Biology.",
  "https://github.com/enthought/uchicago-pyanno/blob/master/data")

space_sep_missing_data('e-2.txt', ',',
                       '../../canonical/rzhetsky-e-2.csv',
		       citation)
space_sep_missing_data('testdata_irregular.txt', ' ',
                       '../../canonical/rzhetsky-irregular.csv',
		       citation)
space_sep_missing_data('testdata_large.txt', ' ',
                       '../../canonical/rzhetsky-large.csv',
		       citation)
space_sep_missing_data('testdata_numerical.txt', ' ',
                       '../../canonical/rzhetsky-numerical.csv',
		       citation)
space_sep_missing_data('testdata_words.txt', ' ',
                       '../../canonical/rzhetsky-words.csv',
  c(citation,
    "coded as factor alphabetically: HIGH = 1, LOW = 2, MED = 3"))
