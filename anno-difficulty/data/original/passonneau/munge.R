write_comments <- function(path, comments) {
  cat(paste("# ", comments, "\n", sep = ""),
      file = path, append = FALSE, sep = "")
}

convert_format <- function(prefix) {
  path_in <- paste(prefix, ".tsv", sep = "")
  path_out <- paste("../../canonical/passonneau-et-al/", prefix, ".csv",
                    sep = "")
  y <- read.csv(path, sep = "\t", header = TRUE)
  df <- data.frame(item = y$item,
                   coder = y$annotator,
                   response = y$rating)
  citation <- c("Rebecca Passonneau and Bob Carpenter. 2014.  The",
                "benefits of a model of annotation.  TACL 2.",
                "https://github.com/bob-carpenter/anno")
  write_comments(path_out, citation)
  write.table(df, file = path_out, append = TRUE,
              sep = ",", row.names = FALSE, quote = FALSE)
}

convert_format("color-n")
convert_format("control-n")
convert_format("fair-j")
convert_format("find-v")
convert_format("full-j")
convert_format("help-v")
