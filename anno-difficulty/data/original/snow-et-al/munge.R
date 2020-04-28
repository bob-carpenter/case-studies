# RTE DATA
y <- read.csv('unpacked/rte.standardized.tsv', sep = '\t')
df <- data.frame(coder = as.numeric(y$X.amt_worker_ids),
                 item = as.numeric(as.factor(y$orig_id)),
		 response = y$response,
		 gold = y$gold)

write_comments <- function(path, comments) {
  cat(paste("# ", comments, "\n", sep = ""),
      file = path, append = FALSE, sep = "")
}
path = "../../canonical/rte.csv"
write_comments(path, c(
  "Rion Snow, Brendan O’Connor, Daniel Jurafsky, Andrew Ng. 2008.",
  "Cheap and Fast---But is it Good? Evaluating Non-Expert Annotations",
  "for Natural Language Tasks. EMNLP 2008:254--263"))
write.table(df, file = path, append = TRUE,
            sep = ",", row.names = FALSE, quote = FALSE)
