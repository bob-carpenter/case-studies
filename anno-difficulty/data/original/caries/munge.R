caries_data <- read.table('caries-data.csv', header = TRUE, sep = ',')
J <- 5
N <- sum(caries_data$count)
df <- data.frame(coder = rep(NA, N),
                 item = rep(NA, N),
                 response = rep(NA, N))
pos <- 1
n <- 1
item <- 1
y <- matrix(NA, , 3)
for (i1 in 0:1) {
  for (i2 in 0:1) {
    for (i3 in 0:1) {
      for (i4 in 0:1) {
        for (i5 in 0:1) {
          for (m in 1:caries_data$count[pos]) {
	    df[n, ] <- c(1, item, i1)
	    n <- n + 1
	    df[n, ] <- c(2, item, i2)
	    n <- n + 1
	    df[n, ] <- c(3, item, i3)
	    n <- n + 1
	    df[n, ] <- c(4, item, i4)
	    n <- n + 1
	    df[n, ] <- c(5, item, i5)
	    n <- n + 1
	    item <- item + 1
          }
	  pos <- pos + 1
	}
      }
    }
  }
}

write_comments <- function(path, comments) {
  cat(paste("# ", comments, "\n", sep = ""),
      file = path, append = FALSE, sep = "")
}

path = "../../canonical/caries.csv"
write_comments(path,
  c("Espeland, M. A. and S. L. Handelman. 1989. Using latent class models",
    "to characterize and assess relative-error in discrete measurements.",
    "Biometrics 45:587--599."))
write.table(df, sep = ",", row.names = FALSE, col.names = TRUE, quote = FALSE,
            file = path, append = TRUE)
