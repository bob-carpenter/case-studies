library('ggplot2')

df <- read.csv('histo.csv')  # kmer-counts
histplot <-
    ggplot(df, aes(x = count)) +
    geom_histogram(color="white") +
    scale_x_log10(breaks = 10^(0:5), labels = c("0", "10", "100", "1K", "10K", "100K")) +
    scale_y_log10(breaks = 10^(0:6), labels = c("0", "10", "100", "1K", "10K", "100K", "1M")) +
    xlab("kmer frequency in transcriptome") +
    ylab("count")
ggsave("transcriptome-kmers.pdf", histplot)
