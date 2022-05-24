N <- 1000
D <- 20
alpha <- 1
beta <- rnorm(D)
sigma <- 2
x <- matrix(rnorm(N * D), N, D)
y <- rnorm(alpha + x %*% beta, sigma)

library('rstan')
vars <- c("N", "D", "x", "y")
stan_rdump(vars, file="eg.data.R")
