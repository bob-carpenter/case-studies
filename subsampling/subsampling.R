library(rstan)
library(ggplot2)

printf <- function(msg, ...) cat(sprintf(msg, ...)); cat('\n')

inv_logit <- function(u) 1 / (1 + exp(-u))

program <- "
data {
  int<lower = 0> N;
  vector[N] x;
  int y[N];
}
parameters {
  real alpha;
  real beta;
}
model {
  alpha ~ normal(0, 2);
  beta ~ normal(0, 2);
  y ~ bernoulli_logit(alpha + x * beta);
}
"

model <- stan_model(model_code = program)

G <- 15
N <- 50
x <- matrix(rnorm(N * G), N, G)
alpha <- 1
beta <- -1
y <- matrix(rbinom(G * N, 1, inv_logit(alpha + beta * as.vector(x))), N, G)

df <- data.frame()
for (g in 1:G) {
  fit <- sampling(model, data = list(N = N, x = x[, g], y = y[ ,g]), refresh = 0)
  ss <- extract(fit)
  df <- rbind(df, data.frame(alpha = ss$alpha, beta = ss$beta,
                             run =  rep(paste("run", g), 4000)))
}
big_fit <-
  sampling(model,
           data = list(N = G * N, x = as.vector(x), y = as.vector(y)),
	   refresh = 0)
ss <- extract(big_fit)
df <- rbind(df, data.frame(alpha = ss$alpha, beta = ss$beta,
                             run =  rep("combined", 4000)))

plot <-
  ggplot(df, aes(x = alpha, y = beta)) +
  facet_wrap(run ~ .) +
  geom_hline(yintercept = beta, size = 0.25, color = "red", alpha = 0.5) +
  geom_vline(xintercept = alpha, size = 0.25, color = "red", alpha = 0.5) +
  geom_point(size = 0.1, alpha = 0.1) +
  scale_x_continuous(lim = c(-0.5, 2.5), breaks = c(0, 1, 2)) +
  scale_y_continuous(lim = c(-4, 2), breaks = c(-4, -1, 2))

plot
