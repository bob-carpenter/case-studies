data {
  int<lower = 0> K;
  int<lower = 0> N;
  matrix[N, K] x;
  int y[N];
}
parameters {
  real alpha;
  vector[K] beta;
}
model {
  alpha ~ normal(0, 2);
  beta ~ normal(0, 2);
  y ~ bernoulli_logit(alpha + x * beta);
}
