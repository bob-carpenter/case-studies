data {
  int<lower=0> N;  // observations
  int<lower=0> D;  // predictors
  matrix[N, D] x;  // data matrix
  vector[N] y;     // observations
}
parameters {
  real alpha;
  vector[D] beta;
  real<lower=0> sigma;
}
model {
  y ~ normal(alpha + x * beta, sigma);
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
  sigma ~ lognormal(0, 1);
}
