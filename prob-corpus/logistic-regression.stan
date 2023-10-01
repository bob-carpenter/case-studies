data {
  int<lower=1> K;                    // num predictors
  int<lower=0> N;                    // num observations
  matrix[N, K] x;                    // predictors
  array[N] int<lower=0, upper=1> y;  // outcomes
}
parameters {
  vector[K] beta;
}
model {
  y ~ bernoulli_logit(x * beta);
  beta ~ normal(0, 1);
}
