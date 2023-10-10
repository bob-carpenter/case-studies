data {
  int<lower=1> D;                                   // num covariates per item
  int<lower=0> N;                                   // num observations
  matrix[N, D] x;                                   // test covariates
  array[N] int<lower=0, upper=1> y;                 // outcomes
}
parameters {
  vector[D] beta;                                   // parameters
}
generated quantities {
  real log_p = bernoulli_logit_lpmf(y | x * beta);  // likelihood
}
