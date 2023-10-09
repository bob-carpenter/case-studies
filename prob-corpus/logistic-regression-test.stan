data {
  int<lower=0> N_t;                     // num observations
  matrix[N, D] x_t;                     // test covariates
  array[N] int<lower=0, upper=1> y_t;   // test outcomes
}
parameters {
  vector[D] beta;
}
generated quantities {
  real log_p_t = bernoulli_logit_lpmf(y_t | x_t * beta);
}