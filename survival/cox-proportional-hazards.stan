data {
  int<lower=0> K;          // num covariates

  int<lower=0> N;          // num uncensored obs
  vector[N] t;             // event time (non-strict decreasing)
  matrix[N, K] x;          // covariates for uncensored obs

  int N_c;                 // num censored obs
  real<lower=t[N]> t_c;    // censoring time
  matrix[N_c, K] x_c;      // covariates for censored obs
}
parameters {
  vector[K] beta;          // slopes (no intercept)
}
model {
  // prior
  beta ~ normal(0, 2);

  // likelihood
  vector[N] log_theta = x * beta;
  vector[N_c] log_theta_c = x_c * beta;
  real log_denom = log_sum_exp(log_theta_c);
  for (n in 1:N) {
    log_denom = log_sum_exp(log_denom, log_theta[n]);
    target += log_theta[n] - log_denom;   // log likelihood
  }
}
