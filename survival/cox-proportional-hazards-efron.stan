functions {
  int num_unique_starts(vector t) {
    if (size(t) == 0) return 0;
    int us = 1;
    for (n in 2:size(t)) {
      if (t[n] != t[n - 1]) us += 1;
    }
    return us;
  }
  array[] int unique_starts(vector t, int J) {
    array[J + 1] int starts;
    if (J == 0) return starts;
    starts[1] = 1;
    int pos = 2;
    for (n in 2:size(t)) {
      if (t[n] != t[n - 1]) {
	starts[pos] = n;
	pos += 1;
      }
    }
    starts[J + 1] = size(t) + 1;
    return starts;
  }
}
data {
  int<lower=0> K;          // num covariates

  int<lower=0> N;          // num uncensored obs
  vector[N] t;             // event time (non-strict decreasing)
  matrix[N, K] x;          // covariates for uncensored obs

  int N_c;                 // num censored obs
  real<lower=t[N]> t_c;    // censoring time
  matrix[N_c, K] x_c;      // covariates for censored obs
}
transformed data {
  int<lower=0> J = num_unique_starts(t);
  array[J + 1] int<lower=0> starts = unique_starts(t, J);
  print("starts = ", starts);
}
parameters {
  vector[K] beta;          // slopes (no intercept)
}
model {
  beta ~ normal(0, 2);     // prior

  vector[N] log_theta = x * beta;
  vector[N_c] log_theta_c = x_c * beta;
  real log_denom_lhs = log_sum_exp(log_theta_c);
  for (j in 1:J) {
    int start = starts[j];
    int end = starts[j + 1] - 1;
    int len = end - start + 1;
    real numerator = sum(log_theta[start:end]);
    log_denom_lhs = log_sum_exp(log_denom_lhs, log_sum_exp(log_theta[start:end]));
    vector[len] diff;
    for (ell in 1:len) {
      diff[ell] = log_diff_exp(log_denom_lhs, log(ell - 1) - log(len) + log_sum_exp(log_theta[start:end]));
    }
    target += numerator - sum(diff);
  }
}
