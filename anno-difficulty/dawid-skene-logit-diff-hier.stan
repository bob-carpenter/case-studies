/**
 * Copyright (2019) Bob Carpenter.  Released under BSD 3-clause license.
 *
 * Dawid and Skene model for binary data given uniform priors and then
 * reparameterized to the log odds scale, with the addition of a
 * centered normal prior on difficulty and the addition of
 * hierarchical priors on ability and difficulty.
 *
 * The model is hard coded for five annotators, each of whom annotates
 * every item.
 *
 * TODO:  In Stan 2.20, the affine transforms will make this easier to
 * understand.
 *
 * Dawid, A. P., & Skene, A. M. (1979) Maximum likelihood estimation
 * of observer error-rates using the EM algorithm. *Applied
 * Statistics*, 20--28.
 *
 * Copyright (2019) Bob Carpenter.  Released under BSD 3-clause license.
 */
functions {
  int[] total_response_counts(int[ , ] y) {
    int total_counts[6] = rep_array(0, 6);  // indexing + 1
    for (i in 1:size(y))
      total_counts[sum(y[i]) + 1] += 1;
    return total_counts;
  }
}
data {
  int<lower = 0> I;
  int<lower = 0> J;
  int<lower = 0, upper = 1> y[I, J];
}
transformed data {
  int<lower = 0> total_counts[6] = total_response_counts(y);
}
parameters {
  real<upper = 0> pi;  // constraint identifies mode

  real mu_theta[2];
  real<lower = 0> sigma_theta[2];
  vector[J] theta_std[2];

  real<lower = 0> sigma_beta;
  vector[I] beta_std;
}
transformed parameters {
  vector[J] theta[2] = { mu_theta[1] + theta_std[1] * sigma_theta[1],
                         mu_theta[2] + theta_std[2] * sigma_theta[2] };
  real inv_logit_pi = inv_logit(pi);
  vector[I] beta = beta_std * sigma_beta;
}
model {
  pi ~ logistic(0, 1);

  mu_theta ~ normal(0, 2);
  sigma_theta ~ normal(0, 1);
  theta_std[1] ~ normal(0, 1);
  theta_std[2] ~ normal(0, 1);

  sigma_beta ~ normal(0, 0.5);
  beta_std ~ normal(0, 1);
  sum(beta) ~ normal(0, 1);

  // marginalized log likelihood: log p(y | theta, beta, pi)
  for (i in 1:I)
    target += log_mix(inv_logit_pi,
                      bernoulli_logit_lpmf(y[i] | theta[1] - beta[i]),
                      bernoulli_logit_lpmf(y[i] | theta[2] + beta[i]));
}
generated quantities {
  int<lower = 0> total_counts_sim[6];
  int<lower = 0, upper = 1> total_counts_gt_sim[6];
  real<lower = 0, upper = 1> Pr_z_eq_1;
  {
    int z_sim[I];
    int y_sim[I, J];
    for (i in 1:I)
      z_sim[i] = bernoulli_logit_rng(pi);
    for (i in 1:I)
      for (j in 1:J)
        y_sim[i, j] = bernoulli_logit_rng(z_sim[i] == 1
                                          ? theta[1, j] - beta[i]
                                          : theta[2, j] + beta[i]);
    total_counts_sim = total_response_counts(y_sim);
    Pr_z_eq_1 = sum(z_sim) / (1.0 * I);  // more accurate in expectation
  }
  for (n in 1:6)
    total_counts_gt_sim[n] = total_counts[n] > total_counts_sim[n];
}
