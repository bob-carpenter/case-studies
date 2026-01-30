data {
  int<lower=0> N;
  int<lower=0> R;
  int<lower=0> C;
  array[N] int<lower=1, upper=R> ii;
  array[N] int<lower=1, upper=C> jj;
  vector[N] y;
}
parameters {
  real mu;
  sum_to_zero_vector[R] alpha;
  sum_to_zero_vector[C] beta;
  real<lower=0> sigma, sigma_alpha, sigma_beta;
}
model {
  { sigma_alpha, sigma_beta } ~ lognormal(0, 0.5);
  alpha ~ normal(0, sigma_alpha);
  beta ~ normal(0, sigma_beta);
  mu ~ normal(0, 1);
  sigma ~ lognormal(0, 0.5);

  y ~ normal(mu + alpha[ii] + beta[jj], sigma);
}