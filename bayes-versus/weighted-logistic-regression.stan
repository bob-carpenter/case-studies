data {
  int<lower = 0> K;
  int<lower = 0> N;
  matrix[N, K] x;
  vector[N] phi;
}
transformed data {
  vector[N] inv_logit_phi = inv_logit(phi);
}
parameters {
  real alpha;
  vector[K] beta;
}
model {
  vector[N] log_odds = alpha + x * beta;
  alpha ~ normal(0, 2);
  beta ~ normal(0, 2);
  for (n in 1:N) {
    target += inv_logit_phi[n] * bernoulli_logit_lpmf(1 | log_odds[n]);
    target += (1 - inv_logit_phi[n]) * bernoulli_logit_lpmf(0 | log_odds[n]);
  }
}
