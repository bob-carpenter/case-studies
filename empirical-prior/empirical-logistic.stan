data {
  int<lower=0> N;
  int<lower=0> D;
  matrix[N, D] x;
  array[N] int<lower=0, upper=1> y;

  real<lower=0> h;
  int<lower=0> B;
  array[B] vector[D] beta0;

}
parameters {
  vector[D] beta;
}
model {
  y ~ bernoulli_logit(x * beta);
  vector[B] lp;
  for (b in 1:B) {
    lp[b] = normal_lpdf(beta | beta0[b], h);
  }
  target += log_sum_exp(lp) - log(B);
}
