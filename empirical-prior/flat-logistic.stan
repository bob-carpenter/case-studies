data {
  int<lower=0> N;
  int<lower=0> D;
  matrix[N, D] x;
  array[N] int<lower=0, upper=1> y;
}
parameters {
  vector[D] beta;
}
model {
  y ~ bernoulli_logit(x * beta);
}
