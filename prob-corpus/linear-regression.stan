data {
  int<lower=1> K;                     // num predictors
  int<lower=0> N;                     // num observations
  matrix[N, K] x;                     // predictors
  vector[N] y;                        // log odds Pr[outcome = 1]
}
parameters {
  vector[K] beta;
}
model {
  y ~ normal(x * beta, 1);            // fixed scale
  beta ~ normal(0, 1);
}