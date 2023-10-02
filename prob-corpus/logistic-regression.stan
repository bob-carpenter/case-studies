data {
  int<lower=1> D;                    // num predictors
  int<lower=0> N;                    // num observations
  matrix[N, D] x;                    // covariates
  array[N] int<lower=0, upper=1> y;  // outcomes Y[n] = y[n]
}
parameters {
  vector[D] beta;                    // regression coefficients
}
model {
  y ~ bernoulli_logit(x * beta);     // likelihood: logistic regression
  beta ~ normal(0, 1);               // prior:      ridge
}
