data {
  int<lower=1> D;                              // num predictors
  int<lower=0> N;                              // num observations
  matrix[N, D] x;                              // predictors
  vector<lower=0, upper=1>[N] p;               // Pr[Y_n = 1 | x_n]
}
parameters {
  vector[D] beta;                     // regression coefficients
}
model {
  logit(p) ~ normal(x * beta, 1);     // likelihood: linear regression on log odds
  beta ~ normal(0, 1);                // prior:      ridge
}
