data {
  int<lower=1> K;                     // num predictors
  int<lower=0> N;                     // num observations
  matrix[N, K] x;                     // predictors
  vector<lower=0, upper=1>[N] y;         // Pr[outcome = 1]
}
parameters {
  vector[K] beta;
}
model {
  beta ~ normal(0, 1);
  vector[N] E_y = inv_logit(x * beta);
  target += sum(y .* log(E_y));
  target += sum((1 - y) .* log1m(E_y));
}