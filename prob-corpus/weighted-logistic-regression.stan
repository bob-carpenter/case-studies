data {
  int<lower=1> D;                        // num predictors
  int<lower=0> N;                        // num observations
  matrix[N, D] x;                        // covariates
  vector<lower=0, upper=1>[N] p;         // Pr[Y[n] = 1]
}
parameters {
  vector[D] beta;                        // regression coefficients
}
model {
  vector[N] E_Y = inv_logit(x * beta);   // expected Y 
  target += sum(p .* log(E_Y));          // likelihood: weighted logistic regression
  target += sum((1 - p) .* log1m(E_Y));  // 
  beta ~ normal(0, 1);                   // prior:    ridge
}