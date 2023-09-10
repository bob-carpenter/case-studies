functions {
  real plackett_luce_lpmf(array[] int y, vector beta) {
    vector[size(y)] beta_y = beta[y];
    return sum(log(beta_y ./ cumulative_sum(beta_y)));
  }
}
data {
  int<lower=1> I;                        // # items
  int<lower=1> K;                        // # items per rater
  int<lower=0> R;                        // # raters
  int<lower=0> J;                        // maximum rating
  array[R, K] int<lower=1, upper=I> y;   // rankings (y[r, 1] < y[r, 2] < ...)
  array[R, K] int<lower=1, upper=I> u;   // items rated
  array[R, K] int<lower=1, upper=J> z;   // ordinal ratings
}
parameters {
  simplex[I] beta;                       // item quality (ranking)
  ordered[K - 1] c;                      // cutpoints
  real alpha;                            // intercept
  real<lower=0> gamma;                   // slope (enforce monotonicity)
}
transformed parameters {
  // rating quality is log-linear function of ranking quality
  vector[I] eta = alpha + gamma * log(beta);  
}
model {
  c ~ normal(0, 3);
  alpha ~ normal(0, 3);
  gamma ~ lognormal(0, 0.5);
  for (r in 1:R) {
    y[r] ~ plackett_luce(beta);
  }
  for (r in 1:R) {
    z[r] ~ ordered_logistic(eta[u[r]], c);
  }
}
