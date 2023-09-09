data {
  int<lower=1> I;                        // # items
  int<lower=1> K;                        // # items per rater
  int<lower=0> R;                        // # raters
  array[R, K] int<lower=1, upper=I> u;   // items rated
  array[R, K] int<lower=1, upper=K> z;   // ordinal ratings
}
parameters {
  ordered[K - 1] c;                      // cutpoints
  vector[I - 1] eta_pre;                 // item quality
}
transformed parameters {
  vector[I] eta = append_row(eta_pre, -sum(eta_pre));
}
model {
  eta ~ normal(0, 3);
  c ~ normal(0, 3);
  for (r in 1:R) {
    z[r] ~ ordered_logistic(eta[u[r]], c);
  }
}
