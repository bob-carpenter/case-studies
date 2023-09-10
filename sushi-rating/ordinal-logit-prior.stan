data {
  int<lower=1> I;                        // # items
  int<lower=1> K;                        // # items per rater
}
parameters {
  ordered[K - 1] c;                      // cutpoints
  vector[I - 1] eta_pre;                 // item quality
}
transformed parameters {
  vector[I] eta = append_row(eta_pre, -sum(eta_pre));
}
model {
  eta_pre ~ normal(0, 3);
  c ~ normal(0, 3);
}
