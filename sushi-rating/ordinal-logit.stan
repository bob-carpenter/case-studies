data {
  int<lower=1> I;                        // # items
  int<lower=1> K;                        // # items per rater
  int<lower=0> R;                        // # raters
  array[R, K] int<lower=1, upper=I> u;   // items rated
  array[R, K] int<lower=1, upper=5> z;   // ordinal ratings
}
parameters {
  simplex[I] alpha;                      // item quality
  ordered[K - 1] c;                      // cutpoints
}
model {
  c ~ normal(0, 1);
  for (r in 1:R) {
    z[r] ~ ordered_logistic(alpha[u[r]], c);
  }
}
