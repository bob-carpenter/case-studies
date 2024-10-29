functions {
  real plackett_luce_lpmf(array[] int y, vector beta) {
    vector[size(y)] beta_y = beta[y];
    return sum(log(beta_y ./ cumulative_sum(beta_y)));
  }
}
data {
  int<lower=1> I;                       // # items
  int<lower=1> K;                       // # items ranked per rater
  int<lower=1> R;                       // # raters
  array[R, K] int<lower=1, upper=I> y;  // rankings (y[r, 1] > y[r, 2] > ...)
}
parameters {
  simplex[I] beta;                      // item quality
}
model {
  for (r in 1:R) {
    y[r] ~ plackett_luce(beta);
  }
}
