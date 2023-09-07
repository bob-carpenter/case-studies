functions {
  real plackett_luce_lpmf(array[] int y, vector alpha) {
    vector[size(y)] alpha_y = alpha[y];
    return sum(log(alpha_y ./ cumulative_sum(alpha_y)));
  }
}
data {
  int<lower=1> I;                       // # items
  int<lower=1> K;                       // # items ranked per rater
  int<lower=1> R;                       // # raters
  array[R, K] int<lower=1, upper=I> y;  // rankings (y[r, 1] < y[r, 2] < ...)
}
parameters {
  simplex[I] alpha;                     // item quality
}
model {
  for (y_r in y)
    y_r ~ plackett_luce(alpha);
}
