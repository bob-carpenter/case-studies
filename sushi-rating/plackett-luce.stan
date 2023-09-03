data {
  int<lower=1> K;  // # items
  int<lower=1> J;  // # items ranked per observation
  int<lower=0> N;  // # observations
  array[N, J] y;   // ranks descending
}
parameters {
  simplex[K] alpha;
}
model {
  for (n in 1:N) {
    vector[J] alpha_y_n = alpha[y[n]];
    vector[J] probs = alpha_y_n ./ cumulative_sum(alpha_y_n);
    target += log(probs);
  }
}
