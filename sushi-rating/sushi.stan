data {
  int<lower=0> S;           // number of sushi types
  int<lower=0> N;           // number of raters
  int<lower=1, upper=S> K;  // number of types per ordering
  array[N, K] int y;        // orderings
}
parameters {
  vector[S - 1] alpha_prefix;
}
transformed parameters {
  vector[S] alpha = append_row(alpha_prefix, -sum(alpha_prefix));
}
model {
  alpha ~ normal(0, 4);
  for (n in 1:N) {
    for (k in 1:K - 1) {
      1 ~ bernoulli_logit(alpha[y[n, k]] - alpha[y[n, k+1:K]]);
    }
  }
}
