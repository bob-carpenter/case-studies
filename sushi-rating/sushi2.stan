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
  array[S] vector[S] log_prob_prefer;
  for (s in 1:S)
    log_prob_prefer[s] = log_inv_logit(alpha[s] - alpha);
  for (n in 1:N)
    for (k in 1:K - 1)
      target += log_prob_prefer[y[n, k], y[n, k+1:K]];
}
