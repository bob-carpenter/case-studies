data {
  int<lower=0> S;           // number of sushi types
  int<lower=0> N;           // number of raters
  int<lower=1, upper=S> K;  // number of types per ordering
  array[N, K] int y;        // orderings
}
parameters {
  vector<lower=0, upper=1>[S] alpha;  // overall rating
  real<lower=0> beta;                 // scale
}
model {
  alpha ~ uniform(0, 1);
  beta ~ lognormal(log(5), 1);
  for (n in 1:N) {
    for (k in 1:K - 1) {
      1 ~ bernoulli_logit(beta * (alpha[y[n, k]] - alpha[y[n, k+1:K]]));
    }
  }
}
