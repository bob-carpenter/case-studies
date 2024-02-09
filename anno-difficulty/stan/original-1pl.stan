functions {
  array[] int total_response_count(array[ , ] int y) {
    array[6] int total_count = rep_array(0, 6);
    for (i in 1:size(y))
      total_count[sum(y[i]) + 1] += 1;
    return total_count;
  }
}
data {
  int<lower = 0> I;
  int<lower = 0> J;
  array[I, J] int<lower = 0, upper = 1> y;
}
transformed data {
  array[6] int<lower = 0> response_count = total_response_count(y);
  print("***** response count: ", response_count);
}
parameters {
  real pi;
  array[2] vector[J] theta;
  vector[I] beta;
}
model {
  pi ~ logistic(0, 1);
  for (k in 1:2)
    theta[k] ~ logistic(0, 1);
  beta ~ normal(0, 1);
  sum(beta) ~ normal(0, 1);

  for (i in 1:I)
    target += log_mix(inv_logit(pi),
                      bernoulli_logit_lpmf(y[i] | theta[1] - beta[i]),
                      bernoulli_logit_lpmf(y[i] | theta[2] + beta[i]));
}
generated quantities {
  array[6] int<lower = 0> pp_response_count;
  {
    array[I] int z_sim;
    array[I, J] int y_sim;
    for (i in 1:I)
      z_sim[i] = bernoulli_logit_rng(pi);
    for (i in 1:I)
      for (j in 1:J)
        y_sim[i, j] = bernoulli_logit_rng(z_sim[i] == 1
                                          ? theta[1, j] - beta[i]
                                          : theta[2, j] + beta[i]);
    pp_response_count = total_response_count(y_sim);
  }
}
