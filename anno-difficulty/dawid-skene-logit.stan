functions {
  int[] total_response_counts(int[ , ] y) {
    int total_counts[6] = rep_array(0, 6);  // indexing + 1
    for (i in 1:size(y))
      total_counts[sum(y[i]) + 1] += 1;
    return total_counts;
  }
}
data {
  int<lower = 0> I;
  int<lower = 0> J;
  int<lower = 0, upper = 1> y[I, J];
}
transformed data {
  int<lower = 0> total_counts[6] = total_response_counts(y);
}
parameters {
  real pi;
  vector[J] theta[2];
}
model {
  // prior
  pi ~ logistic(0, 1);
  for (k in 1:2)
    theta[k] ~ logistic(0, 1);

  // likelihood
  for (i in 1:I)
    target += log_mix(inv_logit(pi),
                      bernoulli_logit_lpmf(y[i] | theta[1]),
                      bernoulli_logit_lpmf(y[i] | theta[2]));
}
generated quantities {
  int<lower = 0> total_counts_sim[6];
  int<lower = 0, upper = 1> total_counts_gt_sim[6];
  {
    int z_sim[I];
    int y_sim[I, J];
    for (i in 1:I)
      z_sim[i] = bernoulli_logit_rng(pi);
    for (i in 1:I)
      for (j in 1:J)
        y_sim[i, j] = bernoulli_logit_rng(theta[z_sim[i] == 1 ? 1 : 2, j]);
    total_counts_sim = total_response_counts(y_sim);
  }
  for (n in 1:6)
    total_counts_gt_sim[n] = total_counts[n] > total_counts_sim[n];
}
