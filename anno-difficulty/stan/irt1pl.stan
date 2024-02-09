data {
  int<lower=1> I;
  int<lower=1> J;
  int<lower=1> N;
  array[N] int<lower=0, upper=1> rating; 
  array[N] int<lower=1, upper=I> item; 
  array[N] int<lower=1, upper=J> rater; 
}
parameters {
  real logit_prev;
  vector<lower=0>[J] logit_sens;
  vector<lower=0>[J] logit_spec;
  vector[I] logit_diff;
}
model {
  // prior
  logit_prev ~ logistic(0, 1);
  logit_sens ~ logistic(2, 2);
  logit_spec ~ logistic(2, 2);
  logit_diff ~ logistic(0, 0.5);
  sum(logit_diff) ~ normal(0, 1);  // soft center

  // likelihood
  vector[I] pos_lp = rep_vector(log_inv_logit(logit_prev), I);
  vector[I] neg_lp = rep_vector(log1m_inv_logit(logit_prev), I);
  for (n in 1:N) {
    pos_lp[item[n]] += bernoulli_logit_lpmf(rating[n] | logit_sens[rater[n]] - logit_diff[item[n]]);
    neg_lp[item[n]] += bernoulli_logit_lpmf(rating[n] | -(logit_spec[rater[n]] - logit_diff[item[n]]));
  }
  for (i in 1:I) {
    target += log_sum_exp(pos_lp[i], neg_lp[i]);
  }
}
generated quantities {
  array[J + 1] int votes_sim = rep_array(0, J + 1); 
  for (i in 1:I) {
    int votes = 0;
    for (j in 1:J) {
      int z = bernoulli_logit_rng(logit_prev);
      int y = bernoulli_logit_rng(z
                                  ? logit_sens[j] - logit_diff[i]
                                  : -(logit_spec[j] - logit_diff[i]));
      votes += y;
    }
    votes_sim[votes + 1] += 1;
  }
}
