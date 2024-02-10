functions {
  array[] int vote_count(array[] int rating,
                         array[] int item,
                         array[] int rater,
                         int I, int J) {
    int N = size(rating);
    array[I] int count_by_item = rep_array(1, I);  // index 0:5 by 1:6
    for (n in 1:N) {
      count_by_item[item[n]] += rating[n];
    }
    array[J + 1] int count = rep_array(0, J + 1);
    for (i in 1:I) {
      count[count_by_item[i]] += 1;
    }
    return count;
  }
}
data {
  int<lower=1> I;
  int<lower=1> J;
  int<lower=1> N;
  array[N] int<lower=0, upper=1> rating; 
  array[N] int<lower=1, upper=I> item; 
  array[N] int<lower=1, upper=J> rater; 
}
transformed data {
  print("***** votes: ", vote_count(rating, item, rater, I, J));
}
parameters {
  real<lower=0, upper=1> prev;
  vector[J] spec;
  vector<lower=-spec>[J] sens;
  vector[I] diff;
}
model {
  prev ~ uniform(0, 1);
  sens ~ normal(0, 3);
  spec ~ normal(0, 3);
  diff ~ normal(0, 1);
  vector[I] lp_pos = rep_vector(log(prev), I);
  vector[I] lp_neg = rep_vector(log1m(prev), I);
  for (n in 1:N) {
    int i = item[n];
    int j = rater[n];
    int y = rating[n];
    lp_pos[i] += bernoulli_logit_lpmf(y | sens[j] - diff[i]);
    lp_neg[i] += bernoulli_logit_lpmf(y | -(spec[j] - diff[i]));
  }
  for (i in 1:I) {
    target += log_sum_exp(lp_pos[i], lp_neg[i]);
  }
}
generated quantities {
  array[J + 1] int<lower = 0> votes_sim;
  {
    array[N] int rating_sim;
    array[I] int z_sim;
    for (i in 1:I) {
      z_sim[i] = bernoulli_rng(prev);
    }
    for (n in 1:N) {
      int i = item[n];
      int j = rater[n];
      rating_sim[n]
          = bernoulli_logit_rng(z_sim[i] == 1
                                ? sens[j] - diff[i]
                                : -(spec[j] - diff[i]));
    }
    votes_sim = vote_count(rating_sim, item, rater, I, J);
  }
}
