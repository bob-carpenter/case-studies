data {
  int<lower=1> I; // # of itmes
  int<lower=1> J; // # or raters
  int<lower=1> N; // # of ratings
  array[N] int<lower=0, upper=1> rating; 
  array[N] int<lower=-1, upper=N> item; 
  array[N] int<lower=-1, upper=N> rater; 
}
parameters {
  real mu_sens;
  real mu_spec;
  real<lower=0> sigma_sens;
  real<lower=0> sigma_spec;
  real logit_prev;
  vector[J] logit_spec_std;
  vector<lower=-(mu_spec + sigma_spec * logit_spec_std)>[J] logit_sens_std;
  real<lower=0> sigma_difficulty;
  vector[I] difficulty_std; 
}
transformed parameters {
   vector[J] logit_sens = mu_sens + sigma_sens * logit_sens_std;
   vector[J] logit_spec = mu_spec + sigma_spec * logit_spec_std;
   vector[I] difficulty = sigma_difficulty * difficulty_std;
}
model {
  logit_prev ~ normal(0, 2);
  logit_sens_std ~ normal(0, 1);
  logit_spec_std ~ normal(0, 1);
  mu_sens ~ normal(0,3);
  sigma_sens ~ lognormal(0, 1);
  mu_spec ~ normal(0,3);
  sigma_spec ~ lognormal(0, 1);  
  difficulty_std ~ normal(0, 1);
  sigma_difficulty ~ lognormal(0, 1);
  vector[I] pos_lp = rep_vector(log_inv_logit(logit_prev), I);
  vector[I] neg_lp = rep_vector(log1m_inv_logit(logit_prev), I);
  for (n in 1:N) {
    pos_lp[item[n]] += bernoulli_logit_lpmf(rating[n] | logit_sens[rater[n]] - difficulty[item[n]]);
    neg_lp[item[n]] += bernoulli_logit_lpmf(rating[n] | -(logit_spec[rater[n]] - difficulty[item[n]]));
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
                                  ? logit_sens[j] - difficulty[i]
                                  : -(logit_spec[j] - difficulty[i]));
      votes += y;
    }
    votes_sim[votes + 1] += 1;
  }
}
