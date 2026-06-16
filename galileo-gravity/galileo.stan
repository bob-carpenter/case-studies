
data {
  int<lower=1> N;                    // number of rolls
  vector<lower=0>[N] d;              // distance rolled along the incline (m)
  vector<lower=0>[N] w;              // water mass collected (g)  -- the data
  real<lower=0> r;                   // calibrated water flow rate (g / s)
  real<lower=0> kappa;               // rolling-inertia factor (7/5 for a solid sphere)
  real<lower=0, upper=1> sin_theta;  // sine of the incline angle
}
parameters {
  real<lower=0> g;                   // gravitational acceleration (m / s^2)
  real<lower=0> sigma;               // weighing noise (g)
}
model {
  // Forward model: predicted water mass at each distance.
  vector[N] mu = r * sqrt(2 * kappa * d / (g * sin_theta));

  // Weakly-informative priors (positivity from the <lower=0> declarations).
  g     ~ normal(10, 5);
  sigma ~ normal(0, 5);

  // Likelihood.
  w ~ normal(mu, sigma);
}
