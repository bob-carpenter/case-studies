import numpy as np
import scipy as sp
import cmdstanpy as csp
from scipy.special import expit

seed = 583883
D = 6
N = 1500
mu_x_OH = -1
mu_x_NY  = 1
B = 10_000
h = 0.04

rng = np.random.default_rng(seed)

# generate single parameter vectors
beta = rng.normal(loc=0.0, scale=1.0, size=D)
print(f"{beta=}")

# generate Ohio data from parameters
x_OH = rng.normal(loc=-1.0, scale=1.0, size=(N, D))
y_OH = rng.binomial(n=1, p = expit(x_OH @ beta))

# generate NY data from parameters
x_NY = rng.normal(loc=1.0, scale=1.0, size=(N, D))
y_NY = rng.binomial(n=1, p = expit(x_NY @ beta))
print(f"{np.mean(y_NY)=}")  # shouldn't be too extreme

# fit OH data with simple logistic regression
data_OH = {'N': N, 'D': D, 'x': x_OH, 'y': y_OH }
model_OH = csp.CmdStanModel(stan_file='flat-logistic.stan')
fit_OH = model_OH.sample(data=data_OH, chains=4, iter_sampling=B // 4, seed=seed)
print(fit_OH.summary())
beta_OH_draws = fit_OH.stan_variable('beta')

# fit NY data with empirical prior from posterior of OH data
data_NY = {'N': N, 'D': D, 'x': x_NY, 'y': y_NY,
               'h': h, 'B': B, 'beta0': beta_OH_draws }
model_NY = csp.CmdStanModel(stan_file='empirical-logistic.stan')
fit_NY = model_NY.sample(data=data_NY, chains=4, iter_warmup=500, iter_sampling=500, seed=seed)
print(fit_NY.summary())

