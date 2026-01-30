import json
import numpy as np
import cmdstanpy as csp

data_json_path='simcross.json'
seed = 2026
chains = 4

m_fixed = csp.CmdStanModel(stan_file='crossed-fix.stan')
m_random = csp.CmdStanModel(stan_file='crossed-hier.stan')
opt_fit = m_fixed.optimize(
    data=data_json_path,
    seed=seed,
    inits=0.05,
    show_console=True,
    refresh=20
)
sigma_alpha = np.std(opt_fit.stan_variable('alpha'))
sigma_beta = np.std(opt_fit.stan_variable('beta'))
mle = opt_fit.optimized_params_dict
mle['sigma_alpha'] = sigma_alpha
mle['sigma_beta'] = sigma_beta

print(f"{sigma_alpha=}  {sigma_beta=}  {mle['mu']=}  {mle['sigma']=}")
mle_copies = [mle for _ in range(chains)]
fit = m_fixed.sample(
    data=data_json_path,
    chains=chains,
    iter_warmup=200, iter_sampling=200,
    max_treedepth=7,
    parallel_chains=chains,
    seed=seed,
    inits=mle_copies,
    show_console=True,
    show_progress=False,
    refresh=1
)
print(fit.summary())
mu_hat = np.mean(fit.stan_variables('mu'))
mu_sd = np.std(fit.stan_variables('mu'))
