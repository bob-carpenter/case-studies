import json
import numpy as np
import cmdstanpy as csp

def mle(data_path, model_path, seed, inits=0.05, refresh=20):
    m = csp.CmdStanModel(stan_file=model_fixed_path)
    fit = m.optimize(
        data=data_path,
        seed=seed,
        inits=inits,
        show_console=True,
        refresh=refresh)
    mle = fit.optimized_params_dict
    sigma_alpha_hat = np.std(fit.stan_variable('alpha'))
    sigma_beta_hat = np.std(fit.stan_variable('beta'))
    mle['sigma_alpha'] = sigma_alpha_hat
    mle['sigma_beta'] = sigma_beta_hat
    return mle

def posterior(data_path, model_path, seed, mle,
                  chains=4, iter_warmup=200, iter_sampling=200, max_treedepth=5):
    m = csp.CmdStanModel(stan_file=model_path)
    inits = [mle for _ in range(chains)]
    fit = m.sample(
        data=data_path,
        chains=chains,
        iter_warmup=iter_warmup, iter_sampling=iter_sampling,
        parallel_chains=chains,
        max_treedepth=max_treedepth,
        seed=seed,
        inits=inits,
        show_console=True,
        show_progress=False,
        refresh=1
        )
    return fit
    

def summarize_posterior(fit, vars):
    print("Posterior (hierarchical)")
    for v in vars:
        draws = fit.stan_variable(v)
        mean = draws.mean()
        std = draws.std()
        print(f"    {v}:  {mean:7.5f}({std:7.5f})")

def summarize_mle(mle, vars):
    print("Maximum likelihood estimates")
    for v in vars:
        print(f"    {v}:  {mle[v]:6.4f}")


model_fixed_path='crossed-fix.stan'
model_hier_path='crossed-hier.stan'
data_json_path='simcross.json'
seed = 2026
qoi =  ['mu', 'sigma', 'sigma_alpha', 'sigma_beta']

print("Step 1.  Optimize to find the MLE")
mle = mle(data_json_path, model_fixed_path, seed)

print("Step 2.  Sample from the posterior.")
posterior = posterior(data_json_path, model_hier_path, seed, mle)

print("RESULTS==============================================")
summarize_mle(mle, qoi)
summarize_posterior(posterior, qoi)

