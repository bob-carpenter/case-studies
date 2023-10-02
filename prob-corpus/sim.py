import numpy as np
import logging
import cmdstanpy as csp
csp.utils.get_logger().setLevel(logging.ERROR)


def rw_cov_matrix(D, rho):
    Sigma = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            Sigma[i, j] = rho ** abs(i - j)
    return Sigma

def predictors(N, D, rho):
    Sigma = rw_cov_matrix(D, rho)
    mu = np.zeros(D)
    x = np.random.multivariate_normal(mu, Sigma, N)
    print(f"{np.shape(x) = }")
    for n in range(N):
        x[n, 1] = 1.0  # intercept
    return x        

def sq_error(u, v):
    return sum((u - v)**2)

def fit_bayes(model, data_dict):
    s = model.sample(data = data_dict, show_progress = False, show_console = False)
    return s.stan_variable("beta")

def fit_mle(model, data_dict):
    mle = model.optimize(data = data_dict, show_console = False)
    return mle.stan_variable("beta")

def inv_logit(x):
    return 1 / (1 + exp(-x))

D = 20
N = 500
rho = 0.9

model_logistic = csp.CmdStanModel(stan_file = "logistic-regression.stan")
model_weighted_logistic = csp.CmdStanModel(stan_file = "weighted-logistic-regression.stan")
model_weighted_linear = csp.CmdStanModel(stan_file = "log-odds-linear-regression.stan")

M = 10
for rep in range(M):
    x = predictors(N, D, rho)
    beta = np.random.normal(0, 1, D)

    E_log_odds = np.dot(x, beta)
    inv_logit = lambda u: 1 / (1 + np.exp(-u))
    E_y = inv_logit(E_log_odds)

    y_max = np.where(E_y > 0.5, 1, 0)
    y_random = np.random.binomial(n=1, p=E_y)
    p = E_y
    y_noisy_log_odds = E_log_odds + np.random.normal(0, 1, N)
    noisy_p = inv_logit(y_noisy_log_odds)
    
    data_max = {'D': D, 'N': N, 'x': x, 'y': y_max }
    data_random = {'D': D, 'N': N, 'x': x, 'y': y_random }
    data_probs = {'D': D, 'N': N, 'x': x, 'p': p }
    data_weights = {'D': D, 'N': N, 'x': x, 'p': p }
    data_noisy_weights = {'D': D, 'N': N, 'x': x, 'p': noisy_p}

    mle_max = fit_mle(model_logistic, data_max)
    mle_random = fit_mle(model_logistic, data_random)
    mle_probs = fit_mle(model_weighted_logistic, data_probs)
    mle_weights = fit_mle(model_weighted_linear, data_weights)
    mle_noisy = fit_mle(model_weighted_linear, data_noisy_weights)

    beta_draws_max = fit_bayes(model_logistic, data_max)
    beta_draws_random = fit_bayes(model_logistic, data_random)
    beta_draws_probs = fit_bayes(model_weighted_logistic, data_probs)
    beta_draws_weights = fit_bayes(model_weighted_linear, data_weights)
    beta_draws_noisy_weights = fit_bayes(model_weighted_linear, data_noisy_weights)

    mean_max = np.zeros(D)
    mean_random = np.zeros(D)
    mean_probs = np.zeros(D)
    mean_weights = np.zeros(D)
    mean_noisy = np.zeros(D)
    for d in range(D):
        mean_max[d] = np.mean(beta_draws_max[:, d])
        mean_random[d] = np.mean(beta_draws_random[:, d])
        mean_probs[d] = np.mean(beta_draws_probs[:, d])
        mean_weights[d] = np.mean(beta_draws_weights[:, d])
        mean_noisy[d] = np.mean(beta_draws_noisy_weights[:, d])

    sq_error_max = sq_error(mean_max, beta)
    sq_error_random = sq_error(mean_random, beta)
    sq_error_probs = sq_error(mean_probs, beta)
    sq_error_weights = sq_error(mean_weights, beta)
    sq_error_noisy = sq_error(mean_noisy, beta)

    sq_error_mle_max = sq_error(mle_max, beta)
    sq_error_mle_random = sq_error(mle_random, beta)
    sq_error_mle_probs = sq_error(mle_probs, beta)
    sq_error_mle_weights = sq_error(mle_weights, beta)
    sq_error_mle_noisy = sq_error(mle_noisy, beta)

    mean_p = np.mean(p)

    print(f"\nMEAN Pr[Y_n = 1] = {np.mean(p):5.2f}")
    print(f"BAYES: max: {sq_error_max:5.2f}  random: {sq_error_random:5.2f}  probs:{sq_error_probs:5.2f} weights:{sq_error_weights:5.2f} noisy:{sq_error_noisy:5.2f}")

    print(f"MLE:   max: {sq_error_mle_max:5.2f}  random: {sq_error_mle_random:5.2f}  probs:{sq_error_mle_probs:5.2f} weights:{sq_error_mle_weights:5.2f} noisy:{sq_error_mle_noisy:5.2f}")
