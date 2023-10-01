import numpy as np
import logging
import cmdstanpy as csp
csp.utils.get_logger().setLevel(logging.ERROR)


def rw_cov_matrix(K, rho):
    Sigma = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            Sigma[i, j] = rho ** abs(i - j)
    return Sigma

def predictors(N, K, rho):
    Sigma = rw_cov_matrix(K, rho)
    mu = np.zeros(K)
    return np.random.multivariate_normal(mu, Sigma, N)

def sq_error(u, v):
    return sum((u - v)**2)

def fit(model, data_dict):
    return model.sample(data = data_dict, show_progress = False, show_console = False)

K = 20
N = 1000
rho = 0.9

model_logistic = csp.CmdStanModel(stan_file = "logistic-regression.stan")
model_weighted_logistic = csp.CmdStanModel(stan_file = "weighted-logistic-regression.stan")
model_weighted_linear = csp.CmdStanModel(stan_file = "linear-regression.stan")

M = 10
for rep in range(M):
    x = predictors(N, K, rho)
    beta = np.random.normal(0, 1, K)

    E_log_odds = np.dot(x, beta)
    inv_logit = lambda u: 1 / (1 + np.exp(-u))
    E_y = inv_logit(E_log_odds)

    y_max = np.where(E_y > 0.5, 1, 0)
    y_random = np.random.binomial(n=1, p=E_y)
    y_probs = E_y
    y_weights = E_log_odds
    y_noisy_weights = E_log_odds + np.random.normal(0, 1, N)

    data_max = {'K': K, 'N': N, 'x': x, 'y': y_max }
    data_random = {'K': K, 'N': N, 'x': x, 'y': y_random }
    data_probs = {'K': K, 'N': N, 'x': x, 'y': y_probs }
    data_weights = {'K': K, 'N': N, 'x': x, 'y': y_weights }
    data_noisy_weights = {'K': K, 'N': N, 'x': x, 'y': y_noisy_weights }


    
    fit_max = fit(model_logistic, data_max)
    fit_random = fit(model_logistic, data_random)
    fit_probs = fit(model_weighted_logistic, data_probs)
    fit_weights = fit(model_weighted_linear, data_weights)
    fit_noisy_weights = fit(model_weighted_linear, data_noisy_weights)

    beta_draws_max = fit_max.stan_variable("beta")
    beta_draws_random = fit_random.stan_variable("beta")
    beta_draws_probs = fit_probs.stan_variable("beta")
    beta_draws_weights = fit_weights.stan_variable("beta")
    beta_draws_noisy_weights = fit_noisy_weights.stan_variable("beta")


    mean_max = np.zeros(K)
    mean_random = np.zeros(K)
    mean_probs = np.zeros(K)
    mean_weights = np.zeros(K)
    mean_noisy = np.zeros(K)
    for k in range(K):
        mean_max[k] = np.mean(beta_draws_max[:, k])
        mean_random[k] = np.mean(beta_draws_random[:, k])
        mean_probs[k] = np.mean(beta_draws_probs[:, k])
        mean_weights[k] = np.mean(beta_draws_weights[:, k])
        mean_noisy[k] = np.mean(beta_draws_noisy_weights[:, k])
        sq_error_max = sq_error(mean_max, beta)
        sq_error_random = sq_error(mean_random, beta)
        sq_error_probs = sq_error(mean_probs, beta)
        sq_error_weights = sq_error(mean_weights, beta)
        sq_error_noisy = sq_error(mean_noisy, beta)

    print(f"max: {sq_error_max:5.2f}  random: {sq_error_random:5.2f}  probs:{sq_error_probs:5.2f} weights:{sq_error_weights:5.2f} noisy:{sq_error_noisy:5.2f}")
