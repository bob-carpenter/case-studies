import numpy as np
import logging
import cmdstanpy as csp
csp.utils.get_logger().setLevel(logging.ERROR)
import pandas as pd

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

def add_row(df, beta_hat, beta, estimator, data):
    return pd.concat([df, pd.DataFrame({'error': (sq_error(beta_hat, beta), ),
                                            'estimator': (estimator, ),
                                            'data': (data, )})],
                         ignore_index=True)

D = 21
N = 100
rho = 0.9

model_logistic = csp.CmdStanModel(stan_file = "logistic-regression.stan")
model_weighted_logistic = csp.CmdStanModel(stan_file = "weighted-logistic-regression.stan")
model_weighted_linear = csp.CmdStanModel(stan_file = "log-odds-linear-regression.stan")

M = 3
errs_max = np.zeros(M)
errs_random = np.zeros(M)
errs_probs = np.zeros(M)
errs_weights = np.zeros(M)
errs_noisy = np.zeros(M)
errs_max_mle = np.zeros(M)
errs_random_mle = np.zeros(M)
errs_probs_mle = np.zeros(M)
errs_weights_mle = np.zeros(M)
errs_noisy_mle = np.zeros(M)

df = pd.DataFrame({'error': (), 'estimator': (), 'data': ()})
for m in range(M):
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

    df = add_row(df, mean_max, beta, "Bayes", "max")        
    df = add_row(df, mean_random, beta, "Bayes", "max")        
    df = add_row(df, mean_probs, beta, "Bayes", "max")        
    df = add_row(df, mean_weights, beta, "Bayes", "max")        
    df = add_row(df, mean_noisy, beta, "Bayes", "max")        


    df = add_row(df, mle_max, beta, "MLE", "max")        
    df = add_row(df, mle_random, beta, "MLE", "max")        
    df = add_row(df, mle_probs, beta, "MLE", "max")        
    df = add_row(df, mle_weights, beta, "MLE", "max")        
    df = add_row(df, mle_noisy, beta, "MLE", "max")        

print(df)
