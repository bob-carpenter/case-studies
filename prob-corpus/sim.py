import logging
import numpy as np
import pandas as pd
import plotnine as pn
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
    for n in range(N):
        x[n, 1] = 1.0  # intercept
    return x        

def sq_error(u, v):
    return sum((u - v)**2)

def fit_bayes(model, data_dict):
    s = model.sample(data = data_dict, show_progress = False, show_console = False, seed=12345)
    return s.stan_variable("beta")

def fit_mle(model, data_dict):
    mle = model.optimize(data = data_dict, show_console = False, seed=12345)
    return mle.stan_variable("beta")

def inv_logit(x):
    return 1 / (1 + exp(-x))

def add_row(df, beta_hat, beta, estimator, data):
    return pd.concat([df, pd.DataFrame({'error': (sq_error(beta_hat, beta), ),
                                            'estimator': (estimator, ),
                                            'data': (data, )})],
                         ignore_index=True)

model_logistic = csp.CmdStanModel(stan_file = "logistic-regression.stan")
model_weighted_logistic = csp.CmdStanModel(stan_file = "weighted-logistic-regression.stan")
model_log_odds_linear = csp.CmdStanModel(stan_file = "log-odds-linear-regression.stan")

D = 21           # number of predictors including intercept
N = 200          # number of data points used to train
rho = 0.9        # correlation of predictor RW covariance
M = 10           # number of simulations
df = pd.DataFrame({'error': (), 'estimator': (), 'data': ()})
for m in range(M):
    print(f"Iteration: m = {m + 1} / {M}")

    # Data generation
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

    # Penalized MLE
    mle_max = fit_mle(model_logistic, data_max)
    mle_random = fit_mle(model_logistic, data_random)
    mle_probs = fit_mle(model_weighted_logistic, data_probs)
    mle_weights = fit_mle(model_log_odds_linear, data_weights)
    mle_noisy = fit_mle(model_log_odds_linear, data_noisy_weights)
    df = add_row(df, mle_max, beta, "MLE", "max")        
    df = add_row(df, mle_random, beta, "MLE", "random")        
    df = add_row(df, mle_probs, beta, "MLE", "probs")        
    df = add_row(df, mle_weights, beta, "MLE", "weights")        
    df = add_row(df, mle_noisy, beta, "MLE", "noisy")        

    # Bayesian
    beta_draws_max = fit_bayes(model_logistic, data_max)
    beta_draws_random = fit_bayes(model_logistic, data_random)
    beta_draws_probs = fit_bayes(model_weighted_logistic, data_probs)
    beta_draws_weights = fit_bayes(model_log_odds_linear, data_weights)
    beta_draws_noisy_weights = fit_bayes(model_log_odds_linear, data_noisy_weights)
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
    df = add_row(df, mean_random, beta, "Bayes", "random")        
    df = add_row(df, mean_probs, beta, "Bayes", "probs")        
    df = add_row(df, mean_weights, beta, "Bayes", "weights")        
    df = add_row(df, mean_noisy, beta, "Bayes", "noisy")        

    # Console feedback on each loop
    print(f"MLE:    max: {sq_error(mle_max, beta):5.2f} random: {sq_error(mle_random, beta):5.2f} probs: {sq_error(mle_probs, beta):5.2f} weights: {sq_error(mle_weights, beta):5.2f} noisy: {sq_error(mle_noisy, beta):5.2f}")
    print(f"Bayes:  max: {sq_error(mean_max, beta):5.2f} random: {sq_error(mean_random, beta):5.2f} probs: {sq_error(mean_probs, beta):5.2f} weights: {sq_error(mean_weights, beta):5.2f} noisy: {sq_error(mean_noisy, beta):5.2f}")
    print("\n")
    
plot = (pn.ggplot(df, pn.aes(x = 'error'))
    + pn.geom_histogram()
    + pn.facet_grid('estimator ~ data')
    + pn.scale_x_log10())

def custom_summarize(x):
    return pd.Series({
        'mean': np.mean(x),
        'min': np.min(x),
        'max': np.max(x),
        '0.1 quantile': np.quantile(x, 0.1),
        '0.5 quantile': np.quantile(x, 0.5),
        '0.9 quantile': np.quantile(x, 0.9)
    })

summary = df.groupby(['estimator', 'data'])['error'].apply(custom_agg).unstack()

print(summary.round(1))
