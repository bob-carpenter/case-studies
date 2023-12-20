import cmdstanpy as csp
import numpy as np
import pandas as pd

df = pd.read_csv('data/rossi.csv')
df = df.infer_objects()
df = df.sort_values(by=['arrest', 'week'], ascending=[False, False])
df = df.reset_index(drop=True)

K = 7                              # num covariates
N = (df['arrest'] == 1).sum()      # num uncensored obs
t = df['week'][range(N)]           # arrest time
x = df.iloc[:N, -K:].to_numpy()    # covariates

N_c = (df['arrest'] != 1).sum()    # num censored obs
t_c = df['week'].iloc[0]           # censoring time
x_c = df.iloc[N:, -K:].to_numpy()  # covariates

model = csp.CmdStanModel(stan_file = "cox-proportional-hazards.stan")
data = {'K': K, 'N': N, 't': t, 'x': x, 'N_c': N_c, 't_c': t_c, 'x_c': x_c}
fit = model.sample(data = data)
