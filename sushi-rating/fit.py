#!/usr/bin/env python3

import numpy as np
import pandas as pd
import cmdstanpy as csp

df = pd.read_csv('sushi3-2016/sushi3b.5000.10.order',
                     delim_whitespace=True, header=None, skiprows=1)
df = df.drop(columns=[0, 1])
y = df.to_numpy()
y = y + 1   # data indexes items from 0
y = y.astype(int)
data_dict = {'S': 100, 'N': 5000, 'K': 10, 'y': y }

model = csp.CmdStanModel(stan_file = 'sushi.stan')
fit = model.sample(data = data_dict, chains = 4, parallel_chains=4, show_console = True,
                       refresh=20, iter_warmup = 500, iter_sampling=500)

df_items = pd.read_csv('sushi3-2016/sushi3.idata',
                           delim_whitespace=True, header=None)
item_names = df_items.iloc[:, 1].values
item_scores = [np.mean(fit.stan_variable('alpha')[:,i]) for i in range(100)]
df_out = pd.DataFrame({'type': item_names, 'score': item_scores})
df_out.sort_values(by='score', ascending=False, inplace=True)
print(df_out.to_string())
for n in range(100):
    print(f"{df_out.iloc[n, :]['type']:15s} {df_out.iloc[n, :]['score']:6.3f}")
