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
fit = model.sample(data = data_dict, chains = 1, show_console = True, refresh=1, iter_warmup = 400, iter_sampling=400)
