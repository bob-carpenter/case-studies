import cmdstanpy as csp
import numpy as np
import scipy as sp
import pandas as pd
pd.set_option('display.max_rows', None)


df = pd.read_csv('../data/canonical/espeland-et-al/caries.csv',
                     comment = '#')
item = df['item'].to_list()
rater = df['coder'].to_list()
rating = df['response'].to_list()
I = int(np.max(item))
J = int(np.max(rater))
N = int(len(rater))
data = { 'I': I, 'J': J, 'N': N,
            'item': item, 'rater': rater, 'rating': rating }
init = { 'logit_prev': -2.0,
         'logit_sens': np.full(J, 2.0),
         'logit_spec': np.full(J, 2.0),
         'logit_diff': np.full(I, 0.0) }
model = csp.CmdStanModel(stan_file = '../stan/irt1pl.stan')
sample = model.sample(data = data, show_console = True, refresh = 1,
                          iter_warmup=200, iter_sampling=200,
                          chains = 1, inits = init,
                          seed = 92584)
sample.summary()
