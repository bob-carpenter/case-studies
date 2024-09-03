import cmdstanpy as csp
import numpy as np
import scipy as sp
import pandas as pd
pd.set_option('display.max_rows', None)


df = pd.read_csv('../data/canonical/espeland-et-al/caries.csv',
                     comment = '#')

# REVISED
sample.summary()
item = df['item'].to_list()
rater = df['coder'].to_list()
rating = df['response'].to_list()
I = int(np.max(item))
J = int(np.max(rater))
N = int(len(rater))
data = { 'I': I, 'J': J, 'N': N,
            'item': item, 'rater': rater, 'rating': rating }
init = { 'prev': 0.2,
         'sigma': 1,             
         'sens': np.full(J, 2),
         'spec': np.full(J, 2),
         'diff': np.full(I, 0.0) }
model = csp.CmdStanModel(stan_file = '../stan/revised-1pl.stan')
sample = model.sample(data = data, show_console = True, refresh = 10,
                          iter_warmup=1000, iter_sampling=1000,
                          parallel_chains = 4, adapt_delta=0.9,
                          chains = 4, inits = init,
                          seed = 837689)

# ACTUAL W DENTISTRY: [1880,1065,404,247,173,100]

# ORIGINAL
# y = df.pivot(index='item', columns='coder', values='response').to_numpy()
# I, J = np.shape(y)
# data = {'I': I, 'J': J, 'y': y}
# init = { 'pi': -1.5,
#          'beta': np.full(I, 0.0) }
# model = csp.CmdStanModel(stan_file = '../stan/original-1pl.stan')
# sample = model.sample(data = data, show_console = True, refresh = 1,
#                           iter_warmup=200, iter_sampling=200,
#                           chains = 2, inits = init,
#                           seed = 92584)


