library(rstan)
options(mc.cores = parallel::detectCores(logical = FALSE))
rstan_options(auto_write = TRUE)

options('width' = 120)

dat <- read.csv('../data/canonical/espeland-et-al/caries.csv',
                header = TRUE, comment.char = '#')
N <- dim(dat)[1]
I <- max(dat$item)
J <- max(dat$coder)
ii <- dat$item
jj <- dat$coder
y <- dat$response

data <- list(N = N, I = I, J = J, ii = ii, jj = jj, y = y)

# model_voting <- stan_model('voting.stan')
# fit_voting <- sampling(model_voting, data = data, # init = init_fun,
#                        chains = 4, iter = 800, seed = 12345)

# model_irt_0pl <- stan_model('irt-0pl.stan')
# fit_irt_0pl <- sampling(model_irt_0pl, data = data, # init = init_fun,
#                        chains = 4, iter = 800, seed = 12345)

model_dawid_skene <- stan_model('dawid-skene.stan')
fit_dawid_skene <- sampling(model_dawid_skene, data = data, # init = init_fun,
                            chains = 4, iter = 800, seed = 12345)
