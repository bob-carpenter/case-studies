printf <- function(msg, ...) cat(sprintf(msg, ...), "\n")

to_posterior <- function(model, data) {
  sampling(model, data = data, chains = 1, iter = 1, refresh = 0,
           algorithm = "Fixed_param")
}

# Returns optimization path for initialization x_init using objective
# function fn with gradient function gr, for a total of N iterations.
# Uses the L-BFGS-B algorithm of Nocedal.  Each row in the returned
# matrix is an iteration, and there are N + 1 rows because the
# initialization is included.
#
# For N iterations, the algorithm makes N calls to the base
# R function optim(), with max iteration set to n for each n in 1:N.
# This allows programmatic access to the iterations of optim(), but is
# O(N^2). Thanks to Calvin Whealton for the approach:
#   https://stackoverflow.com/a/46904780).
#
# @param x_init initial parameter values
# @param fn objective function
# @param gr gradient function of objective
# @param N total number of iterations (default 25)
# @return optimization path, matrix with N rows, each of which is a
#         point on the optimization path followed by objective
opt_path <- function(x_init, fn, gr, N = 25) {
  D <- length(x_init)
  y <- matrix(NA, nrow = N + 1, ncol = D + 1)
  y[1, 1:D] <- x_init
  y[1, D + 1] <- fn(x_init)
  for (n in 1:N) {
    z <- optim(par = x_init,
               fn = function(x) -fn(x),  # negate for maximization
	       gr = function(x) -gr(x),
               method = "L-BFGS-B",
               control = list(maxit = n))
    y[n + 1, 1:D] <- z$par
    y[n + 1, D + 1] <- fn(z$par)
  }
  y
}

# Returns optimization path for specified Stan model and data for
# the specified number of iterations using the specified bound on
# uniform initialization on the unconstrained scale.  See opt_path()
# for a description of output format and algorithm.
#
# @param model Stan model (compiled using rstan::stan_model())
# @param data data list for call to rstan::sampling for specified model
# @param N total number of iterations (default = 25)
# @param init_bound upper bound on uniform initialization; negation is lower
# @return optimization path (see opt_path documentation)
opt_path_stan <- function(model, data, N = 25, init_bound = 2) {
  # require chains > 0, iter > 0, so use Fixed_param to avoid work
  posterior <- to_posterior(model, data)
  D <- get_num_upars(posterior)
  init <- runif(D, -init_bound, init_bound)
  fn <- function(theta) log_prob(posterior, theta)
  gr <- function(theta) grad_log_prob(posterior, theta)
  opt_path(init, fn, gr, N)
}

# Return optimization path with last column (objective function value)
# removed.
#
# @param path optimization path with last column for objective
#        function value
# @return
params_only <- function(path) {
  N <- dim(path)[1]
  D <- dim(path)[2]
  path[1:N, 1:(D - 1)]
}

lp_draws <- function(model, data, init_param_unc) {
  iter <- 2
  max_treedepth <- 3
  stepsize <- 0.01
  posterior <- to_posterior(model, data)
  init_fun <- function(chain_id) constrain_pars(posterior, init_param_unc)
  fit <- sampling(model, data = data, init = init_fun,
                  chains = 1, iter = iter, warmup = 0, refresh = 0,
		  control = list(metric = "unit_e",
		                 adapt_engaged = FALSE,
				 max_treedepth = max_treedepth,
				 stepsize = stepsize))
  draws <- extract(fit, pars = c("lp__"), permute = FALSE)
  draws[1:iter]
}

increased <- function(lps) {
  # simple comparison of endpoint rather than whole path
  lps[length(lps)] > lps[1]
}

is_typical <- function(model, data, param) {
  M <- 100
  increase_count <- 0
  for (m in 1:M) {
    lps <- lp_draws(model, data, param)
    increase_count <- increase_count + increased(lps)
  }
  increase_count / M
}

find_typical <- function(model, data, param_path, M = 100) {
  N <- dim(param_path)[1]
  D <- dim(param_path)[2] - 1   # includes objective in last position
  for (n in 1:N) {
    increase_prop <- is_typical(model, data, param_path[n, 1:D])
    printf("n = %3d;  increase proportion = %3.2f",
           n, increase_prop)
    # declare typical if in central 90% interval of random increase/decrease
    lb = qbinom(0.05, M, 0.5) / M
    ub = qbinom(0.95, M, 0.5) / M
    if (increase_prop >= lb && increase_prop <= ub)
      print(param_path[n, 1:(D + 1)], digits = 2)
  }
}

library('rstan')
program <-
"
data {
  int D;
} parameters {
  vector[D] theta;
  real<lower = 0> sigma;
} model {
  theta ~ normal(0, sigma);
  sigma ~ normal(0, 1);
}
"
model <- stan_model(model_code = program)
data = list(D = 8)
opath <- opt_path_stan(model, data = data, N = 50, init_bound = 5)
find_typical(model, data, opath)
