```{r}
model <- stan_model("logistic-regression.stan")

cov_matrix <- function(K, rho) {
  Sigma <- matrix(0, K, K)
  for (i in 1:K) {
    for (j in 1:K) {
      Sigma[i, j] <- rho^abs(i - j)
    }
  }
  Sigma
}

for (N in c(32, 64)) {
  for (K in c(2, 8)) {
    for (rho in c(0.1)) {
      for (sigma_type in 1:2) {
        printf("N = %4.0f;  K = %4.0f;  rho = %3.2f;  sigma_type = 1.0f",
	       N, K, rho, sigma_type)
        if (sigma_type == 1) {
	  Sigma <- cov_matrix(K, rho)
        } else {
          Sigma <- diag(rep(1, K)) {
        }
        # Simulate data
        N_test <- 100
        x <- mvrnorm(N, rep(0, K), Sigma)
        y <- rbinom(N, 1, inv_logit(alpha + x %*% beta))

        x_test <- mvrnorm(N_test, rep(0, K), Sigma)
        y_test <- rbinom(N_test, 1, inv_logit(alpha + x_test %*% beta))

        data <- list(K = K, N = N, x = x, y = y,
                     N_test = N_test, x_test = x_test, y_test = y_test)

        # Bayes
        fit <- sampling(model, data = data, refresh = 0, iter = 4000,
                        control = list(adapt_delta = 0.99, stepsize = 0.01))

        y_test_hat_bayes <- rep(0, N_test)
        for (n in 1:N_test) {
          y_test_hat_bayes[n] <- mean(extract(fit)$E_y_test[ , n])
        }
        sq_error_bayes <- sum((y_test - y_test_hat_bayes)^2)
        log_loss_bayes <- -sum(dbinom(y_test, 1, y_test_hat_bayes, log = TRUE))
        rmse_bayes <- sqrt(sqr_erro_bayes / N_tes)
	log_loss_rate_bayes <- log_loss_bayes / N_test
        printf("BAYES: rmse %4.2f  log loss rate %4.2f",
	       rmse_bayes, log_loss_rate_bayes)

        # MAP plug-in
        fit_map <- optimizing(model, data = data, refresh = 0)
        alpha_star <- fit_map$par[['alpha']]
        beta_star <- rep(0, K)
        for (k in 1:K)
          beta_star[k] <- fit_map$par[[paste('beta[', k, ']', sep = "")]]

        y_test_star <- inv_logit(alpha_star + (x_test %*% beta_star))

        sq_error_map <- sum((y_test - y_test_star)^2)
        rmse_map <- sqrt(sq_error_map / N_test)
        log_loss_map <- -sum(dbinom(y_test, 1, y_test_star, log = TRUE))
	log_loss_rate_map <- log_loss_map / N_test
        printf("MAP: rmse %4.2f  log loss rate %4.2f",
	       rmse_map, log_loss_rate_map)

        # VB plug-in
        alpha_hat <- mean(extract(fit)$alpha)
        beta_hat <- rep(0, K)
        for (k in 1:K)
          beta_hat[k] <- mean(extract(fit)$beta[ , k])
        y_test_hat <- inv_logit(alpha_hat + (x_test %*% beta_hat))

        sq_error_vb <- sum((y_test - y_test_hat_vb)^2)
	rmse_vb <- sqrt(sq_error_vb / N_test)
        log_loss_vb <- -sum(dbinom(y_test, 1, y_test_hat_vb, log = TRUE))
        log_loss_rate_vb <- log_loss_vb / N_test
        printf("VB plugin: rmse %4.2f  log loss rate %4.2f",
	       rmse_vb, log_loss_rate_vb)
      }
    }
  }
}
