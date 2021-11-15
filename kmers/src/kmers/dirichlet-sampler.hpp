#ifndef DIRICHLET_SAMPLER_HPP
#define DIRICHLET_SAMPLER_HPP

#include <Eigen/Dense>
#include <random>
#include <iostream>

namespace kmers {
  
/**
 * Return Dirichlet random variate with specified positive vector of
 * values.  
 */
template <class RNG>
Eigen::VectorXd dirichlet_rng(const Eigen::VectorXd& alpha, RNG& rng) {
  int N = alpha.size();
  Eigen::VectorXd theta(N);
  for (int n = 0; n < N; ++n) {
    std::gamma_distribution<double> gamma_d(alpha(n), 1);
    theta(n) = gamma_d(rng);
  }
  return theta / theta.array().sum();
}

template <class RNG>
Eigen::VectorXd
normal_rng(uint64_t N, double mu, double sigma, RNG& rng) {
  std::normal_distribution<double> normal_d(mu, sigma);
  Eigen::VectorXd y;
  for (int n = 0; n < N; ++n) {
    y(n) = normal_d(rng);
  }
  return y;
}

template <class RNG>
std::vector<uint64_t> multinomial_rng(uint64_t N, const std::vector<double>& theta, RNG& rng) {
  std::vector<uint64_t> y(N, 0);
  std::discrete_distribution<uint64_t> discrete_d(theta.begin(), theta.end());
  for (uint64_t n = 0; n < N; ++n) {
    uint64_t z = discrete_d(rng);
    ++y[z];
  }
  return y;
}
}

#endif
