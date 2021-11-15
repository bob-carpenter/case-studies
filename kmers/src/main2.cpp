
#include "kmers/dirichlet-sampler.hpp"
#include <Eigen/Dense>
#include <iostream>

int main() {
  std::random_device dev;
  std::mt19937 rng(dev());

  uint64_t N = 20;
  Eigen::VectorXd alpha = Eigen::VectorXd::Ones(N);
  Eigen::VectorXd theta = kmers::dirichlet_rng(alpha, rng);
  std::vector<double> theta_sv(&theta[0], &theta[0] + N);
  std::vector<uint64_t> y = kmers::multinomial_rng(10000, theta_sv, rng);
  for (int n = 0; n < N; ++n) {
    std::cout << "theta[" << n << "] = " << theta[n]
	      << ";  y[" << n << "] = " << y[n]
	      << std::endl;
  }
}
  
