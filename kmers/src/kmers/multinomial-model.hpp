#ifndef MULTINOMIAL_MODEL_HPP
#define MULTINOMIAL_MODEL_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <stdexcept>
#include <vector>

/**
 * Return the softmax of the specified vector.  Softmax is defined by
 *
 * ```
 * softmax(alpha) = exp(alpha) / sum(exp(alpha))
 * ```
 *
 * using offsets to prevent underflow of the exponentiation.
 *
 * @param alpha unconstrained input vector
 * @return softmax of input
 */
Eigen::VectorXf softmax(const Eigen::VectorXf& alpha) {
  using std::exp;
  auto delta = Eigen::VectorXf::Constant(alpha.size(), alpha.maxCoeff());
  Eigen::VectorXf phi = (alpha - delta).array().exp();
  return phi / phi.sum();
}

/**
 * This class defines a Bayesian model implementing the joint density
 *
 * ```
 * log p(y, alpha | xt)
 *   = log multinomial(y | xt * softmax(alpha)) + log normal(alpha | 0, 3)
 *   = y' * log(xt * softmax(alpha)) - 1 / (2 * 3^2) * alpha' * alpha
 * ```
 *
 * K: kmer size
 * M: number of k-mers (4^K)
 * N: number of reads
 * T: number of isoforms
 * y: (M x 1) matrix of shredded reads
 * x: (T x M) sparse matrix of kmers per isoform with simplex rows
 * xt: (M x T) sparse matrix of kmers per isoform with simplex columns
 * alpha: (T x 1) vector of log odds
 */
struct multinomial_model {
  const Eigen::Map<Eigen::SparseMatrix<float, Eigen::RowMajor>>& xt_;
  const Eigen::VectorXf& y_;

  multinomial_model(
      const Eigen::Map<Eigen::SparseMatrix<float, Eigen::RowMajor>>& xt,
      const Eigen::VectorXf& y)
      : xt_(xt), y_(y) {
    if (xt.rows() != y.rows()) {
       throw std::runtime_error("xt rows must equal y cols");
    }
  }

  float log_density(const Eigen::VectorXf& beta) {
    Eigen::VectorXf theta = softmax(beta);

    std::cout << "model:  xt_.rows() = " << xt_.rows() << std::endl;
    std::cout << "model:  xt_.cols() = " << xt_.cols() << std::endl;
    std::cout << "model:  theta.rows() = " << theta.rows() << std::endl;
    std::cout << "model:  theta.cols() = " << theta.cols()
              << std::endl;
    std::cout << "model:  theta.sum() = " << theta.sum() << std::endl;

    Eigen::VectorXf xt_theta =  1e-6 * Eigen::VectorXf::Ones(xt_.rows())
        + (1 - 1e-6) * (xt_ * theta);

    std::cout << "model:: xt_theta.sum() = " << xt_theta.sum() << std::endl;

    Eigen::VectorXf log_xt_theta = xt_theta.array().log();
    for (int i = 0; i < log_xt_theta.size(); ++i)
      if (!(log_xt_theta(i) < 0))
        std::cout << "UH log_xt_theta(" << i << ") = " << log_xt_theta(i)
                  << std::endl;

    std::cout << "model:  log_xt_theta.rows() = " << log_xt_theta.rows()
              << std::endl;
    std::cout << "model:  log_xt_theta.cols() = " << log_xt_theta.cols()
              << std::endl;

    float log_likelihood = y_.transpose() * log_xt_theta;
    float log_prior = -0.125 * beta.transpose() * beta;
    return log_likelihood + log_prior;
  }

  void grad_log_density(const Eigen::VectorXf& beta, Eigen::VectorXf& grad) {
    Eigen::VectorXf t1 = softmax(beta);
    Eigen::VectorXf t2 = xt_ * t1;
    Eigen::VectorXf t3 = (y_.cwiseProduct(t2.cwiseInverse()).transpose() * xt_).transpose();
    Eigen::VectorXf grad_likelihood = t1.cwiseProduct(t3) - t1.dot(t3) * t1;
    Eigen::VectorXf grad_prior = -0.25 * beta;
    grad = grad_likelihood + grad_prior;
  }

  //   std::vector<int> sample(uint64_t N, const Eigen::VectorXf& beta) {
  // return std::multinomial_rng(N, xt_ * softmax(beta));
  // }

  // std::vector<int> sample(uint64_t N, double mu, double sigma) {
  // return std::vector<int>();
  // }
};

#endif
