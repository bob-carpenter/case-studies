#ifndef MULTINOMIAL_MODEL_HPP
#define MULTINOMIAL_MODEL_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>

// typedef Eigen::SparseMatrix<float, Eigen::RowMajor> sparse_matrix_t;

Eigen::VectorXf softmax(const Eigen::VectorXf& alpha) {
  using std::exp;
  auto delta = Eigen::VectorXf::Constant(alpha.maxCoeff(), alpha.size());
  Eigen::VectorXf phi = (alpha - delta).array().exp();
  return delta + phi / phi.sum();
}

/**
 * Model is
 * 
 * p(y, alpha | xt)
 *   = multinomial(y | xt * softmax(alpha)) * normal(alpha | 0, 3)
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
  const Eigen::VectorXf& y_;
  const Eigen::SparseMatrix<float, Eigen::RowMajor>& xt_;

  multinomial_model(const Eigen::VectorXf& y,
		    const Eigen::SparseMatrix<float, Eigen::RowMajor>& xt)
    : y_(y), xt_(xt) { }

  float log_density(const Eigen::VectorXf& alpha) {
    Eigen::VectorXf theta = softmax(alpha);
    Eigen::VectorXf xt_sm_a =  (xt_ * softmax(alpha)).array().log();
    float log_likelihood = y_.transpose() * xt_sm_a;
    float log_prior = -0.125 * alpha.dot(alpha);
    return log_likelihood + log_prior;
  }

  void grad_log_density(const Eigen::VectorXf& alpha, Eigen::VectorXf& grad) {
    Eigen::VectorXf t1 = softmax(alpha);
    auto t2 = xt_ * t1;
    Eigen::VectorXf t3 = (y_.cwiseProduct(t2.cwiseInverse()).transpose() * xt_).transpose();
    auto grad_likelihood = t1.cwiseProduct(t3) - t1.dot(t3) * t1;
    auto grad_prior = -0.25 * alpha;
    grad = grad_likelihood + grad_prior;
  }

};

int main() {
  return 0;
}

#endif
