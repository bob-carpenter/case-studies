#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <random>
#include <chrono>


typedef Eigen::Triplet<float, uint32_t> triplet_t;
typedef std::vector<triplet_t> triplets_t;
typedef Eigen::SparseMatrix<float, Eigen::RowMajor> sparse_mat_t;
typedef Eigen::VectorXf vec_t;

int main() {
  int rows = 100 * 1000;
  int cols = 1000 * 1000;
  int nonzeros_per_row = 6000;
  int num_nonzeros = rows * nonzeros_per_row;

  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<float> col_dist(0, cols - 1);
  std::uniform_real_distribution<float> val_dist(0, 99);

  triplets_t trips;
  trips.reserve(num_nonzeros);
  for (int i = 0; i < rows; ++i) {
    if (i % 10000 == 0) std::cout << "row = " << i << std::endl;
    for (int n = 0; n < nonzeros_per_row; ++n) {
      int j = col_dist(gen);
      int val = val_dist(gen);
      trips.emplace_back(i, j, val);
    }
  }

  std::cout << "setting from triplets, with non-zero entries = " << trips.size() << std::endl;
  
  sparse_mat_t x(rows, cols);
  x.setFromTriplets(trips.begin(), trips.end());

  std::cout << "x.size() = " << x.size() << std::endl;

  vec_t beta(cols);
  for (int i = 0; i < cols; ++i) {
    beta(i) = val_dist(gen);
  }

  std::cout << "start multiply" << std::endl;
  auto t1 = std::chrono::high_resolution_clock::now();
  vec_t mu = x * beta;
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "done multiplying" << std::endl;
  std::chrono::duration<double, std::milli> dur = t2 - t1;
  std::cout << "multiply time = " << dur.count() << "ms" << std::endl;

  std::cout << "mu = " << mu(100) << ", " << mu(1000) << std::endl;
  
  std::cout << "Fini." << std::endl;
}
