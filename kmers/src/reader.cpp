#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "kmers/multinomial-model.hpp"

template <typename T>
T read_val(std::istream& in) {
  T x;
  in.read(reinterpret_cast<char*>(&x), sizeof(T));
  return x;
}
template <typename T>
std::vector<T> read_array(std::istream& in, int32_t N) {
  std::vector<T> y(N);
  for (int32_t n = 0; n < N; ++n)
    y[n] = read_val<T>(in);
  return y;
}

Eigen::Map<Eigen::SparseMatrix<float, Eigen::RowMajor>>
read_xt(const std::string& filename) {
  std::fstream in(filename, std::ios::binary | std::ios::in);
  if (!in) {
    std::string msg = "reader: couldn't open file = " + filename;
    throw std::runtime_error(msg);
  }
  int32_t rows = read_val<int>(in);
  std::cout << "reader:  rows = " << rows
            << std::endl;

  int32_t cols = read_val<int>(in);
  std::cout << "reader:  cols = " << cols
            << std::endl;

  int32_t nnz = read_val<int>(in);
  std::cout << "reader:  nnz = " << nnz
            << std::endl;

  std::vector<int> outerIndexPtr = read_array<int>(in, rows + 1);
  for (size_t i = 0; i < outerIndexPtr.size(); ++i) {
    if (outerIndexPtr[i] > nnz || outerIndexPtr[i] < 0) {
      throw std::runtime_error("OOPS: outerIndexPtr out of range");
    }
  }
  std::cout << "reader:  read outerIndexPtr"
            << std::endl;

  std::vector<int> innerIndices = read_array<int>(in, nnz);
  for (size_t i = 0; i < innerIndices.size(); ++i) {
    if (innerIndices[i] < 0 || innerIndices[i] >= cols) {
      throw std::runtime_error("innerIndices exceeded cols");
    }
  }
  std::cout << "reader:  read innerIndices"
            << std::endl;

  std::vector<float> values = read_array<float>(in, nnz);
  std::cout << "reader:  read values"
            << std::endl;

  in.close();
  std::cout << "reader:  finished reading, closed file stream"
            << std::endl;

  std::cout << "reader:  building Eigen::Map"
            << std::endl;


  return Eigen::Map<Eigen::SparseMatrix<float, Eigen::RowMajor>>(
      rows, cols, nnz, outerIndexPtr.data(), innerIndices.data(),
      values.data());
}



int main(int argc, char* argv[]) {
  std::string filename = argv[1];
  std::cout << "reader: reading from filename = " << filename
            << std::endl;

  Eigen::Map<Eigen::SparseMatrix<float, Eigen::RowMajor>> xt
      = read_xt(filename);

  std::cout << "reader:  built matrix"
            << std::endl;
  std::cout << "reader:  xt.rows() = " << xt.rows()
            << std::endl;
  std::cout << "reader:  xt.cols() = " << xt.cols()
            << std::endl;
  std::cout << "reader:  xt.nonZeros() = " << xt.nonZeros()
            << std::endl;

  std::cout << "reader:  building multinomial model"
            << std::endl;

  for (int32_t i = 0; i < 10; ++i)
    for (int32_t j = 0; j < 10; ++j)
      std::cout << "xt[" << i << ", " << j << "] = " << xt.coeffRef(i, j)
                << std::endl;

  Eigen::VectorXf y = Eigen::VectorXf::Ones(xt.rows());
  std::cout << "reader: y.rows() = " << y.rows()
            << std::endl;
  multinomial_model model(xt, y);
  std::cout << "reader: model.y_.size() = " << model.y_.size()
            << std::endl;

  // Eigen::VectorXf beta = Eigen::VectorXf::Ones(xt.cols());
  // float lp = model.log_density(beta);
  // std::cout << "lp = " << lp << std::endl;
}
