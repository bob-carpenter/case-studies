#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

template <typename T>
T read_val(std::istream& in) {
  T x;
  in.read(reinterpret_cast<char*>(&x), sizeof(T));
  return x;
}
template <typename T>
std::vector<T> read_array(std::istream& in, int N) {
  std::vector<T> y(N);
  for (int n = 0; n < N; ++n)
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
  int cols = read_val<int>(in);
  std::cout << "reader:  cols = " << cols
            << std::endl;

  int rows = read_val<int>(in);
  std::cout << "reader:  rows = " << rows
            << std::endl;

  int nnz = read_val<int>(in);
  std::cout << "reader:  nnz = " << nnz
            << std::endl;

  std::vector<int> outerIndexPtr = read_array<int>(in, rows + 1);
  std::cout << "reader: read outerIndexPtr"
            << std::endl;

  std::vector<int> innerIndices = read_array<int>(in, nnz);
  std::cout << "reader: read innerIndices"
            << std::endl;

  std::vector<float> values = read_array<float>(in, nnz);
  std::cout << "reader: read innerIndices"
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
  std::cout << "reader:  xt.cols() = " << xt.rows()
            << std::endl;
  std::cout << "reader:  xt.nonZeros() = " << xt.nonZeros()
            << std::endl;
}
