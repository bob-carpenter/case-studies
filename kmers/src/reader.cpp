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

int main() {
  std::string filename = "xt.bin";
  std::fstream in(filename, std::ios::binary | std::ios::in);
  if (!in) {
    std::string msg = "couldn't open file = " + filename;
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

  std::cout << "reader:  building Eigen::Map"
            << std::endl;

  Eigen::Map<Eigen::SparseMatrix<float, Eigen::RowMajor>>
      xt(rows, cols, nnz, outerIndexPtr.data(), innerIndices.data(), values.data());

  std::cout << "reader:  finished building map"
            << std::endl;
  std::cout << "reader:  xt.rows() = " << xt.rows()
            << std::endl;

  in.close();
}
