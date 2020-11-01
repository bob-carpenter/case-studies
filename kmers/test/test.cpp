#include <cmath>

#include <test/kmers/softmax.hpp>

int main() {
  test_softmax();
}

template <typename T1, typename T2>
void expect_near(const T1& x1, const T2& x2, double tol = 1e-8) {
  if (is_near(x1, x2, tol)) return true;

}

template <typename T1, typename T2>
bool is_near(const T1& x1, const T2& x2, double tol = 1e-8) {
  return std::fabs(x1 - x2) < tol;
}


void test_softmax() {
  expect_near(1, 1) << "EXPECT 1 == 1";
  expect_near(1.001, 1.002, 0.1);
}
