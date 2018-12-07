#include <stan/math/rev/mat.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/rev/mat/util.hpp>
#include <stan/math/rev/mat/fun/multiply.hpp>
#include <stan/math/rev/mat/fun/to_var.hpp>
#include <vector>

TEST(MathMatrix, multiply_seg_fault_vd) {
  Eigen::Matrix<stan::math::var, -1, -1> A=Eigen::Matrix<stan::math::var, -1, -1>::Random(1, 1);
  Eigen::Matrix<double, -1, -1> B=Eigen::Matrix<double, -1, -1>::Random(1, 1);

  Eigen::Matrix<stan::math::var, -1, -1> result = stan::math::multiply(A, B);

  std::vector<double> gradients;

  std::vector<stan::math::var> vars = stan::math::to_array_1d(A);
  result(0, 0).grad(vars, gradients);
}
