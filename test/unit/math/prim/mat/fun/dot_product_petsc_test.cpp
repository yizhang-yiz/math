#include <stan/math/prim/mat/fun/dot_product_petsc.hpp>
#include <stan/math/prim/mat.hpp>
#include <gtest/gtest.h>


#include <stan/math.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <iostream>
#include <sstream>
#include <vector>

#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>
#include <petscdraw.h>
#include <petscdt.h>
#include <petscviewer.h>

#include <chrono>

TEST(MathMatrix, dot_product_petsc) {
  using Eigen::VectorXd;
  using stan::math::dot_product_petsc;
  using stan::math::dot_product_eigen;

  VectorXd v1 = VectorXd::Random(8000000,1);
  VectorXd v2 = VectorXd::Random(8000000,1);
  auto res1 = dot_product_eigen(v1, v2);
  auto res2 = dot_product_petsc(v1, v2);

  std::cout << "eigen result: " << res1 << "\n";
  std::cout << "petsc result: " << res2 << "\n";

}
