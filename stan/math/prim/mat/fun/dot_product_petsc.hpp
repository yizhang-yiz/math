#ifndef STAN_MATH_PRIM_MAT_FUN_DOT_PRODUCT_HPP
#define STAN_MATH_PRIM_MAT_FUN_DOT_PRODUCT_HPP

#include <stan/math/prim/mat/fun/to_vec_petsc.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/err/check_vector.hpp>
#include <stan/math/prim/arr/err/check_matching_sizes.hpp>
#include <petscvec.h>
#include <vector>


#include <chrono>
#include <iostream>

namespace stan {
namespace math {

/**
 * Returns the dot product of the specified vectors.
 *
 * @param v1 First vector.
 * @param v2 Second vector.
 * @return Dot product of the vectors.
 * @throw std::domain_error If the vectors are not the same
 * size or if they are both not vector dimensioned.
 */
inline double dot_product_eigen(const Eigen::VectorXd& v1,
                                const Eigen::VectorXd& v2) {
  std::chrono::duration<double> elapsed_seconds;
  check_vector("dot_product", "v1", v1);
  check_vector("dot_product", "v2", v2);
  check_matching_sizes("dot_product", "v1", v1, "v2", v2);
  auto start = std::chrono::system_clock::now();
  auto res = v1.dot(v2);
  auto end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
   return res;
}
  
inline double dot_product_petsc(const Eigen::VectorXd& v1,
                                const Eigen::VectorXd& v2) {
  PetscErrorCode ierr;
  Vec            vv1, vv2;
  PetscScalar    res;
  std::chrono::duration<double> elapsed_seconds;
  ierr = PetscInitialize(0, NULL,(char*)0,"");if (ierr) return ierr;

  to_vec_petsc(v1, PETSC_COMM_WORLD, &vv1);CHKERRQ(ierr);
  to_vec_petsc(v2, PETSC_COMM_WORLD, &vv2);CHKERRQ(ierr);
  
  // ierr = VecView(vv2,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDot(vv1, vv2, &res);CHKERRQ(ierr);CHKERRQ(ierr);

  ierr = VecDestroy(&vv1);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = VecDestroy(&vv2);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);

  return res;

}

}  // namespace math
}  // namespace stan
#endif
