#ifndef STAN_MATH_PRIM_MAT_FUN_DOT_PRODUCT_HPP
#define STAN_MATH_PRIM_MAT_FUN_DOT_PRODUCT_HPP

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
  std::cout << "elapsed time(dot_product_eigen): " << elapsed_seconds.count() << "\n";
  return res;
}
  
inline double dot_product_petsc(const Eigen::VectorXd& v1,
                                const Eigen::VectorXd& v2) {
  PetscErrorCode ierr;
  PetscInt       i, n=1000000, rank, size;
  Vec            vv1, vv2;
  PetscScalar    res;
  std::chrono::duration<double> elapsed_seconds;
  ierr = PetscInitialize(0, NULL,(char*)0,"");if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&vv1);CHKERRQ(ierr); ierr = VecSetSizes(vv1,PETSC_DECIDE,n*size);CHKERRQ(ierr); ierr = VecSetFromOptions(vv1);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&vv2);CHKERRQ(ierr); ierr = VecSetSizes(vv2,PETSC_DECIDE,n*size);CHKERRQ(ierr); ierr = VecSetFromOptions(vv2);CHKERRQ(ierr);

  for (i=n*rank; i<n*(rank+1); i++) {
    ierr  = VecSetValues(vv1,1,&i,&v1[i],INSERT_VALUES);CHKERRQ(ierr);
    ierr  = VecSetValues(vv2,1,&i,&v2[i],INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(vv1);CHKERRQ(ierr); ierr = VecAssemblyEnd(vv1);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(vv2);CHKERRQ(ierr); ierr = VecAssemblyEnd(vv2);CHKERRQ(ierr);

  // ierr = VecView(vv2,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  auto start = std::chrono::system_clock::now();
  ierr = VecDot(vv1, vv2, &res);CHKERRQ(ierr);
  auto end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  std::cout << "elapsed time(dot_product_petsc): " << elapsed_seconds.count() << "\n";

  ierr = VecDestroy(&vv1);CHKERRQ(ierr);
  ierr = VecDestroy(&vv2);CHKERRQ(ierr);
  ierr = PetscFinalize();

  return res;

}

}  // namespace math
}  // namespace stan
#endif
