#ifndef STAN_MATH_PRIM_ARR_FUN_TO_VEC_PETSC_HPP
#define STAN_MATH_PRIM_ARR_FUN_TO_VEC_PETSC_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/err/check_vector.hpp>
#include <stan/math/prim/arr/err/check_matching_sizes.hpp>
#include <petscvec.h>
#include <vector>

#include <iostream>

namespace stan {
namespace math {

  PetscErrorCode to_vec_petsc(Eigen::VectorXd& v, const MPI_Comm &comm, Vec *vv) {
    PetscErrorCode ierr;
    PetscInt       n, rank, size, bs=1;
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

    n = v.size() / size;
    if (rank == size - 1 && v.size() % size != 0) {
      n = v.size() % size;
    }
    ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD, bs, n, v.size(), &v[rank*n], vv);CHKERRQ(ierr);
    return ierr;
  }

  PetscErrorCode to_vec_petsc(const Eigen::VectorXd& v, const MPI_Comm &comm, Vec *vv) {
    PetscErrorCode ierr;
    PetscInt       n, n0, rank, size, vsize =v.size(), bs = 1;

    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    n0 = vsize / size;
    if (rank == size - 1) {
      n = vsize - (size - 1) * n0;
    } else {
      n = n0;
    }
    ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD, bs, n, v.size(), &v[rank*n0], vv);CHKERRQ(ierr);
    return ierr;
  }

}
}

#endif
