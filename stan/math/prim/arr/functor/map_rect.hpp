#pragma once

#ifdef STAN_HAS_MPI
#include <stan/math/prim/arr/functor/map_rect_mpi.hpp>
#endif

namespace stan {
  namespace math {

    template <typename F>
    std::vector<double>
    map_rect(const std::vector<double>& eta,
             const std::vector<std::vector<double> >& theta,
             const std::vector<std::vector<double> >& x_r,
             const std::vector<std::vector<int> >& x_i,
             const std::size_t uid) {
#ifdef STAN_HAS_MPI
      return(map_rect_mpi<F>(eta, theta, x_r, x_i, uid));
#else
      std::vector<double> res;
      const std::size_t N = theta.size();

      for(std::size_t i = 0; i != N; ++i) {
        const std::vector<double> f = F::apply(eta, theta[i], x_r[i], x_i[i]);
        res.insert(res.end(), f.begin(), f.end());
      }
      return(res);
#endif
    }
  }
}
