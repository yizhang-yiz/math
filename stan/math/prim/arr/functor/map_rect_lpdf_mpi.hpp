#pragma once

#include <vector>
#include <set>

#include <stan/math/prim/mat/fun/to_array_1d.hpp>
#include <stan/math/prim/arr/functor/mpi_command.hpp>

namespace stan {
  namespace math {

    template <typename F>
    double
    map_rect_lpdf_mpi(const std::vector<double>& eta,
                      const std::vector<std::vector<double> >& theta,
                      const std::vector<std::vector<double> >& x_r,
                      const std::vector<std::vector<int> >& x_i,
                      const std::size_t uid) {

      double res = 0;
      const std::size_t N = theta.size();

      for(std::size_t i = 0; i != N; ++i) {
        res += F::apply(eta, theta[i], x_r[i], x_i[i]);
      }

      return(res);
    }

  }
}
