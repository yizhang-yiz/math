#ifndef STAN_MATH_PRIM_SCAL_ERR_IS_BOUNDED_HPP
#define STAN_MATH_PRIM_SCAL_ERR_IS_BOUNDED_HPP

#include <stan/math/prim/scal/meta/max_size.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>

namespace stan {
  namespace math {

    template <typename T_y, typename T_low, typename T_high>
    bool is_bounded(const T_y& y, const T_low& low, const T_high& high) {
      VectorView<const T_y> y_vec(y);
      VectorView<const T_low> low_vec(low);
      VectorView<const T_high> high_vec(high);
      for (size_t n = 0; n < stan::max_size(y, low, high); n++) {
        if (!(low_vec[n] <= y_vec[n] && y_vec[n] <= high_vec[n])) {
          return false;
        }
      }
      return true;
    }
    
  }
}
#endif
