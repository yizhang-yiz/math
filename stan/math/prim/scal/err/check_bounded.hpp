#ifndef STAN_MATH_PRIM_SCAL_ERR_CHECK_BOUNDED_HPP
#define STAN_MATH_PRIM_SCAL_ERR_CHECK_BOUNDED_HPP

#include <stan/math/prim/scal/err/domain_error.hpp>
#include <stan/math/prim/scal/err/domain_error_vec.hpp>
#include <stan/math/prim/scal/err/is_bounded.hpp>
#include <stan/math/prim/scal/meta/max_size.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>
#include <string>

namespace stan {
  namespace math {

    /**
     * Check if the value is between the low and high
     * values, inclusively.
     *
     * @tparam T_y Type of value
     * @tparam T_low Type of low value
     * @tparam T_high Type of high value
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param y Value to check
     * @param low Low bound
     * @param high High bound
     *
     * @throw <code>std::domain_error</code> otherwise. This also throws
     *   if any of the arguments are NaN.
     */
    template <typename T_y, typename T_low, typename T_high>
    inline void check_bounded(const char* function,
                              const char* name,
                              const T_y& y,
                              const T_low& low,
                              const T_high& high) {
      if (is_bounded(y, low, high))
        return;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_low> low_vec(low);
      VectorView<const T_high> high_vec(high);
      for (size_t n = 0; n < max_size(y, low, high); n++) {
        if (!(low_vec[n] <= y_vec[n] && y_vec[n] <= high_vec[n])) {
          std::stringstream msg;
          msg << ", but must be between ";
          msg << "(" << low_vec[n] << ", " << high_vec[n] << ")";
          std::string msg_str(msg.str());
          domain_error(function, name, y_vec[n],
                       "is ", msg_str.c_str());
        }
      }
    }
    
  }
}
#endif
