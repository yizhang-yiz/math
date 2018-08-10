#ifndef STAN_MATH_REV_SCAL_FUNCTOR_COMPLEX_STEP_DERIVATIVE_HPP
#define STAN_MATH_REV_SCAL_FUNCTOR_COMPLEX_STEP_DERIVATIVE_HPP

#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/prim/scal/functor/complex_step_derivative.hpp>
#include <stan/math/rev/core/precomp_v_vari.hpp>
#include <stan/math/rev/core/vari.hpp>
#include <stan/math/rev/core/var.hpp>

#include <vector>
#include <complex>
#include <iostream>

namespace stan {
namespace math {

/**
 * Return a var that has value of given functor F and derivative
 * of df/d(theta), using complex step derivative
 * approximation. "f" does not have to support "var"
 * type, as its signature should be
 * (complex, std::vector<double>, std::vector<int>, stream*) : complex
 *
 * @tparam F type of functor F
 * @param[in] f functor for the complex number evaluation,
 * must support @c std::complex<double> as arg.
 * @param[in] theta parameter where f and df/d(theta) is requested.
 * @param[in] x_r continuous data vector for the ODE.
 * @param[in] x_i integer data vector for the ODE.
 * @param[in] h complex step size
 * @param[out] msgs the print stream for warning messages.
 * @return a var with value f(theta.val()) and derivative at theta.
 */
template <typename F>
stan::math::var complex_step_derivative(const F& f,
                                        const stan::math::var& theta,
                                        const std::vector<double>& x_r,
                                        const std::vector<int>& x_i,
                                        const double h,
                                        std::ostream* msgs) {
  using stan::math::var;
  using std::complex;
  const double theta_d = theta.val();
  const complex<double> res = f(complex<double>(theta_d, h), x_r, x_i, msgs);
  const double fx = std::real(res);
  const double g = std::imag(res) / h;
  return var(new stan::math::precomp_v_vari(fx, theta.vi_, g));
}

/**
 * CSDA, default h version, with h = 1.E-20
 *
 * @tparam F type of functor F
 * @param[in] f functor for the complex number evaluation,
 * must support @c std::complex<double> as arg.
 * @param[in] theta parameter where f and df/d(theta) is requested.
 * @param[in] x_r continuous data vector for the ODE.
 * @param[in] x_i integer data vector for the ODE.
 * @param[out] msgs the print stream for warning messages.
 * @return a var with value f(theta.val()) and derivative at theta.
 */
template <typename F>
stan::math::var complex_step_derivative(const F& f,
                                        const stan::math::var& theta,
                                        const std::vector<double>& x_r,
                                        const std::vector<int>& x_i,
                                        std::ostream* msgs) {
  return complex_step_derivative(f, theta, x_r, x_i, 1.E-20, msgs);
}

}  // namespace math
}  // namespace stan

#endif
