#ifndef STAN_MATH_PRIM_PROB_WIENER5_LPDF_HPP
#define STAN_MATH_PRIM_PROB_WIENER5_LPDF_HPP

#include <stan/math/prim/fun.hpp>

namespace stan {
namespace math {
namespace internal {

/**
 * Calculate the 'error_term' term for a wiener5 density or gradient
 *
 * @param y A scalar variable; the reaction time in seconds
 * @param a The boundary separation
 * @param v_value The drift rate
 * @param w_value The relative starting point
 * @param sv The inter-trial variability of the drift rate
 * @return 'error_term' term
 */
template <typename Scalar, typename T_y, typename T_a, typename T_v,
          typename T_w, typename T_sv>
inline Scalar wiener5_compute_error_term(T_y&& y, T_a&& a, T_v&& v_value,
                                         T_w&& w_value, T_sv&& sv) noexcept {
  const Scalar w = 1.0 - w_value;
  const Scalar v = -v_value;
  const Scalar sv_sqr = square(sv);
  const Scalar one_plus_svsqr_y = 1 + sv_sqr * y;
  const Scalar two_avw = 2 * a * v * w;
  const Scalar two_log_a = 2 * log(a);
  if (sv != 0) {
    return stan::math::eval((sv_sqr * square(a * w) - two_avw - square(v) * y)
                                / 2.0 / one_plus_svsqr_y
                            - two_log_a - 0.5 * log(one_plus_svsqr_y));
  } else {
    return stan::math::eval((-two_avw - square(v) * y) / 2.0 - two_log_a);
  }
}

/**
 * Calculate the 'density_part_one' term for a wiener5 density or gradient
 *
 * @tparam GradA Whether the calculation is for gradient w.r.t. 'a'
 * @tparam GradT Whether the calculation is for gradient w.r.t. 't'
 * @tparam Scalar type of scalars
 *
 * @param y A scalar variable; the reaction time in seconds
 * @param a The boundary separation
 * @param v_value The drift rate
 * @param w_value The relative starting point
 * @param sv The inter-trial variability of the drift rate
 * @return 'density_part_one' term
 */
template <bool GradA, bool GradT, typename Scalar, typename T_y, typename T_a,
          typename T_v_value, typename T_w_value, typename T_sv>
inline Scalar wiener5_density_part_one(T_y&& y, T_a&& a, T_v_value&& v_value,
                                       T_w_value&& w_value,
                                       T_sv&& sv) noexcept {
  const Scalar w = 1.0 - w_value;
  const Scalar v = -v_value;
  const Scalar sv_sqr = square(sv);
  const Scalar one_plus_svsqr_y = 1 + sv_sqr * y;
  const Scalar two_avw = 2 * a * v * w;

  const Scalar var_a = GradA ? w : a;
  const Scalar var_b = GradA ? a : w;

  if (GradT) {
    if (sv != 0) {
      return -0.5
             * (square(sv_sqr) * (y + square(a * w)) + sv_sqr * (1 - two_avw)
                + square(v))
             / square(one_plus_svsqr_y);
    } else {
      return -0.5 * square(v);
    }
  }

  if (sv != 0) {
    return (-v * var_a + sv_sqr * square(var_a) * var_b) / one_plus_svsqr_y;
  } else {
    return -v * var_a;
  }
}

/**
 * Calculate the 'n_terms_small_t' term for a wiener5 density or gradient
 *
 * @tparam Density Whether the calculation is for the density
 * @tparam GradW Whether the calculation is for gradient w.r.t. 'w'
 * @tparam Scalar type of scalars
 *
 * @param y A scalar variable; the reaction time in seconds
 * @param a The boundary separation
 * @param w_value The relative starting point
 * @param error The error tolerance
 * @return 'n_terms_small_t' term
 */
template <bool Density, bool GradW, typename Scalar, typename T_y, typename T_a,
          typename T_w_value>
inline Scalar wiener5_n_terms_small_t(T_y&& y, T_a&& a, T_w_value&& w_value,
                                      Scalar error) noexcept {
  const Scalar two_error = 2.0 * error;
  const Scalar y_asq = y / square(a);
  const Scalar two_log_a = 2 * log(a);
  const Scalar log_y_asq = log(y) - two_log_a;
  const Scalar w = 1.0 - w_value;

  const Scalar n_1_factor = Density ? 2 : 3;
  const Scalar n_1 = (sqrt(n_1_factor * y_asq) + w) / 2.0;
  Scalar u_eps;
  if (Density || GradW) {
    u_eps = fmin(-1.0, LOG_TWO + LOG_PI + 2.0 * log_y_asq + two_error);
  } else {
    u_eps = fmin(-3.0,
                 (log(8.0) - log(27.0) + LOG_PI + 4.0 * log_y_asq + two_error));
  }
  const Scalar arg_mult = (Density || GradW) ? 1 : 3;
  const Scalar arg = -arg_mult * y_asq * (u_eps - sqrt(-2.0 * u_eps - 2.0));  
  
  const Scalar n_2
      = (arg > 0) ? GradW ? 0.5 * (sqrt(arg) + w) : 0.5 * (sqrt(arg) - w) : n_1;

  return ceil(fmax(n_1, n_2));
}

/**
 * Calculate the 'n_terms_small_t' term for a wiener5 density or gradient
 *
 * @tparam Density Whether the calculation is for the density
 * @tparam GradT Whether the calculation is for gradient w.r.t. 't'
 * @tparam Scalar type of scalars
 *
 * @param y A scalar variable; the reaction time in seconds
 * @param a The boundary separation
 * @param w_value The relative starting point
 * @param error The error tolerance
 * @return 'n_terms_large_t' term
 */
template <bool Density, bool GradW, typename Scalar, typename T_y, typename T_a,
          typename T_w_value>
inline Scalar wiener5_n_terms_large_t(T_y&& y, T_a&& a, T_w_value&& w_value,
                                       Scalar error) noexcept {
  const Scalar two_error = 2.0 * error;
  const Scalar y_asq = y / square(a);
  const Scalar two_log_a = 2 * log(a);
  const Scalar log_y_asq = log(y) - two_log_a;
  const Scalar w = 1.0 - w_value;

  const Scalar n_1_factor = GradW ? 2 : 3;
  static constexpr double PI_SQUARED = pi() * pi();

  Scalar n_1;
  Scalar n_2;
  if (Density) {
    n_1 = 1.0 / (pi() * sqrt(y_asq));
    const Scalar two_log_piy = -2.0 * (LOG_PI + log_y_asq + error);
    n_2 = (two_log_piy >= 0) ? sqrt(two_log_piy / (PI_SQUARED * y_asq)) : 0.0;
  } else {
    n_1 = sqrt(n_1_factor / y_asq) / pi();
    const Scalar u_eps_arg
        = GradW
              ? log(4.0) - log(9.0) + 2.0 * LOG_PI + 3.0 * log_y_asq + two_error
              : log(3.0) - log(5.0) + LOG_PI + 2.0 * log_y_asq + error;
    const Scalar u_eps = fmin(-1, u_eps_arg);
    const Scalar arg_mult = GradW ? 1 : (2.0 / PI_SQUARED / y_asq);
    const Scalar arg = -arg_mult * (u_eps - sqrt(-2.0 * u_eps - 2.0));
    n_2 = GradW ? (arg > 0) ? sqrt(arg / y_asq) / pi() : n_1
                : (arg > 0) ? sqrt(arg) : n_1;
  }

  return ceil(fmax(n_1, n_2));
}

/**
 * Calculate the 'result' term and its sign for a wiener5 density or gradient
 *
 * @tparam Density Whether the calculation is for the density
 * @tparam GradW Whether the calculation is for gradient w.r.t. 'w'
 * @tparam Scalar type of scalars
 *
 * @param y A scalar variable; the reaction time in seconds
 * @param a The boundary separation
 * @param w_value The relative starting point
 * @param n_terms_small_t The n_terms_small_t term
 * @param n_terms_large_t The n_terms_large_t term
 * @return 'result' sum and its sign
 */
template <bool Density, bool GradW, typename Scalar, typename T_y, typename T_a,
          typename T_w, typename T_nsmall, typename T_nlarge>
inline auto wiener5_log_sum_exp(T_y&& y, T_a&& a, T_w&& w_value,
                                T_nsmall&& n_terms_small_t,
                                T_nlarge&& n_terms_large_t) noexcept {
  const auto y_asq = y / square(a);
  const auto w = 1.0 - w_value;
  const bool small_n_terms_small_t
      = Density ? (2 * n_terms_small_t <= n_terms_large_t)
                : (2 * n_terms_small_t < n_terms_large_t);
  const auto scaling = small_n_terms_small_t ? inv(2.0 * y_asq) : y_asq / 2.0;

    using ret_t = return_type_t<T_y, T_a, T_w>;
    ret_t current_val;
    int current_sign;
	if (small_n_terms_small_t) {
	ret_t fplus = NEGATIVE_INFTY;
	ret_t fminus = NEGATIVE_INFTY;
    const auto mult = Density ? 1 : 3;
    const auto offset = GradW ? y_asq : 0;
	
    for (auto k = n_terms_small_t; k >= 1; k--) {
        const auto wp2k = w + 2.0 * k;
        const auto wm2k = w - 2.0 * k;
		if (GradW) {
			const auto sqwp2k_mo = square(wp2k) - offset;
			if (sqwp2k_mo > 0) {
				const auto wp2k_quant = log(sqwp2k_mo) - square(wp2k) * scaling;
				fplus = log_sum_exp(fplus, wp2k_quant);
			} else if (sqwp2k_mo < 0) {
				const auto wp2k_quant = log(-sqwp2k_mo) - square(wp2k) * scaling;
				fminus = log_sum_exp(fminus, wp2k_quant);
			}
		} else {
			const auto wp2k_quant = mult * log(wp2k) - square(wp2k) * scaling;
			fplus = log_sum_exp(fplus, wp2k_quant);
		}

		if (GradW) {
			const auto sqwm2k_mo = square(wm2k) - offset;
			if (sqwm2k_mo > 0) {
				const auto wm2k_quant = log(sqwm2k_mo) - square(wm2k) * scaling;
				fplus = log_sum_exp(fplus, wm2k_quant);
			} else if (sqwm2k_mo < 0) {
				const auto wm2k_quant = log(-sqwm2k_mo) - square(wm2k) * scaling;
				fminus = log_sum_exp(fminus, wm2k_quant);
			}
		} else {
			const auto wm2k_quant = mult * log(-wm2k) - square(wm2k) * scaling;
			if (fminus <= NEGATIVE_INFTY) {
				fminus = wm2k_quant;
			} else if (wm2k_quant <= NEGATIVE_INFTY) {
				fminus = fminus;
			} else if (fminus > wm2k_quant) {
				fminus = fminus + log1p(exp(wm2k_quant - fminus));
			} else {
				fminus = wm2k_quant + log1p(exp(fminus - wm2k_quant));
			}
		}
    }

	if (GradW) {
		const auto sqw_mo = square(w) - offset;
		if (sqw_mo > 0) {
		  const auto new_val = log(sqw_mo) - square(w) * scaling;
		  fplus = log_sum_exp(fplus, new_val);
		} else if (sqw_mo < 0) {
		  const auto new_val = log(-sqw_mo) - square(w) * scaling;
		  fminus = log_sum_exp(fminus, new_val);
		}
	} else {
		const auto new_val = mult * log(w) - square(w) * scaling;
		fplus = log_sum_exp(fplus, new_val);
	}

	if (fplus == NEGATIVE_INFTY) {
		current_val = fminus;
	} else if (fminus == NEGATIVE_INFTY) {
		current_val = fplus;
	} else if (fplus > fminus) {
		current_val = log_diff_exp(fplus, fminus);
	} else if (fplus < fminus) {
		current_val = log_diff_exp(fminus, fplus);
	} else {
		current_val = NEGATIVE_INFTY;
	}
	current_sign = (fplus < fminus)? -1: 1;
	
  } else { // for large t
    auto mult = 3;
    if (Density) {
      mult = 1;
    } else if (GradW) {
      mult = 2;
    }
	ret_t fplus = NEGATIVE_INFTY;
	ret_t fminus = NEGATIVE_INFTY;
    for (auto k = n_terms_large_t; k >= 1; k--) {
        const auto pi_k = k * pi();
        const auto check = (GradW) ? cos(pi_k * w) : sin(pi_k * w);
		if (check > 0) {
		  fplus = log_sum_exp(fplus, mult*log(k) - square(pi_k) * scaling + log(check));
		} else if ((GradW && check < 0) || !GradW) {
		  fminus = log_sum_exp(fminus, mult*log(k) - square(pi_k) * scaling + log(-check));
		}
	}
	if (fplus == NEGATIVE_INFTY) {
		current_val = fminus;
	} else if (fminus == NEGATIVE_INFTY) {
		current_val = fplus;
	} else if (fplus > fminus) {
		current_val = log_diff_exp(fplus, fminus);
	} else if (fplus < fminus) {
		current_val = log_diff_exp(fminus, fplus);
	} else {
		current_val = NEGATIVE_INFTY;
	}
	current_sign = (fplus < fminus)? -1: 1;
  }
  return std::make_pair(current_val, current_sign);
}

/**
 * Calculate the wiener5 density
 *
 * @tparam NaturalScale Whether to return the density on natural (true) or
 * log-scale (false)
 *
 * @param y A scalar variable; the reaction time in seconds
 * @param a The boundary separation
 * @param v_value The drift rate
 * @param w_value The relative starting point
 * @param sv The inter-trial variability of the drift rate
 * @param err The log error tolerance
 * @return density
 */
template <bool NaturalScale = false, typename Scalar, typename T_y,
          typename T_a, typename T_w, typename T_v, typename T_sv>
inline Scalar wiener5_density(const T_y& y, const T_a& a, const T_v& v_value,
                              const T_w& w_value, const T_sv& sv,
                              Scalar err = log(1e-12)) noexcept {
  const Scalar error_term
      = wiener5_compute_error_term<Scalar>(y, a, v_value, w_value, sv);
  const Scalar error = (err - error_term);
  const Scalar n_terms_small_t
      = wiener5_n_terms_small_t<true, false, Scalar>(y, a, w_value, error);
  const Scalar n_terms_large_t
      = wiener5_n_terms_large_t<true, false, Scalar>(y, a, w_value, error);

  auto res = wiener5_log_sum_exp<true, false, Scalar>(
                 y, a, w_value, n_terms_small_t, n_terms_large_t)
                 .first;
  if (2 * n_terms_small_t <= n_terms_large_t) {
    Scalar log_density = error_term - 0.5 * LOG_TWO - LOG_SQRT_PI
                         - 1.5 * (log(y) - 2 * log(a)) + res;
    return NaturalScale ? exp(log_density) : log_density;
  } else {
    Scalar log_density = error_term + res + LOG_PI;
    return NaturalScale ? exp(log_density) : log_density;
  }
}

/**
 * Calculate the derivative of the wiener5 density w.r.t. 't'
 *
 * @tparam WrtLog Whether to return the derivative w.r.t.
 *                  the natural (true) or log-scale (false) density
 * @tparam Scalar type of scalars
 *
 * @param y A scalar variable; the reaction time in seconds
 * @param a The boundary separation
 * @param v_value The drift rate
 * @param w_value The relative starting point
 * @param sv The inter-trial variability of the drift rate
 * @param err The log error tolerance
 * @return Gradient w.r.t. t
 */
template <bool WrtLog = false, typename Scalar, typename T_y, typename T_a,
          typename T_w, typename T_v, typename T_sv>
inline Scalar wiener5_grad_t(const T_y& y, const T_a& a, const T_v& v_value,
                             const T_w& w_value, const T_sv& sv,
                             Scalar err = log(1e-12)) noexcept {
  const Scalar two_log_a = 2 * log(a);
  const Scalar log_y_asq = log(y) - two_log_a;
  const Scalar error_term
      = wiener5_compute_error_term<Scalar>(y, a, v_value, w_value, sv);
  const Scalar density_part_one = wiener5_density_part_one<false, true, Scalar>(
      y, a, v_value, w_value, sv);
  const Scalar error = (err - error_term) + two_log_a;

  const Scalar n_terms_small_t
      = wiener5_n_terms_small_t<false, false, Scalar>(y, a, w_value, error);
  const Scalar n_terms_large_t
      = wiener5_n_terms_large_t<false, false, Scalar>(y, a, w_value, error);
  Scalar result;
  int newsign;
  std::forward_as_tuple(result, newsign)
      = wiener5_log_sum_exp<false, false, Scalar>(
          y, a, w_value, n_terms_small_t, n_terms_large_t);

  const Scalar error_log_density
      = log(fmax(fabs(density_part_one - 1.5 / y), fabs(density_part_one)));
  const Scalar log_density = wiener5_density<false, Scalar>(
      y, a, v_value, w_value, sv, err - error_log_density);
  Scalar ans;
  if (2 * n_terms_small_t < n_terms_large_t) {
    ans = density_part_one - 1.5 / y
          + newsign
                * exp(error_term - two_log_a - 1.5 * LOG_TWO - LOG_SQRT_PI
                      - 3.5 * log_y_asq + result - log_density);
  } else {
    ans = density_part_one
          - newsign
                * exp(error_term - two_log_a + 3.0 * LOG_PI - LOG_TWO + result
                      - log_density);
  }
  return WrtLog ? ans * exp(log_density) : ans;
}

/**
 * Calculate the derivative of the wiener5 density w.r.t. 'a'
 *
 * @tparam WrtLog Whether to return the derivative w.r.t.
 *                  the natural (true) or log-scale (false) density
 * @tparam Scalar type of scalars
 *
 * @param y A scalar variable; the reaction time in seconds
 * @param a The boundary separation
 * @param v_value The drift rate
 * @param w_value The relative starting point
 * @param sv The inter-trial variability of the drift rate
 * @param err The log error tolerance
 * @return Gradient w.r.t. a
 */
template <bool WrtLog = false, typename Scalar, typename T_y, typename T_a,
          typename T_w, typename T_v, typename T_sv>
inline Scalar wiener5_grad_a(const T_y& y, const T_a& a, const T_v& v_value,
                             const T_w& w_value, const T_sv& sv,
                             Scalar err = log(1e-12)) noexcept {
  const Scalar two_log_a = 2 * log(a);
  const Scalar log_y_asq = log(y) - two_log_a;
  const Scalar error_term
      = wiener5_compute_error_term<Scalar>(y, a, v_value, w_value, sv);
  const Scalar density_part_one = wiener5_density_part_one<true, false, Scalar>(
      y, a, v_value, w_value, sv);
  const Scalar error = err - error_term + 3 * log(a) - log(y) - LOG_TWO;

  const Scalar n_terms_small_t
      = wiener5_n_terms_small_t<false, false, Scalar>(y, a, w_value, error);
  const Scalar n_terms_large_t
      = wiener5_n_terms_large_t<false, false, Scalar>(y, a, w_value, error);
  Scalar result;
  int newsign;
  std::forward_as_tuple(result, newsign)
      = wiener5_log_sum_exp<false, false, Scalar>(
          y, a, w_value, n_terms_small_t, n_terms_large_t);

  const Scalar error_log_density = log(
      fmax(fabs(density_part_one + 1.0 / a), fabs(density_part_one - 2.0 / a)));
  const Scalar log_density = wiener5_density<false, Scalar>(
      y, a, v_value, w_value, sv, err - error_log_density);
  Scalar ans;
  if (2 * n_terms_small_t < n_terms_large_t) {
    ans = density_part_one + 1.0 / a
          - newsign
                * exp(-0.5 * LOG_TWO - LOG_SQRT_PI - 2.5 * log(y)
                      + 2.0 * two_log_a + error_term + result - log_density);
  } else {
    ans = density_part_one - 2.0 / a
          + newsign
                * exp(log(y) + error_term - 3 * (log(a) - LOG_PI) + result
                      - log_density);
  }
  return WrtLog ? ans * exp(log_density) : ans;
}

/**
 * Calculate the derivative of the wiener5 density w.r.t. 'v'
 *
 * @tparam WrtLog Whether to return the derivative w.r.t.
 *                  the natural (true) or log-scale (false) density
 * @tparam Scalar type of scalars
 *
 * @param y A scalar variable; the reaction time in seconds
 * @param a The boundary separation
 * @param v_value The drift rate
 * @param w_value The relative starting point
 * @param sv The inter-trial variability of the drift rate
 * @param err The log error tolerance
 * @return Gradient w.r.t. v
 */
template <bool WrtLog = false, typename Scalar, typename T_y, typename T_a,
          typename T_w, typename T_v, typename T_sv>
inline Scalar wiener5_grad_v(const T_y& y, const T_a& a, const T_v& v_value,
                             const T_w& w_value, const T_sv& sv,
                             Scalar err = log(1e-12)) noexcept {
  Scalar ans = (a * (1 - w_value) - v_value * y);
  if (sv != 0) {
    ans /= 1 + square(sv) * y;
  }
  return WrtLog ? ans
                      * wiener5_density<true, Scalar>(y, a, v_value, w_value,
                                                      sv, err)
                : ans;
}

/**
 * Calculate the derivative of the wiener5 density w.r.t. 'w'
 *
 * @tparam WrtLog Whether to return the derivative w.r.t.
 *                  the natural (true) or log-scale (false) density
 * @tparam Scalar type of scalars
 *
 * @param y A scalar variable; the reaction time in seconds
 * @param a The boundary separation
 * @param v_value The drift rate
 * @param w_value The relative starting point
 * @param sv The inter-trial variability of the drift rate
 * @param err The log error tolerance
 * @return Gradient w.r.t. w
 */
template <bool WrtLog = false, typename Scalar, typename T_y, typename T_a,
          typename T_w, typename T_v, typename T_sv>
inline Scalar wiener5_grad_w(const T_y& y, const T_a& a, const T_v& v_value,
                             const T_w& w_value, const T_sv& sv,
                             Scalar err = log(1e-12)) noexcept {
  const Scalar two_log_a = 2 * log(a);
  const Scalar log_y_asq = log(y) - two_log_a;
  const Scalar error_term
      = wiener5_compute_error_term<Scalar>(y, a, v_value, w_value, sv);
  const Scalar density_part_one
      = wiener5_density_part_one<false, false, Scalar>(y, a, v_value, w_value,
                                                       sv);
  const Scalar error = (err - error_term);

  const Scalar n_terms_small_t
      = wiener5_n_terms_small_t<false, true, Scalar>(y, a, w_value, error);
  const Scalar n_terms_large_t
      = wiener5_n_terms_large_t<false, true, Scalar>(y, a, w_value, error);
  Scalar result;
  int newsign;
  std::forward_as_tuple(result, newsign)
      = wiener5_log_sum_exp<false, true, Scalar>(y, a, w_value, n_terms_small_t,
                                                 n_terms_large_t);

  const Scalar log_density = wiener5_density<false, Scalar>(
      y, a, v_value, w_value, sv, err - log(fabs(density_part_one)));
  Scalar ans;
  if (2 * n_terms_small_t < n_terms_large_t) {
    ans = -(density_part_one
            - newsign
                  * exp(result - (log_density - error_term) - 2.5 * log_y_asq
                        - 0.5 * LOG_TWO - 0.5 * LOG_PI));
  } else {
    ans = -(density_part_one
            + newsign * exp(result - (log_density - error_term) + 2 * LOG_PI));
  }
  return WrtLog ? ans * exp(log_density) : ans;
}


/**
 * Calculate the derivative of the wiener5 density w.r.t. 'sv'
 *
 * @tparam WrtLog Whether to return the derivative w.r.t.
 *                  the natural (true) or log-scale (false) density
 * @tparam Scalar type of scalars
 *
 * @param y A scalar variable; the reaction time in seconds
 * @param a The boundary separation
 * @param v_value The drift rate
 * @param w_value The relative starting point
 * @param sv The inter-trial variability of the drift rate
 * @param err The log error tolerance
 * @return Gradient w.r.t. sv
 */
template <bool WrtLog = false, typename Scalar, typename T_y, typename T_a,
          typename T_w, typename T_v, typename T_sv>
inline Scalar wiener5_grad_sv(const T_y& y, const T_a& a, const T_v& v_value,
                              const T_w& w_value, const T_sv& sv,
                              Scalar err = log(1e-12)) noexcept {
  const Scalar one_plus_svsqr_y = 1 + square(sv) * y;
  const Scalar w = 1.0 - w_value;
  const Scalar v = -v_value;
  const Scalar t1 = -y / one_plus_svsqr_y;
  const Scalar t2 = (square(a * w) + 2 * a * v * w * y + square(v * y))
                    / square(one_plus_svsqr_y);
  const Scalar ans = sv * (t1 + t2);
  return WrtLog ? ans
                      * wiener5_density<true, Scalar>(y, a, v_value, w_value,
                                                      sv, err)
                : ans;
}

/**
 * Utility function for replacing a value with a specified error value
 */
template <size_t NestedIndex, typename Scalar>
inline void assign_err(Scalar arg, Scalar err) {
  arg = err;
}

/**
 * Utility function for replacing a value with a specified error value,
 * overload for use when the value is stored within a tuple.
 */
template <size_t NestedIndex, typename Scalar, typename... TArgs>
inline void assign_err(std::tuple<TArgs...>& args_tuple, Scalar err) {
  std::get<NestedIndex>(args_tuple) = err;
}

/**
 * Utility function for estimating a function with a given set of arguments,
 * checking the result against a provided error tolerance, and re-estimating
 * the function with the increased error if it fails.
 *
 * @tparam Scalar type of scalars
 * @tparam ErrIndex Position of the error argument in the provided arguments
 * @tparam NestedIndex Nested position if the error argument is within a tuple
 * @tparam F Type of functor
 * @tparam ArgsTupleT Type of tuple of arguments for functor
 *
 * @param functor Function to apply
 * @param err Error value to check against
 * @param args_tuple Tuple of arguments to pass to functor
 * @param log_result Whether the function result is already on the log-scale
 */
template <typename Scalar, size_t ErrIndex, bool GradW7 = false,
          size_t NestedIndex = 0, bool LogResult = true, typename F,
          typename... ArgsTupleT>
Scalar estimate_with_err_check(const F& functor, Scalar err,
                               ArgsTupleT&&... args_tuple) {
  Scalar result = functor(args_tuple...);
  Scalar log_fabs_result = LogResult ? log(fabs(result)) : fabs(result);
  if (log_fabs_result < err) {
    log_fabs_result = is_inf(log_fabs_result) ? 0 : log_fabs_result;
    auto err_args_tuple = std::make_tuple(args_tuple...);
    const Scalar new_error
        = GradW7 ? err + log_fabs_result + LOG_TWO : err + log_fabs_result;
    assign_err<NestedIndex>(std::get<ErrIndex>(err_args_tuple), new_error);
    result = math::apply([&](auto&&... args) { return functor(args...); },
                         err_args_tuple);
  }
  return result;
}
}  // namespace internal

/**
 * Log-density function for the 5-parameter Wiener density.
 * See 'wiener_full_lpdf' for more comprehensive documentation
 *
 * @tparam T_y type of scalar
 * @tparam T_a type of boundary
 * @tparam T_t0 type of non-decision time
 * @tparam T_w type of relative starting point
 * @tparam T_v type of drift rate
 * @tparam T_sv type of inter-trial variability of drift rate
 *
 * @param y A scalar variable; the reaction time in seconds
 * @param a The boundary separation
 * @param t0 The non-decision time
 * @param w The relative starting point
 * @param v The drift rate
 * @param sv The inter-trial variability of the drift rate
 * @param precision_derivatives Level of precision in estimation
 * @return The log of the Wiener first passage time density with
 *  the specified arguments for upper boundary responses
 */
template <bool propto = false, typename T_y, typename T_a, typename T_t0,
          typename T_w, typename T_v, typename T_sv,
          typename ReturnT = return_type_t<T_y, T_a, T_t0, T_w, T_v, T_sv>>
inline return_type_t<T_y, T_a, T_t0, T_w, T_v, T_sv> wiener5_lpdf(
    const T_y& y, const T_a& a, const T_t0& t0, const T_w& w, const T_v& v,
    const T_sv& sv, const ReturnT& precision_derivatives) {
  using T_partials_return = partials_return_t<T_y, T_a, T_t0, T_w, T_v, T_sv>;

  if (!include_summand<propto, T_y, T_a, T_t0, T_w, T_v, T_sv>::value) {
    return 0;
  }

  using T_y_ref = ref_type_if_t<!is_constant<T_y>::value, T_y>;
  using T_a_ref = ref_type_if_t<!is_constant<T_a>::value, T_a>;
  using T_t0_ref = ref_type_if_t<!is_constant<T_t0>::value, T_t0>;
  using T_w_ref = ref_type_if_t<!is_constant<T_w>::value, T_w>;
  using T_v_ref = ref_type_if_t<!is_constant<T_v>::value, T_v>;
  using T_sv_ref = ref_type_if_t<!is_constant<T_sv>::value, T_sv>;

  static constexpr const char* function_name = "wiener5_lpdf";

  check_consistent_sizes(function_name, "Random variable", y,
                         "Boundary separation", a, "Drift rate", v,
                         "A-priori bias", w, "Nondecision time", t0,
                         "Inter-trial variability in drift rate", sv);

  T_y_ref y_ref = y;
  T_a_ref a_ref = a;
  T_t0_ref t0_ref = t0;
  T_w_ref w_ref = w;
  T_v_ref v_ref = v;
  T_sv_ref sv_ref = sv;

  decltype(auto) y_val = to_ref(as_value_column_array_or_scalar(y_ref));
  decltype(auto) a_val = to_ref(as_value_column_array_or_scalar(a_ref));
  decltype(auto) v_val = to_ref(as_value_column_array_or_scalar(v_ref));
  decltype(auto) w_val = to_ref(as_value_column_array_or_scalar(w_ref));
  decltype(auto) t0_val = to_ref(as_value_column_array_or_scalar(t0_ref));
  decltype(auto) sv_val = to_ref(as_value_column_array_or_scalar(sv_ref));
  check_positive_finite(function_name, "Random variable", y_val);
  check_positive_finite(function_name, "Boundary separation", a_val);
  check_finite(function_name, "Drift rate", v_val);
  check_less(function_name, "A-priori bias", w_val, 1);
  check_greater(function_name, "A-priori bias", w_val, 0);
  check_nonnegative(function_name, "Nondecision time", t0_val);
  check_finite(function_name, "Nondecision time", t0_val);
  check_nonnegative(function_name, "Inter-trial variability in drift rate",
                    sv_val);
  check_finite(function_name, "Inter-trial variability in drift rate", sv_val);

  if (size_zero(y, a, t0, w, v, sv)) {
    return 0;
  }
  const size_t N = max_size(y, a, t0, w, v, sv);
  if (!N) {
    return 0;
  }

  scalar_seq_view<T_y_ref> y_vec(y_ref);
  scalar_seq_view<T_a_ref> a_vec(a_ref);
  scalar_seq_view<T_t0_ref> t0_vec(t0_ref);
  scalar_seq_view<T_w_ref> w_vec(w_ref);
  scalar_seq_view<T_v_ref> v_vec(v_ref);
  scalar_seq_view<T_sv_ref> sv_vec(sv_ref);
  const size_t N_y_t0 = max_size(y, t0);

  for (size_t i = 0; i < N_y_t0; ++i) {
    if (y_vec[i] <= t0_vec[i]) {
      std::stringstream msg;
      msg << ", but must be greater than nondecision time = " << t0_vec[i];
      std::string msg_str(msg.str());
      throw_domain_error(function_name, "Random variable", y_vec[i], " = ",
                         msg_str.c_str());
    }
  }

  const T_partials_return log_error_density = log(1e-6);
  const T_partials_return log_error_derivative = log(precision_derivatives);
  const T_partials_return log_error_absolute = log(1e-12);
  T_partials_return log_density = 0.0;
  auto ops_partials
      = make_partials_propagator(y_ref, a_ref, t0_ref, w_ref, v_ref, sv_ref);

  static constexpr double LOG_FOUR = LOG_TWO + LOG_TWO;

  // calculate density and partials
  for (size_t i = 0; i < N; i++) {
    // Calculate 4-parameter model without inter-trial variabilities (if
    // sv_vec[i] == 0) or 5-parameter model with inter-trial variability in
    // drift rate (if sv_vec[i] != 0)
	
    const T_partials_return y_value = y_vec.val(i);
    const T_partials_return a_value = a_vec.val(i);
    const T_partials_return t0_value = t0_vec.val(i);
    const T_partials_return w_value = w_vec.val(i);
    const T_partials_return v_value = v_vec.val(i);
    const T_partials_return sv_value = sv_vec.val(i);
   
    T_partials_return l_density = internal::estimate_with_err_check<
        T_partials_return, 5, false, 0, false>(
        [&](auto&&... args) {
          return internal::wiener5_density<false, T_partials_return>(args...);
        },
        log_error_density - LOG_TWO, y_value - t0_value, a_value, v_value,
        w_value, sv_value, log_error_absolute);

    log_density += l_density;

    const T_partials_return new_est_err
        = l_density + log_error_derivative - LOG_FOUR;

    // computation of derivative for t and precision check in order to give
    // the value as deriv_y to edge1 and as -deriv_y to edge5
    const T_partials_return deriv_y = internal::estimate_with_err_check<
        T_partials_return, 5, false, 0, true>(
        [&](auto&&... args) {
          return internal::wiener5_grad_t<false, T_partials_return>(args...);
        },
        new_est_err, y_value - t0_value, a_value, v_value, w_value, sv_value,
        log_error_absolute);

    // computation of derivatives and precision checks
    if (!is_constant_all<T_y>::value) {
      partials<0>(ops_partials)[i] = deriv_y;
    }
    if (!is_constant_all<T_a>::value) {
      partials<1>(ops_partials)[i] = internal::estimate_with_err_check<
          T_partials_return, 5, false, 0, true>(
          [&](auto&&... args) {
            return internal::wiener5_grad_a<false, T_partials_return>(args...);
          },
          new_est_err, y_value - t0_value, a_value, v_value, w_value, sv_value,
          log_error_absolute);
    }
    if (!is_constant_all<T_t0>::value) {
      partials<2>(ops_partials)[i] = -deriv_y;
    }
    if (!is_constant_all<T_w>::value) {
      partials<3>(ops_partials)[i] = internal::estimate_with_err_check<
          T_partials_return, 5, false, 0, true>(
          [&](auto&&... args) {
            return internal::wiener5_grad_w<false, T_partials_return>(args...);
          },
          new_est_err, y_value - t0_value, a_value, v_value, w_value, sv_value,
          log_error_absolute);
    }
    if (!is_constant_all<T_v>::value) {
      partials<4>(ops_partials)[i]
          = internal::wiener5_grad_v<false, T_partials_return>(
              y_value - t0_value, a_value, v_value, w_value, sv_value);
    }
    if (!is_constant_all<T_sv>::value) {
      partials<5>(ops_partials)[i]
          = internal::wiener5_grad_sv<false, T_partials_return>(
              y_value - t0_value, a_value, v_value, w_value, sv_value);
    }
  }  // end for loop
  return ops_partials.build(log_density);
}  // end wiener5_lpdf
}  // namespace math
}  // namespace stan
#endif
