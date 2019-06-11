#include <stan/math.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/mat/functor/ito_process_integrator.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>

using stan::math::ito_process_euler;
using stan::math::var;

struct linear_1d_sde {
  template<typename Ty, typename T>
  inline Eigen::Matrix<typename stan::return_type<Ty, T>::type, -1, 1>
  operator()(const Eigen::Matrix<Ty, -1, 1>& y, const std::vector<T>& theta) const {
    return theta[0] * y;
  }
};

TEST(ito_process_euler_test, linear_1d) {
  linear_1d_sde f1;
  linear_1d_sde f2;

  const size_t n = 100;
  Eigen::MatrixXd step_normal(1, n);
  step_normal <<
  -0.227031346,  0.458041443, -1.579591365,  1.175445492,  0.119544588,
   0.558492074,  1.105970751, -1.436829855,  0.342849252, -1.345981060,
  -0.550210348,  1.197567836,  0.051984390, -0.605358174,  0.734544128,
   0.353114406,  0.019629616,  0.562807970, -1.465477762,  0.678149017,
  -0.171619676,  1.175385076,  1.519844138, -0.346590383, -0.745657647,
  -0.328375998,  0.068063898, -1.783454633,  0.013063462,  0.714949560,
  -1.114476575,  0.006569428,  0.498795072, -0.082497795, -0.570758809,
   1.277299748,  0.240810730,  2.110826049,  0.330361802, -0.300154027,
  -1.020465365, -0.163852017, -1.993294496,  0.447473730,  0.010716684,
  -0.491290787, -0.568813674,  0.068327523,  0.784916059,  1.552803020,
   2.768448761,  0.588521589, -1.263617390,  0.118946927, -0.891862091,
   0.287182207, -0.490645341, -0.849732427, -0.151875223, -0.270740843,
  -0.254553642, -2.021715555,  1.291047437, -0.895050476, -0.869481380,
   1.705331080,  1.046698757, -0.829974183, -0.677389072,  0.187091578,
   1.111322910,  0.783582215,  1.290011615, -0.043900397,  0.272410938,
   2.348848679, -0.467825001,  2.209460577,  0.032898713,  1.205695917,
  -1.374362365, -0.091768263,  0.070804504, -0.509105393,  0.065652878,
   0.898239917,  0.067963672,  0.424453990,  0.731980187,  1.830465840,
  -0.990680249, -1.982484822,  0.055027686, -0.783264035,  0.793723338,
  -0.275728723,  1.267190326, -0.500147232, -1.001674061,  1.287339032;

  const std::vector<double> mu{2.0};
  const std::vector<double> sigma{1.0};
  Eigen::Matrix<double, -1, 1> y0(1);
  y0 << 1.0;
  
  const double t_end = 0.1;
  Eigen::MatrixXd y = ito_process_euler(f1, f2, y0, step_normal, mu, sigma, t_end);

  // exact solution
  const double h = t_end / double(n);
  double y_exact;
  
  double wiener = 0.0;
  for (int i = 0; i < n; ++i) {
    wiener += sqrt(h) * step_normal(i);
    y_exact = y0[0] * stan::math::exp((mu[0] - 0.5 * sigma[0] * sigma[0] ) * i * h + sigma[0] * wiener);
    EXPECT_NEAR(y(i), y_exact, 7.5e-3);
  }
}

/*
 * Stochastic SIR model by
 * 
 * Tornatore, E., Buccellato, S. M., and Vetro, P. Stability of a stochastic SIR system.
 * Physica A: Statistical Mechanics and its Applications 354, 15 (2005), 111–126.
 * 
 */
struct stochastic_sir_drift {
  /*
   * @c theta = {alpha, beta, gamma, delta}
   */
  template<typename Ty, typename T>
  inline Eigen::Matrix<typename stan::return_type<Ty, T>::type, -1, 1>
  operator()(const Eigen::Matrix<Ty, -1, 1>& y, const std::vector<T>& theta) const {
    using scalar_t = typename stan::return_type<Ty, T>::type;
    Eigen::Matrix<scalar_t, -1, 1> res = Eigen::Matrix<scalar_t, -1, 1>::Zero(3);
    const T& a = theta[0];
    const T& c = theta[2];
    const T& d = theta[3];
    if (y(0) > 0 && y(1) > 0 && y(2) > 0) {
      res(0) = -a * y(0) * y(1) - d * y(0) + d;
      res(1) = a * y(0) * y(1) - (c + d) * y(1);
      res(2) = c * y(1) - d * y(2);
    }
    return res;
  }
};

/*
 * Stochastic SIR model by
 * 
 * Tornatore, E., Buccellato, S. M., and Vetro, P. Stability of a stochastic SIR system.
 * Physica A: Statistical Mechanics and its Applications 354, 15 (2005), 111–126.
 * 
 * which shows that the system has disease-free state {1.0, 0.0, 0.0} with
 * globally asymptotically Lyapunov stability.
 *
 */
struct stochastic_sir_diffusion {
  /*
   * @c theta = {alpha, beta, gamma, delta}
   */
  template<typename Ty, typename T>
  inline Eigen::Matrix<typename stan::return_type<Ty, T>::type, -1, 1>
  operator()(const Eigen::Matrix<Ty, -1, 1>& y, const std::vector<T>& theta) const {
    using scalar_t = typename stan::return_type<Ty, T>::type;
    Eigen::Matrix<scalar_t, -1, 1> res = Eigen::Matrix<scalar_t, -1, 1>::Zero(3);
    const T& b = theta[1];
    if (y(0) > 0 && y(1) > 0 && y(2) > 0) {
      res(0) = -b * y(0) * y(1);
      res(1) =  b * y(0) * y(1);
    }
    return res;
  }
};

TEST(ito_process_euler_test, stochastic_SIR) {
  stochastic_sir_drift f1;
  stochastic_sir_diffusion f2;

  const size_t n = 100;
  Eigen::MatrixXd step_normal(1, n);
  step_normal <<
   -1.57897441, -0.35818871,  2.03659328, -0.17945727,  0.35367191, -0.01355063,
   -0.14308078, -0.52589990,  0.62348373, -0.67316549, -0.28270535,  2.20480659,
   -0.36244412, -0.05279088, -1.60581425,  0.99302399,  1.75852801,  0.72132252,
   -0.90468343,  0.71028153,  0.42509134,  0.25898053,  0.93533781, -0.74418412,
   -0.17784473,  0.37595112, -1.78233997,  1.47856031, -1.52410228,  0.06331306,
    0.34682617,  0.02567723,  1.04596529,  0.15888549,  1.05681907, -0.05950151,
    0.09813256, -0.84812529,  0.73265170,  1.24724395,  0.27976836, -0.20927854,
    1.35064100,  0.14255626, -0.77789686, -1.25477575,  0.29630693, -0.36178728,
   -0.67998419, -1.12509306,  0.31940237, -0.22758767, -0.37014759,  0.18617732,
    0.27857130,  0.22387955,  1.77981157, -1.57242888,  0.87122817,  0.29855531,
   -1.04155339, -0.69416972, -2.05525811,  0.98246207, -0.91522730,  0.13531040,
   -1.17055068, -0.02005696,  0.95412267,  0.30238508,  0.62161288,  0.80191187,
    0.52468362, -1.23489973, -0.42102839, -1.27908801, -1.22746536,  0.67593309,
    0.65218103,  0.25098710,  0.33533639, -0.14092592, -1.71517241, -1.91633767,
    0.06401554, -0.15272495,  1.96062817, -0.44781888,  0.37903729, -2.16222328,
   -0.48631562,  0.95008125,  0.15529862, -1.12904730, -2.16615155, -0.21537449,
    1.47482641, -0.05169457,  2.00635913, -0.01466573;
  
  // stable solution, used for test
  const std::vector<var> theta{0.2, 0.15, 0.1, 0.2};

  // less stable solution, takes longer to reach fixed point.
  // const std::vector<var> theta{0.31, 0.20, 0.1, 0.2};

  // unstable solution
  // const std::vector<var> theta{0.50, 0.20, 0.1, 0.2};

  Eigen::Matrix<double, -1, 1> y0(3);
  y0 << 0.975, 0.020, 0.005;

  const double t_end = 100.0;
  Eigen::Matrix<var, -1, -1> y = ito_process_euler(f1, f2, y0, step_normal, theta, theta, t_end);
  EXPECT_NEAR(y(0, n-1).val(), 1.0, 5e-7);
  EXPECT_NEAR(y(1, n-1).val(), 0.0, 1e-7);
  EXPECT_NEAR(y(2, n-1).val(), 0.0, 5e-7);
}
