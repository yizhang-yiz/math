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

struct linear_1d_sde {
  template<typename Ty, typename T>
  inline Eigen::Matrix<typename stan::return_type<Ty, T>::type, -1, 1>
  operator()(const Eigen::Matrix<Ty, -1, 1>& y, const std::vector<T>& theta) const {
    return theta[0] * y;
  }
};

using stan::math::ito_process_euler;
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
