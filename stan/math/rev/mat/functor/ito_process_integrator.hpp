#ifndef STAN_MATH_PRIM_MAT_FUNCTOR_ITO_PROCESS_INTEGRATOR_HPP
#define STAN_MATH_PRIM_MAT_FUNCTOR_ITO_PROCESS_INTEGRATOR_HPP

#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {
namespace math {

  /*
   * Tamed Euler scheme from
   *
   * Hutzenthaler, M., Jentzen, A., and Kloeden, P. E. Strong convergence of an explicit
   * numerical method for SDEs with non-globally Lipschitz continuous coefficients. Ann. Appl.
   * Probab. 22, 4 (2012), 1611–1641.
   *
   */
  template<typename F1, typename F2, typename T1, typename T2>
  struct ito_process_tamed_euler_stepper {
    const F1& f1;
    const F2& f2;    
    const std::vector<T1> & theta1;
    const std::vector<T2> & theta2;

    using param_t = typename stan::return_type<T1, T2>::type;

    ito_process_tamed_euler_stepper(const F1& f1_in, const F2& f2_in,
                                    const std::vector<T1>& theta1_in,
                                    const std::vector<T2>& theta2_in) :
      f1(f1_in), f2(f2_in), theta1(theta1_in), theta2(theta2_in)
    {}

    template<typename T0, typename Tw>
    inline Eigen::Matrix<typename stan::return_type<T0, Tw, param_t>::type, -1, 1>
    operator()(double h,
               const Eigen::Matrix<T0, -1, 1>& y,
               const Eigen::Matrix<Tw, -1, 1>& w) const {
      using drift_t = typename stan::return_type<T0, T1>::type;
      using diffu_t = typename stan::return_type<T0, Tw, T2>::type;
      const Eigen::Matrix<drift_t, -1, 1> drift = f1(y, theta1);
      const Eigen::Matrix<diffu_t, -1, 1> diffu = f2(y, theta2) * sqrt(h) * w;
      return y + h * drift / (1.0 + h * stan::math::sqrt(drift.squaredNorm())) + diffu;
    }
  };


  struct ito_process {

    ito_process() {}

    /*
     * Apply stepper @c S to a constant step size, given
     * initial condition @c y0 and Wiener process @c w.
     */
    template<typename T0, typename Tw, typename S>
    inline Eigen::Matrix<typename stan::return_type<T0, Tw, typename S::param_t>::type, -1, -1>
    operator()(const S& stepper,
               const Eigen::Matrix<T0, -1, 1>& y0,
               const Eigen::Matrix<Tw, -1, -1>& w,
               double t) const {
      using scalar_t = typename stan::return_type<T0, Tw, typename S::param_t>::type;

      Eigen::Matrix<scalar_t, -1, -1> res(y0.size(), w.cols());
      int n = w.cols();
      const double h = t / n;
      Eigen::Matrix<T0, -1, 1> yv;
      Eigen::Matrix<Tw, -1, 1> wv = w.col(0);
      res.col(0) = stepper(h, y0, wv);
      for (int i = 1; i < n; ++i) {
        yv = res.col(i - 1);
        wv = w.col(i);
        res.col(i) = stepper(h, yv, wv);
      }

      return res;
    }
  };

  template<typename F1, typename F2, typename T0, typename Tw, typename T1, typename T2>
  inline Eigen::Matrix<typename stan::return_type<T0, Tw, T1, T2>::type, -1, -1>  
  ito_process_euler(const F1& f1, const F2& f2,
                    const Eigen::Matrix<T0, -1, 1>& y0,
                    const Eigen::Matrix<Tw, -1, -1>& w,
                    const std::vector<T1>& theta1,
                    const std::vector<T2>& theta2,
                    double t) {
    ito_process_tamed_euler_stepper<F1, F2, T1, T2> stepper(f1, f2, theta1, theta2);
    ito_process p;
    return p(stepper, y0, w, t);
  }
}
}

#endif
