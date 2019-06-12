#ifndef STAN_MATH_PRIM_MAT_FUNCTOR_ITO_PROCESS_INTEGRATOR_HPP
#define STAN_MATH_PRIM_MAT_FUNCTOR_ITO_PROCESS_INTEGRATOR_HPP

#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {
namespace math {

  /*
   * Stepper for tamed Euler scheme, based on
   *
   * Hutzenthaler, M., Jentzen, A., and Kloeden, P. E. Strong convergence of an explicit
   * numerical method for SDEs with non-globally Lipschitz continuous coefficients. Ann. Appl.
   * Probab. 22, 4 (2012), 1611–1641.
   *
   * @tparam F1 functor type for autonomous drift function mu
   * @tparam F2 functor type for autonomous diffusion delta
   * @tparam T1 type of the parameters that @c F1 depends on
   * @tparam T2 type of the parameters that @c F2 depends on
   *
   */
  template<typename F1, typename F2, typename T1, typename T2>
  struct ito_process_tamed_euler_stepper {
    const F1& f1;
    const F2& f2;    
    const std::vector<T1> & theta1;
    const std::vector<T2> & theta2;

    using param_t = typename stan::return_type<T1, T2>::type;

    /*
     * Constructor
     *
     * @param f1_in drift function
     * @param f2_in diffusion function
     * @param theta1_in drift function parameters
     * @param theta2_in diffusion function parameters
     */
    ito_process_tamed_euler_stepper(const F1& f1_in, const F2& f2_in,
                                    const std::vector<T1>& theta1_in,
                                    const std::vector<T2>& theta2_in) :
      f1(f1_in), f2(f2_in), theta1(theta1_in), theta2(theta2_in)
    {}

    /*
     * stepping operation that move forward one time step
     * with size @c h, returning numerical solution at that step.
     *
     * @tparam T0 current solution type.
     * @tparam Tw type for iid R.Vs with standard normal
     *            distribution that serve each step of the Wiener
     *            process is based upon.
     * @param h time step size
     * @param y current numerical solution
     * @param w vector of values from iid random variables
     *          with normal distribution. For such a value @c w, 
     *          Wiener process increment is calculated as @c sqrt(h)*w.
     * @return numerical solution at next time step.
     */
    template<typename T0, typename Derived>
    inline Eigen::Matrix<typename stan::return_type<T0, typename Derived::Scalar, param_t>::type, -1, 1> // NOLINT
    operator()(double h,
               const Eigen::Matrix<T0, -1, 1>& y,
               const Eigen::MatrixBase<Derived>& w) const {
      const Eigen::Matrix<typename stan::return_type<T0, T1>::type, -1,  1> drift = f1(y, theta1); // NOLINT
      const Eigen::Matrix<typename stan::return_type<T0, T2>::type, -1, -1> diffu = f2(y, theta2); // NOLINT
      stan::math::check_size_match("Ito process", "diffusion", diffu.cols(), "standard normal", w.size()); // NOLINT
      Eigen::Matrix<typename stan::return_type<T0, typename Derived::Scalar, param_t>::type, -1, 1> res(y); // NOLINT
      res += h * drift / (1.0 + h * stan::math::sqrt(drift.squaredNorm()));
      res += sqrt(h) * diffu * w;
      return res;
    }
  };


  /*
   * Generation of an Ito process as numerical solution of SDEs.
   */
  struct ito_process {
    /*
     * default constructor.
     */
    ito_process() {}

    /*
     * Apply stepper @c S to a sequence with constant step size, given
     * initial condition @c y0 and Wiener process @c w.
     *
     * @tparam T0 type of initial condition
     * @tparam Tw type for iid R.Vs with standard normal
     *            distribution that serve each step of the Wiener
     *            process is based upon. 
     *
     * @param stepper numerical stepper.
     * @param y0 initial condition.
     * @param w Matrix of iid R.Vs for Wiener process, with
     *          size nr x nc, and the number of cols(nc)
     *          defines the number of points of the process to be returned.
     * @param t final time to be reached through @c stepper.
     * @return solution matrix with size nr x nc.
     *         Solution i, i=0...nc-1 is at time @c t/nc*(i+1).
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
      Eigen::Matrix<scalar_t, -1, 1> yv(y0.size());
      res.col(0) = stepper(h, y0, w.col(0));
      for (int i = 1; i < n; ++i) {
        yv = res.col(i - 1);
        res.col(i) = stepper(h, yv, w.col(i));
      }

      return res;
    }
  };

  /*
   * Ito process by Euler scheme @c ito_process_tamed_euler_stepper
   *
   * @tparam F1 functor type for autonomous drift function mu.
   * @tparam F2 functor type for autonomous diffusion delta.
   * @tparam T0 current solution type.
   * @tparam Tw type for iid R.Vs with standard normal
   *            distribution that serve each step of the Wiener
   *            process is based upon.
   * @tparam T1 type of the parameters that @c F1 depends on.
   * @tparam T2 type of the parameters that @c F2 depends on.
   * @param f1 drift function
   * @param f2 diffusion function, its return must be left-multiplicable
   *           to @c w
   * @param y0 initial condition.
   * @param w Matrix of iid R.Vs for Wiener process, with
   *          size nr x nc, and the number of cols(nc)
   *          defines the number of points of the process to be returned.
   * @param theta1 drift function parameters
   * @param theta2 diffusion function parameters
   * @param t final time to be reached through @c stepper.
   * @return solution matrix with size nr x nc.
   *         Solution i, i=0...nc-1 is at time @c t/nc*(i+1).
   */
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
