#ifndef STAN_MATH_PRIM_ARR_FUN_LINEAR_INTERPOLATION_HPP
#define STAN_MATH_PRIM_ARR_FUN_LINEAR_INTERPOLATION_HPP

#include <vector>

namespace stan {
  namespace math {

    int min(int a, int b);  // forward declare

    /**
     * Returns the position of the largest value smaller or greater than srchNum
     * Assumes that v is sorted.
     * The numltm is used to set an upper limit for the index
     * the function can return. If numltm is greater than the size of v,
     * there is no upper limit and the function searches the entire vector.
     *
     * @tparam: type of scalar in input vector
     * @param[in]: v searched vector
     * @param[in]: numltm maximum index
     * @param[in]: srchNum searched Number
     * @return: index of largest value <= srchNum
     *
     */
    template<typename T>
    inline int SearchReal(std::vector<T> v, int numltm, T srchNum) {
      int first = 0, last, mid, real_limit;

      // limit cannot exceed size of searched vector
      real_limit = min(numltm, v.size());
      last = real_limit - 1;

      if (srchNum < v[first]) mid = first;
      else
	if (srchNum >= v[last]) {
	  mid = last + 1;
	} else {
	  while (first <= last) {
	    mid = (first + last) / 2;
	    if (srchNum < v[mid]) last = mid - 1;
	    else if (srchNum > v[mid]) first = mid + 1;
	    else
	      first = last + 1;
	  }

	  while (srchNum >= v[mid]) mid += 1;
	}
      return mid;
    }

    inline int min(int a, int b) {
      if (a < b) return a;
      else
	return b;
    }

    template <typename T0, typename T1>
    typename boost::math::tools::promote_args <T0, T1>::type
    linear_interpolation1(const T0& xout,
			  const std::vector<double>& x,
			  const std::vector<T1>& y){
      typedef typename boost::math::tools::promote_args <T0, T1>::type scalar;
      using std::vector;
      int nx = x.size();
      scalar yout;

      if(xout <= x[0]){
	yout = y[0];
      }else if(xout >= x[nx - 1]){
	yout = y[nx - 1];
      }else{
	int i = SearchReal(x, nx, xout) - 1;
	yout = y[i] + (y[i+1] - y[i]) / (x[i+1] - x[i]) * (xout - x[i]);
      }

      return yout;
    }
    
    template <typename T0, typename T1>
    std::vector <typename boost::math::tools::promote_args <T0, T1>::type>
    linear_interpolation(const std::vector<T0>& xout,
			 const std::vector<double>& x,
			 const std::vector<T1>& y){
      typedef typename boost::math::tools::promote_args <T0, T1>::type scalar;
      using std::vector;

      int nx = x.size();
      vector<scalar> yout(nx);

      for(int i = 0; i < nx; i++){
	yout[i] = linear_interpolation1(xout[i], x, y);
      }
      return yout;
    }
   
  }
}
#endif
