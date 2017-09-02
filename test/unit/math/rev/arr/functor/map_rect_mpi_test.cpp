#include <stan/math.hpp>
#include <stan/math/prim/arr/functor/mpi_command.hpp>
#include <stan/math/prim/arr/functor/mpi_cluster.hpp>
#include <stan/math/rev/arr/functor/map_rect_mpi.hpp>
#include <boost/mpi.hpp>
#include <boost/shared_ptr.hpp>

#include <iostream>

struct hard_work {
  template<typename T>
  std::vector<T> operator()(std::vector<T> eta, std::vector<T> theta, std::vector<double> x_r, std::vector<int> x_i) const {
    std::vector<T> res(2);
    res[0] = theta[0]*theta[0];
    res[1] = x_r[0]*theta[1]*theta[0];
    return(res);
  }

  template<typename T>
  static
  std::vector<T> apply(std::vector<T> eta, std::vector<T> theta, std::vector<double> x_r, std::vector<int> x_i) {
    const hard_work f;
    return f(eta, theta, x_r, x_i);
  }
};

BOOST_CLASS_EXPORT(stan::math::distributed_apply<stan::math::internal::distributed_map_rect_data>);
BOOST_CLASS_TRACKING(stan::math::distributed_apply<stan::math::internal::distributed_map_rect_data>,track_never);

BOOST_CLASS_EXPORT(stan::math::distributed_apply<stan::math::internal::distributed_map_rect<hard_work> >);
BOOST_CLASS_TRACKING(stan::math::distributed_apply<stan::math::internal::distributed_map_rect<hard_work> >,track_never);


int main(int argc, const char* argv[]) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  using std::vector;
  
  // on non-root processes this makes the workers listen to commands
  // send from the root
  stan::math::mpi_cluster cluster;

  std::cout << "Root process starts distributing work..." << std::endl;

  // create a task to be distributed
  vector<stan::math::var > eta = {2, 0};
  vector<double> eta_d = stan::math::value_of(eta);
  vector<vector<stan::math::var> > theta;
  vector<vector<double> > theta_d;

  const std::size_t N = 10;

  for(std::size_t n = 0; n != N; ++n) {
    vector<stan::math::var> theta_run(2);
    theta_run[0] = n;
    theta_run[1] = n*n;
    theta.push_back(theta_run);
    theta_d.push_back(stan::math::value_of(theta_run));
  }

  vector<vector<double> > x_r(N, vector<double>(1,1.0));
  vector<vector<int> > x_i(N, vector<int>(0));

  std::cout << "Distributing the data to the nodes..." << std::endl;
  
  const std::size_t uid = 0;
  vector<stan::math::var> res = stan::math::map_rect_mpi<hard_work>(eta, theta, x_r, x_i, uid);
  
  std::cout << "Executing with cached data for uid = " << uid << std::endl;

  vector<stan::math::var> res2 = stan::math::map_rect_mpi<hard_work>(eta, theta, x_r, x_i, uid);

  std::cout << "run things as double only locally..." << std::endl;
  vector<double> res3 = stan::math::map_rect_mpi<hard_work>(eta_d, theta_d, x_r, x_i, uid);

  std::cout << "Root process ends." << std::endl;
  
  return(0);
}
