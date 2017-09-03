#pragma once

#include <vector>
#include <set>

#include <stan/math/prim/mat/fun/to_array_1d.hpp>
#include <stan/math/prim/arr/functor/mpi_command.hpp>

namespace stan {
  namespace math {

    namespace internal {
      // forward declare
      void
      distribute_map_rect_data(const std::vector<std::vector<double> >& x_r,
                               const std::vector<std::vector<int> >& x_i,
                               const std::size_t uid);


      // data distribution needs to go to prim
      struct distributed_map_rect_data {
        distributed_map_rect_data(std::size_t uid) : uid_(uid), N_(-1) {}
        std::size_t uid_;
        std::size_t N_;
        std::vector<std::vector<double> > x_r_;
        std::vector<std::vector<int> > x_i_;
        static void distributed_apply() {
          // called on workers to register data
          distribute_map_rect_data(std::vector<std::vector<double> >(), std::vector<std::vector<int> >(), 0);
        }
      };

      typedef boost::serialization::singleton<std::map<std::size_t,const distributed_map_rect_data> > distributed_data;

      void
      distribute_map_rect_data(const std::vector<std::vector<double> >& x_r,
                               const std::vector<std::vector<int> >& x_i,
                               const std::size_t uid) {
        
        boost::mpi::communicator world;
        
        const std::size_t W = world.size();
        const std::size_t R = world.rank();

        // first check if uid is already registered
        if(distributed_data::get_const_instance().find(uid) == distributed_data::get_const_instance().cend()) {
          //std::cout << "Job " << R << " registers data..." << std::endl;
        } else {
          //std::cout << "UID = " << uid << " is ALREADY distributed." << std::endl;
          return;
        }

        std::vector<int> meta(4, 0);
        
        if(R == 0) {
          // initiate on the root call of this function on the workers
          mpi_cluster::broadcast_command<stan::math::distributed_apply<distributed_map_rect_data> >();

          meta[0] = uid;
          meta[1] = x_r.size();
          meta[2] = x_r[0].size();
          meta[3] = x_i[0].size();
        }

        boost::mpi::broadcast(world, meta.data(), 4, 0);

        distributed_map_rect_data data(meta[0]);

        const std::size_t N = meta[1];
        const std::size_t X_r = meta[2];
        const std::size_t X_i = meta[3];

        data.N_ = N;

        //std::cout << "worker " << R << " / " << W << " registers shapes " << N << ", " << X_r << ", " << X_i << std::endl;

        const std::vector<int> chunks = mpi_cluster::map_chunks(N, 1);
        
        data.x_r_.resize(chunks[R]);
        data.x_i_.resize(chunks[R]);

           // flatten data and send out/recieve using scatterv
        if(X_r > 0) {
          const std::vector<double> world_x_r = to_array_1d(x_r);
          const std::vector<int> chunks_x_r = mpi_cluster::map_chunks(N, X_r);
          std::vector<double> flat_x_r_local(chunks_x_r[R]);

          boost::mpi::scatterv(world, world_x_r.data(), chunks_x_r, flat_x_r_local.data(), 0);

          // now register data
          for(std::size_t i = 0; i != chunks[R]; ++i)
            data.x_r_[i] = std::vector<double>(flat_x_r_local.begin() + i * X_r, flat_x_r_local.begin() + (i+1) * X_r);
          
          //std::cout << "Job " << R << " got " << flat_x_r_local.size() << " real data " << std::endl;
        }
        if(X_i > 0) {
          const std::vector<int> world_x_i = to_array_1d(x_i);
          const std::vector<int> chunks_x_i = mpi_cluster::map_chunks(N, X_i);
          std::vector<int> flat_x_i_local(chunks_x_i[R]);

          boost::mpi::scatterv(world, world_x_i.data(), chunks_x_i, flat_x_i_local.data(), 0);

          // now register data
          for(std::size_t i = 0; i != chunks[R]; ++i)
            data.x_i_[i] = std::vector<int>(flat_x_i_local.begin() + i * X_i, flat_x_i_local.begin() + (i+1) * X_i);
          
          //std::cout << "Job " << R << " got " << flat_x_i_local.size() << " int data " << std::endl;
        }

        distributed_data::get_mutable_instance().insert(std::make_pair(uid, data));

        //std::cout << "Job " << R << " done caching data for uid = " << meta[0] << "." << std::endl;
        
        return;
      }
    }

    template <typename F>
    std::vector<double>
    map_rect_mpi(const std::vector<double>& eta,
                 const std::vector<std::vector<double> >& theta,
                 const std::vector<std::vector<double> >& x_r,
                 const std::vector<std::vector<int> >& x_i,
                 const std::size_t uid) {

      std::vector<double> res;
      const std::size_t N = theta.size();

      for(std::size_t i = 0; i != N; ++i) {
        const std::vector<double> f = F::apply(eta, theta[i], x_r[i], x_i[i]);
        res.insert(res.end(), f.begin(), f.end());
      }

      return(res);
    }

  }
}
