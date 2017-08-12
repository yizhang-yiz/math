

#include <vector>

#include <stan/math/prim/functor/mpi_command.hpp>

namespace stan {
  namespace math {

    namespace internal {

      // map N chunks to W with a chunks-size of C; used for
      // deterministic scheduling
      std::vector<int>
      map_chunks(std::size_t N, std::size_t W, std::size_t C = 1) {
        std::vector<int> chunks(W, N / W);
      
        for(std::size_t r = 0; r != N % W; r++)
          ++chunks[r+1];

        for(std::size_t i = 0; i != W; i++)
          chunks[i] *= C;

        return(chunks);
      }
      
      template <typename F>
      struct run_functor : public mpi_command {
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
          ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(mpi_command);
        }
        void run() const {
          boost::mpi::communicator world;

          // check that we are rank 0
          const std::size_t R = world.rank();
      
          // # number of workers
          const std::size_t W = world.size();

          std::vector<int> sizes(4);
          boost::mpi::broadcast(world, sizes.data(), 4, 0);

          const std::size_t N = sizes[0];
          const std::size_t T = sizes[1];
          const std::size_t X_r = sizes[2];
          const std::size_t X_i = sizes[3];

          const std::vector<int> chunks = map_chunks(N, W, 1);
          const std::vector<int> chunks_theta = map_chunks(N, W, M);
          const std::vector<int> chunks_x_r = map_chunks(N, W, X_r);
          const std::vector<int> chunks_x_i = map_chunks(N, W, X_i);

          // reserve space for local version of flattened arrays
          std::vector<double> local_theta(chunks_theta[R]);
          std::vector<double> local_x_r(chunks_x_r[R]);
          std::vector<int> local_x_i(chunks_x_i[R]);

          // recieve from root using scatterv
          boost::mpi::scatterv(world, local_theta.data(), local_theta.size(), 0);
          if(X_r > 0)
            boost::mpi::scatterv(world, local_x_r.data(), local_x_r.size(), 0);
          if(X_r > 0)
            boost::mpi::scatterv(world, local_x_i.data(), local_x_i.size(), 0);

          // calculate jacobian for each parameter set and collect
          // results in a big array at the very end

          std::cout << "Job " << R << " got " << local_theta.size() << " parameters, " << local_x_r.size() << " real data, " << local_x_i.size() << " int data." << std::endl;
          
        }
      };
      
    }

    /* an example user functor in Stan could be */
    // real[] map_rect(F f, real[,] theta, real[,] x_r, int[,] x_i);

    template <typename F>
    //std::vector<var>
    void
    map_rect_mpi(const F& f,
                 const std::vector<std::vector<var> >& theta,
                 const std::vector<std::vector<double> >& x_r,
                 const std::vector<std::vector<int> >& x_i) {
      boost::mpi::communicator world;

      // check that we are rank 0
      const std::size_t R = world.rank();
      
      // # number of workers
      const std::size_t W = world.size();
      
      // # of jobs
      const std::size_t N = theta.size();
      // check x_r and x_i to match in size to the # of jobs

      // # of parameters
      const std::size_t T = theta[0].size();

      // # of data items
      const std::size_t X_r = x_r[0].size();
      const std::size_t X_i = x_i[0].size();

      std::vector<int> sizes = { N, T, X_r, X_i };

      // create fixed map of jobs to workers by equally dividing up
      // the N jobs over the workers

      // make childs aware of upcoming job
      boost::shared_ptr<mpi_command> work(new internal::run_functor<F>);
      boost::mpi::broadcast(world, work, 0);

      boost::mpi::broadcast(world, sizes.data(), 4, 0);

      // map job chunks to workers (indexed by rank of workers)
      const std::vector<int> chunks = internal::map_chunks(N, W, 1);
      const std::vector<int> chunks_theta = internal::map_chunks(N, W, M);
      const std::vector<int> chunks_x_r = internal::map_chunks(N, W, X_r);
      const std::vector<int> chunks_x_i = internal::map_chunks(N, W, X_i);

      // create flat array versions of inputs to be sent out. This
      // allows to use plain C versions of MPI_scatterv

      std::vector<double> world_theta;
      world_theta.reserve(N*M);
      std::vector<double> world_x_r;
      world_theta.reserve(N*X_r);
      std::vector<int> world_x_i;
      world_theta.reserve(N*X_i);
      
      for(std::size_t n = 0; n != N; ++n) {
        std::vector<double> theta_n_d = stan::math::value_of(theta[n]);
        world_theta.insert(theta_n_d);
        world_x_r.insert(x_r[n]);
        world_x_i.insert(x_i[n]);
      }

      // reserve space for local version of flattened arrays
      std::vector<double> local_theta(chunks_theta[R]);
      std::vector<double> local_x_r(chunks_x_r[R]);
      std::vector<int> local_x_i(chunks_x_i[R]);

      // send out using scatterv
      boost::mpi::scatterv(world, world_theta.data(), chunks_theta, local_theta.data(), 0);
      if(X_r > 0)
        boost::mpi::scatterv(world, world_x_r.data(), chunks_x_r, local_x_r.data(), 0);
      if(X_r > 0)
        boost::mpi::scatterv(world, world_x_i.data(), chunks_x_i, local_x_i.data(), 0);
      
      
      // convert theta into a double version (flat array), sent it out
      // via scatterv

      // convert x_r and x_i into a flat array and sent out via
      // scatterv

      // do work on local chunk
      
      // collect problem sizes from all results using gatherv
      
      // allocate memory on AD stack for final result. Note that the
      // gradients and the function results will land there
      //double* final_result
      //  = ChainableStack::memalloc_.alloc_array<double>( (M + 1) * N_world );

      // collect results from workers using gatherv; write stuff
      // directly to AD memory stack

      // build the AD tree with references to the results on the AD
      // stack using precomputed_gradients_vari

      std::cout << "Job " << R << " got " << local_theta.size() << " parameters, " << local_x_r.size() << " real data, " << local_x_i.size() << " int data." << std::endl;

      // done
    }

  }
}
