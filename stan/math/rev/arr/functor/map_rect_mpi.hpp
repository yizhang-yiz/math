#pragma once

#include <vector>

//#include <stan/math/stan/math/prim/mat/fun/to_array_1d.hpp>
#include <stan/math/prim/arr/functor/mpi_command.hpp>

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
      void
      jacobian(const F& f,
               const std::vector<double>& theta,
               const std::vector<double>& x_r,
               const std::vector<int>& x_i,
               std::vector<double>& FJx) {
        const size_t M = theta.size();

        start_nested();
        try {
          std::vector<var> theta_v(theta.begin(), theta.end());
          std::vector<var> fx_v = f(theta_v, x_r, x_i);

          const std::size_t N = fx_v.size();
          FJx.resize(N * (M+1));

          // FJx is filled with the function value followed by its
          // partials which is repeated for all outputs
      
          for (std::size_t i = 0; i != N; ++i) {
            FJx[i*(M+1)] = fx_v[i].val();
            set_zero_all_adjoints_nested();
            grad(fx_v[i].vi_);
            for (std::size_t k = 0; k != M; ++k)
              FJx[i*(M+1) + k + 1] = theta_v[k].adj();
          }
        } catch (const std::exception& e) {
          recover_memory_nested();
          throw;
        }
        recover_memory_nested();
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
          using std::vector;

          const std::size_t R = world.rank();
      
          // # number of workers
          const std::size_t W = world.size();

          vector<int> sizes(4);
          boost::mpi::broadcast(world, sizes.data(), 4, 0);

          const std::size_t N = sizes[0];
          const std::size_t T = sizes[1];
          const std::size_t X_r = sizes[2];
          const std::size_t X_i = sizes[3];

          const vector<int> chunks = map_chunks(N, W, 1);
          const vector<int> chunks_theta = map_chunks(N, W, T);
          const vector<int> chunks_x_r = map_chunks(N, W, X_r);
          const vector<int> chunks_x_i = map_chunks(N, W, X_i);

          // # of jobs for this chunk
          const std::size_t C = chunks[R];

          // reserve space for local version of flattened arrays
          vector<double> local_theta(chunks_theta[R]);
          vector<double> local_x_r(chunks_x_r[R]);
          vector<int> local_x_i(chunks_x_i[R]);

          // recieve from root using scatterv
          boost::mpi::scatterv(world, local_theta.data(), local_theta.size(), 0);
          if(X_r > 0)
            boost::mpi::scatterv(world, local_x_r.data(), local_x_r.size(), 0);
          if(X_r > 0)
            boost::mpi::scatterv(world, local_x_i.data(), local_x_i.size(), 0);

          // calculate jacobian for each parameter set and collect
          // results in a big array at the very end

          std::cout << "Job " << R << " got " << local_theta.size() << " parameters, " << local_x_r.size() << " real data, " << local_x_i.size() << " int data." << std::endl;

          vector<double> local_result;
          // reserve output size for 5 function outputs per job... we
          // will find out during evaluation
          local_result.reserve(C*5*(1+T));

          // number of outputs per job
          vector<int> local_F_out(C, 0);
          
          try {
            for(std::size_t i = 0; i != C; i++) {
              const vector<double> theta_run(local_theta.begin() + i * T, local_theta.begin() + (i+1) * T);
              const vector<double> x_r_run(local_x_r.begin() + i * X_r, local_x_r.begin() + (i+1) * X_r);
              const vector<int> x_i_run(local_x_i.begin() + i * X_i, local_x_i.begin() + (i+1) * X_i);
              vector<double> FJx;
              jacobian(F(), theta_run, x_r_run, x_i_run, FJx);
              local_F_out[i] = FJx.size() / (T+1);
              local_result.insert(local_result.end(), FJx.begin(), FJx.end());
            }
          } catch(const std::exception& e) {
            // we only abort further processing. The root node will
            // detect unfinished runs and throw. Flag failure.
            local_F_out = vector<int>(C, 0);
            local_F_out[0] = -1;
            // we can discard all data
            local_result.resize(0);
          }
          // sent results to root
          boost::mpi::gatherv(world, local_F_out.data(), C, 0);

          // send results
          boost::mpi::gatherv(world, local_result.data(), local_result.size(), 0);
          
        }
      };
      
    }

    /* an example user functor in Stan could be */
    // real[] map_rect(F f, real[,] theta, real[,] x_r, int[,] x_i);

    template <typename F>
    std::vector<var>
    map_rect_mpi(const F& f,
                 const std::vector<std::vector<var> >& theta,
                 const std::vector<std::vector<double> >& x_r,
                 const std::vector<std::vector<int> >& x_i) {
      boost::mpi::communicator world;
      using std::vector;

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
      const std::vector<int> chunks_theta = internal::map_chunks(N, W, T);
      const std::vector<int> chunks_x_r = internal::map_chunks(N, W, X_r);
      const std::vector<int> chunks_x_i = internal::map_chunks(N, W, X_i);

      // # of jobs for this chunk
      const std::size_t C = chunks[R];

      // create flat array versions of inputs to be sent out. This
      // allows to use plain C versions of MPI_scatterv

      std::vector<double> world_theta;
      world_theta.reserve(N*T);
      std::vector<double> world_x_r;
      world_theta.reserve(N*X_r);
      std::vector<int> world_x_i;
      world_theta.reserve(N*X_i);

      // use stan math functions for flattening?
      for(std::size_t n = 0; n != N; ++n) {
        std::vector<double> theta_n_d = stan::math::value_of(theta[n]);
        world_theta.insert(world_theta.end(), theta_n_d.begin(), theta_n_d.end());
        world_x_r.insert(world_x_r.end(), x_r[n].begin(), x_r[n].end());
        world_x_i.insert(world_x_i.end(), x_i[n].begin(), x_i[n].end());
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

      vector<double> local_result;
      // reserve output size for 5 function outputs per job... we
      // will find out during evaluation
      local_result.reserve(C*5*(1+T));

      // number of outputs per job
      vector<int> local_F_out(C, 0);
          
      try {
        for(std::size_t i = 0; i != C; i++) {
          const vector<double> theta_run(local_theta.begin() + i * T, local_theta.begin() + (i+1) * T);
          const vector<double> x_r_run(local_x_r.begin() + i * X_r, local_x_r.begin() + (i+1) * X_r);
          const vector<int> x_i_run(local_x_i.begin() + i * X_i, local_x_i.begin() + (i+1) * X_i);
          vector<double> FJx;
          internal::jacobian(F(), theta_run, x_r_run, x_i_run, FJx);
          local_F_out[i] = FJx.size() / (T+1);
          local_result.insert(local_result.end(), FJx.begin(), FJx.end());
        }
      } catch(const std::exception& e) {
        // we only abort further processing. The root node will
        // detect unfinished runs and throw. Flag failure.
        local_F_out = vector<int>(C, 0);
        local_F_out[0] = -1;
        // we can discard all data
        local_result.resize(0);
      }
      // collect results at root
      vector<int> world_F_out(N, 0);
      std::cout << "gathering output sizes..." << std::endl;
      boost::mpi::gatherv(world, local_F_out.data(), C, world_F_out.data(), chunks, 0);

      // find out cumulative size of output
      std::size_t F_out_sum = 0;
      bool all_ok = true;
      for(std::size_t i=0; i != N; ++i) {
        if(world_F_out[i] == -1) {
          all_ok = false;
          world_F_out[i] = 0; // set to 0 to get a correct sum below
        } else {
          F_out_sum += world_F_out[i];
        }
      }
      
      // allocate memory on AD stack for final result. Note that the
      // gradients and the function results will land there
      double* final_result
        = ChainableStack::memalloc_.alloc_array<double>( (T + 1) * F_out_sum );

      vector<int> chunks_result(W,0);
      for(std::size_t i=0, ij=0; i != W; ++i)
        for(std::size_t j=0; j != chunks[i]; ++j, ++ij)
          chunks_result[i] += world_F_out[ij] * (T + 1);

      std::cout << "F_out_sum = " << F_out_sum << std::endl;
      
      std::cout << "world_F_out = ";
      for(std::size_t i=0; i != N; ++i)
        std::cout << world_F_out[i] << ", ";
      std::cout << std::endl;
      
      std::cout << "chunks_result = ";
      for(std::size_t i=0; i != W; ++i)
        std::cout << chunks_result[i] << ", ";
      std::cout << std::endl;


        // collect results
      std::cout << "gathering actual outputs..." << std::endl;
      boost::mpi::gatherv(world, local_result.data(), local_result.size(), final_result, chunks_result, 0);

      // now we can throw if necessary as all workers have finished
      if(unlikely(!all_ok))
        throw std::runtime_error("MPI error");

      // insert results into the AD tree
      vari** varis
        = ChainableStack::memalloc_.alloc_array<vari*>(N*T);

      std::vector<var> res;
      res.reserve(F_out_sum);
    
      for(std::size_t i=0, ik=0; i != N; i++) {
        // link the operands...
        for(std::size_t j=0; j != T; j++)
          varis[i * T + j] = theta[i][j].vi_;

        // ...with partials of outputs
        for(std::size_t k=0; k != world_F_out[i]; k++, ik++) {
          const double val = *(final_result + (T+1) * ik);
          res.push_back(var(new precomputed_gradients_vari(val, T, varis + i * T, final_result + (T+1) * ik + 1)));
        }
      }
      
      return(res);
    }

  }
}
