

#include <vector>

#include <stan/math/prim/functor/mpi_command.hpp>

namespace stan {
  namespace math {

    namespace internal {
      template <typename F>
      struct run_functor : public mpi_command {
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
          ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(mpi_command);
        }
        void run() const {
          boost::mpi::communicator world;

          // recieve parameter chunk using scatterv
          // recieve data x_r and x_i chunks using scatterv

          // calculate jacobian for each parameter set and collect
          // results in a big array at the very end

          double param = world.rank();
          F fun;
          std::cout << "Job " << world.rank() << " says " << fun(param, global_data) << std::endl;

          // send problem size (output size of each function evaluation) via gatherv
          // send results via gatherv
          
        }
      };
    }

    template <typename F>
    std::vector<var>
    map_rect_mpi(const F& f,
                 const std::vector<std::vector<var> >& theta,
                 const std::vector<std::vector<double> >& x_r,
                 const std::vector<std::vector<int> >& x_i) {
      boost::mpi::communicator world;

      // # of jobs
      const std::size_t N = theta.size();
      // check x_r and x_i to match in size to the # of jobs

      // # of parameters
      const std::size_t M = theta[0].size();

      // create fixed map of jobs to workers by equally dividing up
      // the N jobs over the workers

      // make childs aware of upcoming job
      boost::shared_ptr<mpi_command> work(new internal::run_functor<F>);
      boost::mpi::broadcast(world, work, 0);

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

      // done
    }

  }
}
