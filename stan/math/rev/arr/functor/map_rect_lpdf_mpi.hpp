#pragma once

#include <vector>
#include <set>

#include <boost/mpi/operations.hpp>

#include <stan/math/prim/mat/fun/to_array_1d.hpp>
#include <stan/math/prim/arr/functor/mpi_command.hpp>
#include <stan/math/prim/arr/functor/map_rect_mpi.hpp>
//#include <stan/math/rev/arr/functor/map_rect_mpi.hpp>

namespace stan {
  namespace math {

    namespace internal {
      
      template <typename F>
      class distributed_map_rect_lpdf {
        boost::mpi::communicator world_;
        const std::size_t R_ = world_.rank();
        const std::size_t W_ = world_.size();

        // local version of things in flattened out form
        std::vector<double> local_eta_;
        std::vector<double> local_theta_;
        std::size_t uid_;

        // note: it would be nice to have these const, but I think C++
        // doesn't let me given how it has to be initialized
        std::size_t N_; // # of jobs for world
        std::size_t E_; // # of shared parameters
        std::size_t T_; // # of parameters
        
        // # of outputs per job
        double *final_result_ = 0;

        // the parameters which can be updated; only stored at the
        // root
        const std::vector<var>* eta_;
        typedef const std::vector<std::vector<var> > param_t;
        param_t* theta_;

        std::vector<int> chunks_;

        // # of jobs for this chunk
        std::size_t C_;

      public:
        // called on root, sents problem sizes, data and parameters
        distributed_map_rect_lpdf(const std::vector<var>& eta,
                                  const std::vector<std::vector<var> >& theta,
                                  const std::vector<std::vector<double> >& x_r,
                                  const std::vector<std::vector<int> >& x_i,
                                  const std::size_t uid)
          : uid_(uid), N_(theta.size()), E_(eta.size()), T_(theta[0].size()) {
          //std::cout << "Setting up distributed_map on root " << world_.rank() << " / " << W_ << std::endl;
          if(R_ != 0)
            throw std::runtime_error("problem sizes can only defined on the root.");

          // checks if the data is already cached
          distribute_map_rect_data(x_r, x_i, uid);

          //std::cout << "root uses UID = " << uid_ << std::endl;

          // make childs aware of upcoming job
          mpi_broadcast_command<stan::math::mpi_distributed_apply<distributed_map_rect_lpdf<F> > >();

          //std::cout << "setting up root with uid = " << uid << std::endl;
          setup(uid);
          
          // sent uid and # of params
          std::vector<int> meta(3, 0);
          meta[0] = uid;
          meta[1] = E_ ;
          meta[2] = T_ ;
          boost::mpi::broadcast(world_, meta.data(), 3, 0);
          
          distribute_param(eta, theta);

          (*this)();
        }

        distributed_map_rect_lpdf() {
          //std::cout << "Setting up distributed map on worker " << world_.rank() << " / " << W_ << std::endl;
          if(R_ == 0)
            throw std::runtime_error("problem sizes must be defined on the root.");

          //std::cout << "setting up child ..." << std::endl;
          
          // get uid & # of params from root
          std::vector<int> meta(3);
          boost::mpi::broadcast(world_, meta.data(), 3, 0);
          uid_ = meta[0];
          E_ = meta[1];
          T_ = meta[2];

          //std::cout << "worker " << world_.rank() << " / " << W_ << " uses UID = " << uid_ << std::endl;

          setup(uid_);
          
          distribute_param(std::vector<var>(), std::vector<std::vector<var> >());

          (*this)();
        }
        
        static void distributed_apply() {
          // entry point when called on remote

          // call default constructor
          distributed_map_rect_lpdf<F> job_chunk;
        }

        // process work chunks; collect results at root
        void operator()() {
          using std::vector;

          const distributed_map_rect_data& local_ = distributed_data::get_const_instance().find(uid_)->second;

          // holds accumulated terms: function output and summed
          // partials of shared parameters
          //vector<double> local_result_sum(1+E_,0);
          
          // holds partials of per job parameters and the summed values on this host
          vector<double> local_result(1+E_,0); 
          
          // reserve output size
          local_result.reserve(1+E_+C_*T_);

          bool all_ok = true;

          try {
            vector<double>::const_iterator local_theta_iter = local_theta_.begin();
            vector<double> grad(E_+T_);
            for(std::size_t i = 0; i != C_; i++) {
              start_nested();

              const vector<var> eta_run_v(local_eta_.begin(), local_eta_.end());
              // note: on the root node we could avoid re-creating
              // these theta var instances
              const vector<var> theta_run_v(local_theta_iter, local_theta_iter + T_);

              vector<var> z_vars;
              z_vars.reserve(E_+T_);
              z_vars.insert(z_vars.end(),   eta_run_v.begin(),   eta_run_v.end());
              z_vars.insert(z_vars.end(), theta_run_v.begin(), theta_run_v.end());

              var fx_v = F::apply(eta_run_v, theta_run_v, local_.x_r_[i], local_.x_i_[i]);
              
              fx_v.grad(z_vars, grad);

              local_result[0] += fx_v.val();
              for (std::size_t j = 0; j != E_; ++j) {
                local_result[1+j] += grad[j];
              }

              local_result.insert(local_result.end(), grad.begin() + E_, grad.end());

              local_theta_iter += T_;
              recover_memory_nested();
            }
          } catch(const std::exception& e) {
            recover_memory_nested();
            // we only abort further processing. The root node will
            // detect unfinished runs and throw. Flag failure.
            all_ok = false;
          }
          // collect results at root

          vector<int> chunks_result = mpi_map_chunks(N_, T_);
          int world_size = 0;
          for(std::size_t i=0; i != W_; i++) {
            chunks_result[i] += 1+E_;
            world_size += chunks_result[i];
          }
          // we only allocate any memory on the root
          std::vector<double> world_temp(world_size); // holds all results on root
          // which still need some
          // accumulation

          /* mpi reduce appears to cause lots of latency, avoid
          // collect results, first call a MPI reduce on the results
          // which are directly accumulated
          //std::cout << "gathering actual outputs..." << std::endl;
          boost::mpi::reduce(world_, local_result_sum.data(), local_result_sum.size(), final_result_, std::plus<double>(), 0);

          // next we collect the remaining individual partials
          vector<int> chunks_result = mpi_map_chunks(N_, T_);
          boost::mpi::gatherv(world_, local_result.data(), local_result.size(), final_result_ + 1 + E_, chunks_result, 0);
          */

          // next we collect the remaining individual partials
          boost::mpi::gatherv(world_, local_result.data(), local_result.size(), world_temp.data(), chunks_result, 0);

          if(R_ == 0) {
            // finish on root accumulation and put results onto right place
            // allocate memory on AD stack for final result. Note that the
            // gradients and the function results will land there
            final_result_
              = ChainableStack::memalloc_.alloc_array<double>( 1 + E_ + N_ * T_ );
            std::fill_n(final_result_, 1+E_, 0);
            std::size_t p = 0; // gets increased by the chunk size from each node
            std::size_t t = 0; // counts number of partials already processes
            for(std::size_t i=0; i != W_; i++) {
              for(std::size_t j=0; j != 1+E_; j++)
                final_result_[j] += world_temp[p+j];
              std::size_t tc = chunks_result[i] - 1 - E_; // # of partials from this chunk
              std::copy_n(world_temp.begin() + p + 1 + E_, tc, final_result_ + 1 + E_ + t);
              t += tc;
              p += chunks_result[i];
            }
          }

          // now we can throw on the root if necessary as all workers have finished
          if(unlikely(R_ == 0 & !all_ok))
            throw std::runtime_error("MPI error");

          //std::cout << "results have been send!" << std::endl;
        }

        // called only on the root to register results in the AD stack
        // and setup a the results var vector
        var register_results() {
          if(R_ != 0)
              throw std::runtime_error("results must be registered on root only.");

          //std::cout << "registering results" << std::endl;
          
          // insert results into the AD tree
          vari** varis
            = ChainableStack::memalloc_.alloc_array<vari*>(E_+N_*T_);

          // link the operands...
          for(std::size_t i=0; i != E_; i++)
            varis[i] = (*eta_)[i].vi_;
          
          for(std::size_t i=0; i != N_; i++) {
            for(std::size_t j=0; j != T_; j++)
              varis[E_ + i * T_ + j] = (*theta_)[i][j].vi_;
          }
      
          //std::cout << "results are registered" << std::endl;
          //... to the partials
          const double val = *(final_result_);
          var res(new precomputed_gradients_vari(val, E_+N_*T_, varis, final_result_ + 1));
          
          return(res);
        }

        std::size_t get_uid() const {
          return uid_;
        }

      private:
        // sent a new set of parameters out
        void distribute_param(const std::vector<var>& eta, const std::vector<std::vector<var> >& theta) {
          if(R_ == 0) {
            if(theta.size() != N_)
              throw std::runtime_error("problem size mismatch.");
            if(theta[0].size() != T_)
              throw std::runtime_error("parameter size mismatch.");

            eta_ = &eta;
            theta_ = &theta;
          }

          // sent shared parameters to all workers
          if(E_ > 0) {
            local_eta_.resize(E_);
            if(R_ == 0)
              local_eta_ = value_of(eta);
            boost::mpi::broadcast(world_, local_eta_.data(), E_, 0);
          }
          
          if(T_ > 0) {
            // flatten and send out/recieve using scatterv
            std::vector<double> world_theta;
            world_theta.reserve(theta.size()*T_);
          
            for(std::size_t n = 0; n != theta.size(); ++n) {
              const std::vector<double> theta_n_d = stan::math::value_of(theta[n]);
              world_theta.insert(world_theta.end(), theta_n_d.begin(), theta_n_d.end());
            }
          
            const std::vector<int> chunks_theta = mpi_map_chunks(N_, T_);
            local_theta_.resize(chunks_theta[R_]);
            boost::mpi::scatterv(world_, world_theta.data(), chunks_theta, local_theta_.data(), 0);
            //std::cout << "Job " << R_ << " got " <<
            //local_theta_.size() << " parameters " << std::endl;
          }

        }

        void setup(std::size_t uid) {
           // grab data
          //local_ = distributed_data::get_mutable_instance().find(uid)->second;
          const distributed_map_rect_data& local_ = distributed_data::get_const_instance().find(uid)->second;

          // copy over sizes, etc.
          N_ = local_.N_;

          chunks_ = mpi_map_chunks(N_, 1);
          C_ = chunks_[R_];

          //std::cout << "worker " << world_.rank() << " / " << W_ << " got shapes " << N_ << ", " << T_ << ", " << X_i_ << ", " << X_r_ << std::endl;
          
        }
      };     
      
      }

    /* an example user functor in Stan could be */
    // real map_rect(F f, real[] eta, real[,] theta, real[,] x_r,
    // int[,] x_i);
    // and the user function f has the signature
    // real f(real[] eta, real[] theta, real[] x_r, int[] x_i)

    // this version accumulates all output values which allows to
    // accumulate the partials of the shared parameters as well
    // locally on each worker.

    template <typename F>
    var
    map_rect_lpdf_mpi(const std::vector<var>& eta,
                      const std::vector<std::vector<var> >& theta,
                      const std::vector<std::vector<double> >& x_r,
                      const std::vector<std::vector<int> >& x_i,
                      const std::size_t uid) {

      internal::distributed_map_rect_lpdf<F> root_job_chunk(eta, theta, x_r, x_i, uid);

      return(root_job_chunk.register_results());
    }
      
  }
}
