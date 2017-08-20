#pragma once

#include <boost/mpi.hpp>
#include <stan/math/prim/arr/functor/mpi_command.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/shared_ptr.hpp>

namespace stan {
  namespace math {

    // MPI command which will shut down a child gracefully
    struct stop_worker : public mpi_command {
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version) {
        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(mpi_command);
      }
      void run() const {
        boost::mpi::communicator world;
        std::cout << "Terminating worker " << world.rank() << std::endl;
        MPI_Finalize();
        std::exit(0);
      }
    };

    template<typename T>
    struct distributed_apply : public mpi_command {
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version) {
        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(mpi_command);
      }
      void run() const {
        T::distributed_apply();
      }
    };

    struct mpi_cluster {
      boost::mpi::communicator world_;
      std::size_t const R_ = world_.rank();
      
      mpi_cluster() {
        if(R_ != 0) {
          std::cout << "Worker " << R_ << " waiting for commands..." << std::endl;
          while(1) {
            boost::shared_ptr<mpi_command> work;
            
            boost::mpi::broadcast(world_, work, 0);

            work->run();
          }
        }
      }

      ~mpi_cluster() {
        // the destructor will ensure that the childs are being
        // shutdown
        if(R_ == 0) {
          broadcast_command<stop_worker>();
        }
      }

      template<typename T>
      static void broadcast_command() {
        boost::mpi::communicator world;
        if(world.rank() != 0)
          throw std::runtime_error("only root may broadcast commands.");

          boost::shared_ptr<mpi_command> command(new T);

          boost::mpi::broadcast(world, command, 0);
      }

      // map N chunks to W with a chunks-size of C; used for
      // deterministic scheduling
      static
      std::vector<int>
      map_chunks(std::size_t N, std::size_t C = 1) {
        boost::mpi::communicator world;
        const std::size_t W = world.size();
        
        std::vector<int> chunks(W, N / W);
      
        for(std::size_t r = 0; r != N % W; r++)
          ++chunks[r+1];

        for(std::size_t i = 0; i != W; i++)
          chunks[i] *= C;

        return(chunks);
      }

    };

  }
}

BOOST_CLASS_EXPORT(stan::math::stop_worker);

