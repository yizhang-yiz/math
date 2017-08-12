#pragma once

#include <boost/mpi.hpp>
#include <stan/math/prim/arr/functor/mpi_command.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/shared_ptr.hpp>

namespace stan {
  namespace math {

    struct mpi_worker {
      mpi_worker() {
        boost::mpi::communicator world;
        
        if(world.rank() != 0) {
          std::cout << "Worker " << world.rank() << " waiting for commands..." << std::endl;
          while(1) {
            boost::shared_ptr<mpi_command> work;
            
            boost::mpi::broadcast(world, work, 0);

            work->run();
          }
        }
      }
    };

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

  }
}

BOOST_CLASS_EXPORT(stan::math::stop_worker);

