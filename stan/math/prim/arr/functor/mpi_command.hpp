
#include <boost/serialization/serialization.hpp>

/* define virtual class which gets send around by mpi. The run method
   contains the actual work to be executed.
 */

struct mpi_command {
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {}
  virtual void run() const = 0;
};


BOOST_SERIALIZATION_ASSUME_ABSTRACT( mpi_command );

