#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include <deal.II/base/mpi.h>

int main( int argc, char *argv[] )
{
  using namespace dealii;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  Catch::Session session;
  session.applyCommandLine(argc, argv);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)!=0)
    {
      std::ostringstream ss;
      session.config().setStreamBuf(ss.rdbuf());
    }
  int result = session.run();
  return result;
}
