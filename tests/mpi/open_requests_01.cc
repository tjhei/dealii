// ---------------------------------------------------------------------
//
// Copyright (C) 2019 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------


// check MPI_InitializeFinalize::register_static_request


#include <deal.II/base/mpi.h>

#include "../tests.h"

void
test(MPI_Comm comm)
{
  static MPI_Request request = MPI_REQUEST_NULL;
  Utilities::MPI::MPI_InitFinalize::register_static_request(request);
  int ierr = MPI_Wait(&request, MPI_STATUS_IGNORE);
  AssertThrowMPI(ierr);

  int        tag     = 12345;
  const auto my_rank = Utilities::MPI::this_mpi_process(comm);
  const auto n_ranks = Utilities::MPI::n_mpi_processes(comm);

  if (my_rank == 0)
    {
      std::set<int> received_from;
      MPI_Status    status;

      for (unsigned int n = 1; n < n_ranks; ++n)
        {
          unsigned int value;
          MPI_Recv(&value, 1, MPI_UNSIGNED, MPI_ANY_SOURCE, tag, comm, &status);

          AssertThrow(received_from.count(status.MPI_SOURCE) == 1,
                      ExcMessage("oh no!"));
          received_from.insert(status.MPI_SOURCE);
        }
    }
  else
    {
      unsigned int value = 123;
      int          dest  = 0;
      MPI_Send(&value, 1, MPI_UNSIGNED, dest, tag, comm);
    }

  ierr = MPI_Ibarrier(comm, &request);
  AssertThrowMPI(ierr);
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  mpi_initlog();

  test(MPI_COMM_WORLD);
  test(MPI_COMM_WORLD);
  test(MPI_COMM_WORLD);
}
