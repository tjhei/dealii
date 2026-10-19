// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------


// Test parallel::DefaultTriangulation construction and basic mesh operations.

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/default_tria.h>
#include <deal.II/distributed/tria_base.h>

#include <deal.II/grid/grid_generator.h>

#include "../tests.h"


template <int dim>
void
test()
{
  parallel::DefaultTriangulation<dim> triangulation(MPI_COMM_WORLD);

  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(3);

  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned() && cell->center()[0] < 0.3 &&
        cell->center()[1] < 0.3)
      cell->set_refine_flag();

  triangulation.execute_coarsening_and_refinement();

  deallog << "globally active cells: " << triangulation.n_global_active_cells()
          << std::endl;
  deallog << "owned active cells (rank 0): "
          << triangulation.n_locally_owned_active_cells() << std::endl;

#if defined(DEAL_II_WITH_MPI) && !defined(DEAL_II_WITH_P4EST)
  if (const auto *shared_tria =
        dynamic_cast<parallel::shared::Triangulation<dim> *>(&triangulation))
    Assert(shared_tria->with_artificial_cells(), ExcInternalError());
#endif

  if (dynamic_cast<parallel::TriangulationBase<dim> *>(&triangulation) !=
      nullptr)
    deallog << "is parallel triangulation base" << std::endl;
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();

  deallog.push("2d");
  test<2>();
  deallog.pop();
}
