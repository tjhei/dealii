// ---------------------------------------------------------------------
//
// Copyright (C) 2006 - 2016 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

// Test LevelDataOut for cell and vector data in serial

#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/level_data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/function_lib.h>

#include "../tests.h"

using namespace dealii;

template <int dim>
void do_test (const unsigned int min_level,
              const unsigned int max_level = numbers::invalid_unsigned_int)
{
  FE_Q<dim> fe(1);
  Triangulation<dim> triangulation(Triangulation<dim>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube (triangulation);
  triangulation.refine_global(1);
  triangulation.begin_active()->set_refine_flag();
  triangulation.execute_coarsening_and_refinement();

  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs(fe);

  // Make FE vector
  Vector<double> global_dof_vector(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler, Functions::SquareFunction<dim>(), global_dof_vector);

  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector (dof_handler,
                              global_dof_vector,
                              std::vector<std::string>(1, "bla"));
    data_out.build_patches (0);
    data_out.write_gnuplot(deallog.get_file_stream());
    std::ofstream st("base.vtu");
    data_out.write_vtu(st);

  }

  {
    LevelDataOut<dim> data_out (0, min_level, max_level);
    data_out.attach_dof_handler(dof_handler);

    MGLevelObject<Vector<double> > dof_vector;
    data_out.make_level_dof_data_vector(dof_vector, dof_handler);

    MGTransferPrebuilt<Vector<double>> transfer;
    transfer.build_matrices(dof_handler);
    transfer.copy_to_mg(dof_handler, dof_vector, global_dof_vector);
    data_out.add_data_vector (dof_handler,
                              dof_vector,
                              std::vector<std::string>(1, "bla"));

    MGLevelObject<Vector<double> > cell_data;
    data_out.make_level_cell_data_vector(cell_data);

    const unsigned int n_levels = triangulation.n_levels();
    for (unsigned int lvl = min_level; lvl < n_levels; ++lvl)
      {
        for (unsigned int i=0; i<cell_data[lvl].size(); ++i)
          cell_data[lvl][i] = (double)(100*lvl + i);
      }
    data_out.add_data_vector (cell_data, "some_cell_data");
    data_out.build_patches (0);
    data_out.write_gnuplot(deallog.get_file_stream());
    std::ofstream st("a.vtu");
    data_out.write_vtu(st);
  }
}


int main (int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();

  do_test<2>(0);
  do_test<2>(1,2);/*
  do_test<2>(1,1);

  do_test<3>(1);*/
  return 0;
}
