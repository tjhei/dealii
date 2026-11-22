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

// Check both MatrixFree coefficient layouts against an analytic field.

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/matrix_free/coefficient.h>

#include <deal.II/numerics/vector_tools.h>

#include <cmath>

#include "../tests.h"

constexpr int dim = 2;

void
check_coefficients(const MatrixFree<dim, double>           &matrix_free,
                   const MatrixFreeTools::Coefficient<dim> &from_field,
                   const MatrixFreeTools::Coefficient<dim> &from_function,
                   const Function<dim>                     &function,
                   const bool                               cellwise)
{
  FEEvaluation<dim, -1, 0, 1> fe_eval(matrix_free);
  for (unsigned int batch = 0; batch < matrix_free.n_cell_batches(); ++batch)
    {
      fe_eval.reinit(batch);
      for (const unsigned int q : fe_eval.quadrature_point_indices())
        {
          const auto field_values    = from_field.value(fe_eval, q);
          const auto function_values = from_function.value(fe_eval, q);
          const auto points          = fe_eval.quadrature_point(q);
          for (unsigned int lane = 0;
               lane < matrix_free.n_active_entries_per_cell_batch(batch);
               ++lane)
            {
              const Point<dim> point =
                cellwise ?
                  matrix_free.get_cell_iterator(batch, lane)->center() :
                  Point<dim>(points[0][lane], points[1][lane]);
              const double expected = function.value(point);
              AssertThrow(std::abs(field_values[lane] - expected) < 1e-12,
                          ExcMessage(
                            "Incorrect value from a finite element field."));
              AssertThrow(std::abs(function_values[lane] - expected) < 1e-12,
                          ExcMessage("Incorrect value from a function."));
            }
        }
    }
}

void
test(const FiniteElement<dim> &fe)
{
  Triangulation<dim> coarse, fine;
  GridGenerator::subdivided_hyper_cube(coarse, 3);
  fine.copy_triangulation(coarse);
  fine.refine_global(1);
  const auto stretch = [](Point<dim> p) {
    p[0] *= p[0];
    return p;
  };
  // The children have unequal volumes after stretching.
  GridTools::transform(stretch, coarse);
  GridTools::transform(stretch, fine);

  MGLevelObject<DoFHandler<dim>> dof_handlers(0, 1);
  dof_handlers[0].reinit(coarse);
  dof_handlers[1].reinit(fine);
  for (unsigned int level = 0; level <= 1; ++level)
    dof_handlers[level].distribute_dofs(fe);
  const auto &dof_handler = dof_handlers[1];

  const ScalarFunctionFromFunctionObject<dim> function(
    [](const Point<dim> &p) { return 1. + p[0] + 3. * p[1]; });
  LinearAlgebra::distributed::Vector<double> field(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler, function, field);

  const MappingQ1<dim>      mapping;
  AffineConstraints<double> constraints;
  constraints.close();
  using Backend = MatrixFree<dim, double>;
  typename Backend::AdditionalData additional_data;
  additional_data.mapping_update_flags |=
    update_quadrature_points | update_JxW_values;
  MGLevelObject<std::shared_ptr<Backend>> level_data(0, 1);
  for (unsigned int level = 0; level <= 1; ++level)
    {
      level_data[level] = std::make_shared<Backend>();
      level_data[level]->reinit(mapping,
                                dof_handlers[level],
                                constraints,
                                QGauss<1>(2),
                                additional_data);
    }
  const auto &matrix_free = *level_data[1];

  MatrixFreeTools::Coefficient<dim> from_field, from_function;
  const bool                        cellwise = fe.degree == 0;
  if (cellwise)
    {
      from_field.reinit(matrix_free, dof_handler, field);
      from_function.reinit(matrix_free, function);
    }
  else
    {
      const QGauss<dim> quadrature(2);
      from_field.reinit(matrix_free, mapping, dof_handler, field, quadrature);
      from_function.reinit(matrix_free, mapping, function, quadrature);
    }

  check_coefficients(
    matrix_free, from_field, from_function, function, cellwise);
  const auto transferred_field =
    MatrixFreeTools::interpolate_cellwise_coefficients_to_mg(level_data,
                                                             from_field);
  const auto transferred_function =
    MatrixFreeTools::interpolate_cellwise_coefficients_to_mg(level_data,
                                                             from_function);
  for (unsigned int level = 0; level <= 1; ++level)
    {
      AssertThrow(transferred_field[level].get_layout() ==
                      MatrixFreeTools::CoefficientLayout::cell &&
                    transferred_function[level].get_layout() ==
                      MatrixFreeTools::CoefficientLayout::cell,
                  ExcInternalError());
      check_coefficients(*level_data[level],
                         transferred_field[level],
                         transferred_function[level],
                         function,
                         true);
    }
}

int
main()
{
  initlog();
  test(FE_DGQ<dim>(0));
  test(FE_Q<dim>(1));
  deallog << "OK" << std::endl;
}
