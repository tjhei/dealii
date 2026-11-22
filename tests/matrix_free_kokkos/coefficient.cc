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

// Check both Portable::MatrixFree coefficient layouts in a device cell loop.

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/matrix_free/coefficient.h>
#include <deal.II/matrix_free/portable_matrix_free.templates.h>

#include <deal.II/numerics/vector_tools.h>

#include <Kokkos_Core.hpp>

#include "../tests.h"

using namespace MatrixFreeTools;

constexpr int dim = 2;

DEAL_II_HOST_DEVICE double
exact_value(const Point<dim> &p)
{
  return 1. + p[0] + 3. * p[1];
}

template <CoefficientLayout layout>
struct CheckCoefficient
{
  static constexpr unsigned int n_q_points = 1U << dim;
  using Result =
    Kokkos::View<int *[n_q_points], MemorySpace::Default::kokkos_space>;

  DeviceCoefficient<double, layout> from_field;
  DeviceCoefficient<double, layout> from_function;
  Result                            correct;

  DEAL_II_HOST_DEVICE void
  operator()(const Portable::MatrixFree<dim, double>::Data *data,
             const Portable::DeviceVector<double> &,
             Portable::DeviceVector<double> &) const
  {
    const unsigned int cell = data->cell_index;
    Point<dim>         center;
    if constexpr (layout == CoefficientLayout::cell)
      {
        // The average of the Gauss points is the center of each rectangle.
        for (unsigned int q = 0; q < n_q_points; ++q)
          center += data->get_quadrature_point(cell, q);
        center /= n_q_points;
      }

    data->for_each_quad_point([&](const unsigned int q) {
      const Point<dim> point    = layout == CoefficientLayout::cell ?
                                    center :
                                    data->get_quadrature_point(cell, q);
      const double     expected = exact_value(point);
      correct(cell, q) = Kokkos::abs(from_field(cell, q) - expected) < 1e-12 &&
                         Kokkos::abs(from_function(cell, q) - expected) < 1e-12;
    });
  }
};

template <CoefficientLayout layout>
void
check_coefficients(const Portable::MatrixFree<dim, double> &matrix_free,
                   const Coefficient<dim>                  &from_field,
                   const Coefficient<dim>                  &from_function)
{
  AssertThrow(from_field.get_layout() == layout &&
                from_function.get_layout() == layout,
              ExcInternalError());
  const auto n_cells =
    matrix_free.get_dof_handler().get_triangulation().n_active_cells();
  using Kernel = CheckCoefficient<layout>;
  typename Kernel::Result correct("coefficient checks", n_cells);
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> src, dst;
  matrix_free.initialize_dof_vector(src);
  matrix_free.initialize_dof_vector(dst);
  matrix_free.cell_loop(Kernel{from_field.portable_view<layout>(),
                               from_function.portable_view<layout>(),
                               correct},
                        src,
                        dst);

  const auto host =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), correct);
  for (unsigned int cell = 0; cell < n_cells; ++cell)
    for (unsigned int q = 0; q < Kernel::n_q_points; ++q)
      AssertThrow(host(cell, q), ExcMessage("Incorrect coefficient value."));
}

template <CoefficientLayout layout>
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

  const ScalarFunctionFromFunctionObject<dim> function(exact_value);
  LinearAlgebra::distributed::Vector<double>  field(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler, function, field);

  const MappingQ1<dim>      mapping;
  AffineConstraints<double> constraints;
  constraints.close();
  using Backend = Portable::MatrixFree<dim, double>;
  typename Backend::AdditionalData additional_data;
  additional_data.mapping_update_flags |=
    update_quadrature_points | update_JxW_values;
  additional_data.use_coloring = true;
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

  Coefficient<dim> from_field, from_function;
  if constexpr (layout == CoefficientLayout::cell)
    {
      from_field.reinit(matrix_free, dof_handler, field);
      from_function.reinit(matrix_free, function);
    }
  else
    {
      // The continuous field exercises numbering across several colors.
      AssertThrow(matrix_free.get_colored_graph().size() > 1,
                  ExcInternalError());
      const QGauss<dim> quadrature(2);
      from_field.reinit(matrix_free, mapping, dof_handler, field, quadrature);
      from_function.reinit(matrix_free, mapping, function, quadrature);
    }

  check_coefficients<layout>(matrix_free, from_field, from_function);
  const auto transferred_field =
    interpolate_cellwise_coefficients_to_mg(level_data, from_field);
  const auto transferred_function =
    interpolate_cellwise_coefficients_to_mg(level_data, from_function);
  for (unsigned int level = 0; level <= 1; ++level)
    check_coefficients<CoefficientLayout::cell>(*level_data[level],
                                                transferred_field[level],
                                                transferred_function[level]);
}

int
main()
{
  initlog();
  const Kokkos::ScopeGuard kokkos;

  test<CoefficientLayout::cell>(FE_DGQ<dim>(0));
  test<CoefficientLayout::quadrature>(FE_Q<dim>(1));
  deallog << "OK" << std::endl;
}
