// ---------------------------------------------------------------------
//
// Copyright (C) 2003 - 2020 by the deal.II authors
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



// check GeometryInfo::alternating_form_at_vertices

#include <deal.II/base/geometry_info.h>

#include "../tests.h"



template <int dim>
void
test()
{
  deallog << "Checking in " << dim << "d" << std::endl;

  // check the determinant of the
  // transformation for the reference
  // cell. the determinant should be one in
  // that case
  {
    Point<dim> vertices[GeometryInfo<dim>::vertices_per_cell];
    for (const unsigned int v : GeometryInfo<dim>::vertex_indices())
      vertices[v] = GeometryInfo<dim>::unit_cell_vertex(v);

    Tensor<0, dim> determinants[GeometryInfo<dim>::vertices_per_cell];
    GeometryInfo<dim>::alternating_form_at_vertices(vertices, determinants);
    for (const unsigned int v : GeometryInfo<dim>::vertex_indices())
      {
        deallog << "Reference cell: " << determinants[v] << std::endl;
        AssertThrow(static_cast<double>(determinants[v]) == 1,
                    ExcInternalError());
      }
  }

  // try the same, but move squash the cell
  // in the x-direction by a factor of 10
  {
    Point<dim> vertices[GeometryInfo<dim>::vertices_per_cell];
    for (const unsigned int v : GeometryInfo<dim>::vertex_indices())
      {
        vertices[v] = GeometryInfo<dim>::unit_cell_vertex(v);
        vertices[v][0] /= 10;
      }

    Tensor<0, dim> determinants[GeometryInfo<dim>::vertices_per_cell];
    GeometryInfo<dim>::alternating_form_at_vertices(vertices, determinants);
    for (const unsigned int v : GeometryInfo<dim>::vertex_indices())
      {
        deallog << "Squashed cell: " << determinants[v] << std::endl;
        AssertThrow(static_cast<double>(determinants[v]) == 0.1,
                    ExcInternalError());
      }
  }


  // try the same, but move squash the cell
  // in the x-direction by a factor of 10 and
  // rotate it around the z-axis (unless in
  // 1d)
  {
    Point<dim> vertices[GeometryInfo<dim>::vertices_per_cell];
    for (const unsigned int v : GeometryInfo<dim>::vertex_indices())
      {
        vertices[v] = GeometryInfo<dim>::unit_cell_vertex(v);
        vertices[v][0] /= 10;

        if (dim > 1)
          {
            std::swap(vertices[v][0], vertices[v][1]);
            vertices[v][1] *= -1;
          }
      }

    Tensor<0, dim> determinants[GeometryInfo<dim>::vertices_per_cell];
    GeometryInfo<dim>::alternating_form_at_vertices(vertices, determinants);
    for (const unsigned int v : GeometryInfo<dim>::vertex_indices())
      {
        deallog << "Squashed+rotated cell: " << determinants[v] << std::endl;
        AssertThrow(static_cast<double>(determinants[v]) == 0.1,
                    ExcInternalError());
      }
  }

  // pinched cell
  {
    Point<dim> vertices[GeometryInfo<dim>::vertices_per_cell];
    for (const unsigned int v : GeometryInfo<dim>::vertex_indices())
      vertices[v] = GeometryInfo<dim>::unit_cell_vertex(v);
    vertices[1] /= 10;

    Tensor<0, dim> determinants[GeometryInfo<dim>::vertices_per_cell];
    GeometryInfo<dim>::alternating_form_at_vertices(vertices, determinants);
    for (const unsigned int v : GeometryInfo<dim>::vertex_indices())
      deallog << "Pinched cell: " << determinants[v] << std::endl;
  }


  // inverted cell
  {
    Point<dim> vertices[GeometryInfo<dim>::vertices_per_cell];
    for (const unsigned int v : GeometryInfo<dim>::vertex_indices())
      vertices[v] = GeometryInfo<dim>::unit_cell_vertex(v);
    std::swap(vertices[0], vertices[1]);

    Tensor<0, dim> determinants[GeometryInfo<dim>::vertices_per_cell];
    GeometryInfo<dim>::alternating_form_at_vertices(vertices, determinants);
    for (const unsigned int v : GeometryInfo<dim>::vertex_indices())
      deallog << "Inverted cell: " << determinants[v] << std::endl;
  }
}


int
main()
{
  initlog();

  test<1>();
  test<2>();
  test<3>();

  return 0;
}
