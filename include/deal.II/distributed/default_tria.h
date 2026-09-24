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

#ifndef dealii_distributed_default_tria_h
#define dealii_distributed_default_tria_h


#include <deal.II/base/config.h>

#include <deal.II/base/mpi_stub.h>
#include <deal.II/base/template_constraints.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/tria.h>


DEAL_II_NAMESPACE_OPEN

namespace parallel
{
  namespace internal
  {
#if !defined(DEAL_II_WITH_MPI)

    /**
     * The triangulation type selected by parallel::DefaultTriangulation when
     * deal.II is configured without MPI.
     */
    template <int dim, int spacedim = dim>
    using DefaultTriangulationImpl = dealii::Triangulation<dim, spacedim>;

#elif defined(DEAL_II_WITH_P4EST)

    /**
     * The triangulation type selected by parallel::DefaultTriangulation when
     * deal.II is configured with MPI and p4est.
     */
    template <int dim, int spacedim = dim>
    using DefaultTriangulationImpl =
      dealii::parallel::distributed::Triangulation<dim, spacedim>;

#else

    /**
     * The triangulation type selected by parallel::DefaultTriangulation when
     * deal.II is configured with MPI but without p4est.
     */
    template <int dim, int spacedim = dim>
    using DefaultTriangulationImpl =
      dealii::parallel::shared::Triangulation<dim, spacedim>;

#endif
  } // namespace internal



  /**
   * A triangulation type that selects the most capable parallel mesh
   * implementation available in the current build configuration.
   *
   * Depending on how deal.II was configured, this class is derived from
   * @code
   *   dealii::Triangulation<dim, spacedim>
   * @endcode
   * if MPI support is disabled,
   * @code
   *   parallel::distributed::Triangulation<dim, spacedim>
   * @endcode
   * if both MPI and p4est are available, and
   * @code
   *   parallel::shared::Triangulation<dim, spacedim>
   * @endcode
   * if MPI is available but p4est is not. In the last case, artificial cells
   * are enabled by default so that the mesh offers a layer of ghost and
   * artificial cells similar to parallel::distributed::Triangulation.
   *
   * The class provides a uniform constructor that takes an MPI communicator.
   * This allows user code to instantiate a mesh object without selecting the
   * concrete triangulation class at compile time. For features that are only
   * available on particular triangulation classes, such as
   * parallel::distributed::Triangulation::save(), use
   * dynamic_cast to the desired type or include the corresponding header
   * explicitly.
   *
   * @dealiiConceptRequires{(concepts::is_valid_dim_spacedim<dim, spacedim>)}
   */
  template <int dim, int spacedim = dim>
  DEAL_II_CXX20_REQUIRES((concepts::is_valid_dim_spacedim<dim, spacedim>))
  class DefaultTriangulation
    : public internal::DefaultTriangulationImpl<dim, spacedim>
  {
  public:
    using Base = internal::DefaultTriangulationImpl<dim, spacedim>;

    /**
     * Constructor.
     *
     * @param mpi_communicator The MPI communicator over which this
     * triangulation is distributed.
     *
     * @param smooth_grid The mesh smoothing flag passed to the underlying
     * triangulation class.
     *
     * @param check_for_distorted_cells Whether to check for distorted cells
     * upon mesh creation and refinement. This argument is only used by the
     * serial triangulation class selected when MPI support is disabled.
     */
    explicit DefaultTriangulation(
      const MPI_Comm mpi_communicator,
      const typename dealii::Triangulation<dim, spacedim>::MeshSmoothing
        smooth_grid = (dealii::Triangulation<dim, spacedim>::none),
      const bool check_for_distorted_cells = false);
  };


  template <int dim, int spacedim>
  DEAL_II_CXX20_REQUIRES((concepts::is_valid_dim_spacedim<dim, spacedim>))
  DefaultTriangulation<dim, spacedim>::DefaultTriangulation(
    const MPI_Comm mpi_communicator,
    const typename dealii::Triangulation<dim, spacedim>::MeshSmoothing
      smooth_grid,
    const bool check_for_distorted_cells)
#if !defined(DEAL_II_WITH_MPI)
    : Base(mpi_communicator, smooth_grid, check_for_distorted_cells)
#elif defined(DEAL_II_WITH_P4EST)
    : Base(mpi_communicator,
           smooth_grid,
           dealii::parallel::distributed::Triangulation<dim, spacedim>::
             default_setting)
#else
    : Base(mpi_communicator,
           smooth_grid,
           /*allow_artificial_cells = */ true,
           dealii::parallel::shared::Triangulation<dim, spacedim>::partition_auto)
#endif
  {}
} // namespace parallel


DEAL_II_NAMESPACE_CLOSE

#endif
