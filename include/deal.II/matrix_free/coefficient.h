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

#ifndef dealii_matrix_free_coefficient_h
#define dealii_matrix_free_coefficient_h

#include <deal.II/base/config.h>

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/memory_space.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

#include <deal.II/multigrid/mg_transfer_matrix_free.h>

#include <Kokkos_Core.hpp>

#include <array>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

DEAL_II_NAMESPACE_OPEN

namespace MatrixFreeTools
{
  /**
   * Storage layouts for a scalar coefficient.
   */
  enum class CoefficientLayout
  {
    /**
     * Store one value per cell and use it at every quadrature point.
     */
    cell,

    /**
     * Store one value per quadrature point on each cell.
     */
    quadrature
  };

  /**
   * Access the values of a scalar coefficient in a Portable::MatrixFree
   * kernel. The layout is a template argument so that the index calculation
   * is selected at compile time. Copies share the underlying Kokkos view.
   */
  template <typename Number, CoefficientLayout layout>
  struct DeviceCoefficient
  {
    /**
     * One-dimensional view in the default memory space.
     */
    using View = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>;

    /**
     * Coefficient values in the order used by operator().
     */
    View values;

    /**
     * The number of quadrature points per cell for the quadrature layout.
     */
    unsigned int n_q_points;

    /**
     * Construct an accessor that shares @p values. The cell layout can use
     * zero for @p n_q_points_in since it uses a single value for the cell.
     */
    DEAL_II_HOST_DEVICE
    DeviceCoefficient(const View &values, const unsigned int n_q_points_in)
      : values(values)
      , n_q_points(n_q_points_in)
    {}

    /**
     * Return the value at quadrature point @p q on cell @p cell. The cell
     * index includes the offset of its color in the packed storage. For the
     * quadrature layout the flat index is <code>cell * n_q_points + q</code>.
     * For the cell layout the value depends only on @p cell.
     */
    DEAL_II_HOST_DEVICE Number
    operator()(const unsigned int cell, const unsigned int q) const
    {
      if constexpr (layout == CoefficientLayout::cell)
        {
          (void)q;
          return values(cell);
        }
      else
        return values(cell * n_q_points + q);
    }
  };

  template <int dim, typename Number>
  class Coefficient;

  /**
   * Interpolate @p finest_coefficient to cellwise constants on all levels
   * of a nested geometric coarsening hierarchy, including the finest level.
   * Each coarse value is the volume-weighted mean of the fine cells it
   * contains. For a quadrature-point input, integrate the coefficient using
   * the finest backend's quadrature weights (including the Jacobian
   * determinant) before taking the mean.
   *
   * Each entry of @p level_matrix_free is an initialized MatrixFree or
   * Portable::MatrixFree object with active-cell DoF numbering. The input
   * must have been packed with the finest object in that hierarchy.
   * Initialize that object with update_JxW_values. For MatrixFree, the input
   * quadrature must match quadrature index zero. Each level must support
   * FE_DGQ(0) and geometric MGTwoLevelTransfer.
   *
   * @ingroup matrixfree
   */
  template <typename MatrixFreeType, int dim, typename Number>
  MGLevelObject<Coefficient<dim, Number>>
  interpolate_cellwise_coefficients_to_mg(
    const MGLevelObject<std::shared_ptr<MatrixFreeType>> &level_matrix_free,
    const Coefficient<dim, Number>                       &finest_coefficient);

  /**
   * Store a scalar coefficient in the order used by a MatrixFree or
   * Portable::MatrixFree cell loop. Use active-cell initialization for
   * Portable::MatrixFree.
   *
   * Initialize with a scalar Function or with a scalar finite element field
   * given by a DoFHandler and a source vector. Both backends support two
   * layouts.
   *
   * - For the cell layout, the function is evaluated at each cell center.
   *   A finite element source must have one degree of freedom per cell,
   *   such as FE_DGQ(0). Initialization copies that value.
   * - For the quadrature layout, also supply a mapping and a quadrature
   *   rule to evaluate the source at the mapped quadrature points. Use the
   *   same mapping and quadrature points, in the same order, as the
   *   evaluation object that will read the values.
   *
   * For a finite element source, the DoFHandler must provide active DoF
   * indices for the cells being packed and use the same triangulation as
   * the backend. The source vector must provide up-to-date ghost values for
   * any ghost cells read during packing. Each call to reinit() replaces the
   * stored values. Call it again if the source values or the backend's cell
   * ordering change.
   *
   * With MatrixFree, pass the FEEvaluation object for the current SIMD batch
   * and a quadrature point index to value() to obtain a VectorizedArray.
   * With Portable::MatrixFree, obtain a device accessor from
   * portable_view() using the layout selected at initialization. Pass that
   * accessor to the cell kernel and call it with the index returned by
   * Portable::FEEvaluation::get_current_cell_index() and the quadrature point
   * index. For the cell layout, both accessors return the same value for all
   * quadrature points on a given cell.
   *
   * @ingroup matrixfree
   */
  template <int dim, typename Number = double>
  class Coefficient
  {
    /**
     * Coefficient values for the cells in one SIMD batch.
     */
    using CellArray = std::array<Number, VectorizedArray<Number>::size()>;

    /**
     * Storage type for the MatrixFree backend.
     */
    using MatrixFreeCellData = AlignedVector<CellArray>;

  public:
    /**
     * Type of the source finite element field on the host.
     */
    using VectorType =
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Host>;

    /**
     * Storage type for the Portable::MatrixFree backend.
     */
    using View = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>;

    /**
     * Pack one value per cell from @p field, including ghost cell batches
     * when present in @p data. Use a scalar, piecewise constant finite
     * element with one degree of freedom per cell, such as FE_DGQ(0).
     */
    template <typename MatrixFreeType>
    void
    reinit(const MatrixFreeType  &data,
           const DoFHandler<dim> &field_dof_handler,
           const VectorType      &field)
    {
      Assert(field_dof_handler.get_fe().dofs_per_cell == 1,
             ExcMessage("Coefficient input must have one DoF per cell."));
      std::vector<types::global_dof_index> indices(1);
      pack(data,
           CoefficientLayout::cell,
           1,
           [&](const auto &cell, auto &values) {
             cell->as_dof_handler_iterator(field_dof_handler)
               ->get_dof_indices(indices);
             values[0] = field(indices[0]);
           });
    }

    /**
     * Pack the values of a scalar finite element @p field at the mapped
     * points of @p quadrature, including ghost cell batches when present
     * in @p data.
     */
    template <typename MatrixFreeType>
    void
    reinit(const MatrixFreeType  &data,
           const Mapping<dim>    &mapping,
           const DoFHandler<dim> &field_dof_handler,
           const VectorType      &field,
           const Quadrature<dim> &quadrature)
    {
      FEValues<dim> fe_values(mapping,
                              field_dof_handler.get_fe(),
                              quadrature,
                              update_values);
      pack(data,
           CoefficientLayout::quadrature,
           quadrature.size(),
           [&](const auto &cell, auto &values) {
             fe_values.reinit(cell->as_dof_handler_iterator(field_dof_handler));
             fe_values.get_function_values(field, values);
           });
    }

    /**
     * Pack one value per cell by evaluating the scalar @p function at the
     * geometric cell center returned by CellAccessor::center().
     */
    template <typename MatrixFreeType, typename RangeNumberType>
    void
    reinit(const MatrixFreeType                 &data,
           const Function<dim, RangeNumberType> &function)
    {
      AssertDimension(function.n_components, 1);
      pack(data,
           CoefficientLayout::cell,
           1,
           [&](const auto &cell, auto &values) {
             values[0] = function.value(cell->center());
           });
    }

    /**
     * Pack the values of the scalar @p function at the points of
     * @p quadrature transformed to each real cell by @p mapping.
     */
    template <typename MatrixFreeType, typename RangeNumberType>
    void
    reinit(const MatrixFreeType                 &data,
           const Mapping<dim>                   &mapping,
           const Function<dim, RangeNumberType> &function,
           const Quadrature<dim>                &quadrature)
    {
      AssertDimension(function.n_components, 1);
      pack(data,
           CoefficientLayout::quadrature,
           quadrature.size(),
           [&](const auto &cell, auto &values) {
             for (unsigned int q = 0; q < values.size(); ++q)
               values[q] = function.value(
                 mapping.transform_unit_to_real_cell(cell,
                                                     quadrature.point(q)));
           });
    }

    /**
     * Return the layout used to store the coefficient.
     */
    CoefficientLayout
    get_layout() const
    {
      return coefficient_layout;
    }

    /**
     * Return a device accessor sharing the packed values. This function
     * requires initialization with Portable::MatrixFree and a layout matching
     * @p requested.
     */
    template <CoefficientLayout requested>
    DeviceCoefficient<Number, requested>
    portable_view() const
    {
      Assert(requested == coefficient_layout,
             ExcMessage(
               "Requested device layout does not match the coefficient."));
      return DeviceCoefficient<Number, requested>(
        std::get<View>(representation), n_q_points);
    }

    /**
     * Return the coefficient at quadrature point @p q for all cells in the
     * batch selected by @p fe_eval. This function requires initialization
     * with the same MatrixFree object as @p fe_eval. In the cell layout,
     * the value on each cell is shared by all its quadrature points.
     */
    VectorizedArray<Number>
    value(const FEEvaluationData<dim, VectorizedArray<Number>, false> &fe_eval,
          const unsigned int                                           q) const
    {
      if (coefficient_layout == CoefficientLayout::quadrature)
        AssertIndexRange(q, n_q_points);

      const auto     &cell_data = std::get<MatrixFreeCellData>(representation);
      const CellArray values =
        coefficient_layout == CoefficientLayout::cell ?
          fe_eval.read_cell_data(cell_data) :
          cell_data[fe_eval.get_current_cell_index() * n_q_points + q];
      VectorizedArray<Number> result;
      result.load(values.data());
      return result;
    }

  private:
    template <typename MatrixFreeType, int dim_, typename Number_>
    friend MGLevelObject<Coefficient<dim_, Number_>>
    interpolate_cellwise_coefficients_to_mg(
      const MGLevelObject<std::shared_ptr<MatrixFreeType>> &level_matrix_free,
      const Coefficient<dim_, Number_>                     &finest_coefficient);

    /**
     * Compute the integral and volume of each owned cell in DGQ0 DoF order.
     */
    void
    compute_cell_integrals(const MatrixFree<dim, Number> &data,
                           const DoFHandler<dim>         &dof_handler,
                           VectorType                    &integrals,
                           VectorType                    &volumes) const
    {
      AssertThrow(
        data.get_mapping_info().update_flags_cells & update_JxW_values,
        ExcMessage("Coefficient interpolation requires update_JxW_values."));
      const bool cellwise = coefficient_layout == CoefficientLayout::cell;
      FEEvaluation<dim, -1, 0, 1, Number>  fe_eval(data);
      std::vector<types::global_dof_index> indices(1);
      for (unsigned int batch = 0; batch < data.n_cell_batches(); ++batch)
        {
          fe_eval.reinit(batch);
          AssertThrow(cellwise || n_q_points == fe_eval.n_q_points,
                      ExcMessage(
                        "The coefficient and backend quadratures differ."));
          VectorizedArray<Number> integral{}, volume{};
          for (const unsigned int q : fe_eval.quadrature_point_indices())
            {
              const auto weight = fe_eval.JxW(q);
              volume += weight;
              if (!cellwise)
                integral += value(fe_eval, q) * weight;
            }
          if (cellwise)
            integral = value(fe_eval, 0) * volume;
          for (unsigned int lane = 0;
               lane < data.n_active_entries_per_cell_batch(batch);
               ++lane)
            {
              const auto cell = data.get_cell_iterator(batch, lane);
              cell->as_dof_handler_iterator(dof_handler)
                ->get_dof_indices(indices);
              integrals[indices[0]] = integral[lane];
              volumes[indices[0]]   = volume[lane];
            }
        }
    }

    /**
     * Compute the integral and volume of each owned portable cell in DGQ0
     * DoF order using host mirrors of the packed values and quadrature data.
     */
    void
    compute_cell_integrals(const Portable::MatrixFree<dim, Number> &data,
                           const DoFHandler<dim>                   &dof_handler,
                           VectorType                              &integrals,
                           VectorType &volumes) const
    {
      const bool cellwise = coefficient_layout == CoefficientLayout::cell;
      const auto values =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                            std::get<View>(representation));
      std::vector<types::global_dof_index> indices(1);
      for (unsigned int color = 0; color < data.get_colored_graph().size();
           ++color)
        {
          if (data.n_cells_per_color(color) == 0)
            continue;
          const auto         device_data = data.get_data(color, 0);
          const unsigned int n_points    = device_data.q_weights.extent(0);
          AssertThrow(
            device_data.JxW.size() > 0,
            ExcMessage(
              "Coefficient interpolation requires update_JxW_values."));
          AssertThrow(cellwise || n_q_points == n_points,
                      ExcMessage(
                        "The coefficient and backend quadratures differ."));
          const auto host_data =
            Portable::copy_mf_data_to_host<dim, Number>(device_data,
                                                        update_JxW_values);
          for (unsigned int cell_index = 0; cell_index < host_data.n_cells;
               ++cell_index)
            {
              const unsigned int cell_id  = host_data.first_cell + cell_index;
              Number             integral = 0, volume = 0;
              for (unsigned int q = 0; q < n_points; ++q)
                {
                  const auto weight = host_data.JxW_value(cell_index, q);
                  volume += weight;
                  if (!cellwise)
                    integral += values(cell_id * n_points + q) * weight;
                }
              if (cellwise)
                integral = values(cell_id) * volume;
              const auto cell = data.get_cell_iterator(color, cell_index);
              cell->as_dof_handler_iterator(dof_handler)
                ->get_dof_indices(indices);
              integrals[indices[0]] = integral;
              volumes[indices[0]]   = volume;
            }
        }
    }

    /**
     * Store samples in SIMD batches. The sampling function fills one entry
     * per point for a single cell.
     */
    template <typename Sample>
    void
    pack(const MatrixFree<dim, Number> &data,
         const CoefficientLayout        layout,
         const unsigned int             n_values,
         const Sample                  &sample)
    {
      const unsigned int n_batches =
        data.n_cell_batches() + data.n_ghost_cell_batches();
      MatrixFreeCellData  cell_data(n_batches * n_values);
      std::vector<Number> values(n_values);
      for (unsigned int batch = 0; batch < n_batches; ++batch)
        {
          const unsigned int n_lanes =
            data.n_active_entries_per_cell_batch(batch);
          const unsigned int offset = batch * n_values;
          for (unsigned int lane = 0; lane < n_lanes; ++lane)
            {
              sample(data.get_cell_iterator(batch, lane), values);
              for (unsigned int q = 0; q < n_values; ++q)
                cell_data[offset + q][lane] = values[q];
            }
          if (n_lanes > 0)
            for (unsigned int q = 0; q < n_values; ++q)
              for (unsigned int lane = n_lanes;
                   lane < VectorizedArray<Number>::size();
                   ++lane)
                cell_data[offset + q][lane] = cell_data[offset + q][0];
        }
      representation     = std::move(cell_data);
      coefficient_layout = layout;
      n_q_points = layout == CoefficientLayout::quadrature ? n_values : 0;
    }

    /**
     * Store samples in color order and copy them to the default memory space.
     */
    template <typename Sample>
    void
    pack(const Portable::MatrixFree<dim, Number> &data,
         const CoefficientLayout                  layout,
         const unsigned int                       n_values,
         const Sample                            &sample)
    {
      const auto cell_offsets = portable_cell_offsets(data);
      View       values("coefficient values", cell_offsets.back() * n_values);
      auto       host_values = Kokkos::create_mirror_view(values);
      std::vector<Number> samples(n_values);
      for (unsigned int color = 0; color + 1 < cell_offsets.size(); ++color)
        for (unsigned int cell = 0; cell < data.n_cells_per_color(color);
             ++cell)
          {
            sample(data.get_cell_iterator(color, cell), samples);
            const unsigned int base = (cell_offsets[color] + cell) * n_values;
            for (unsigned int q = 0; q < n_values; ++q)
              host_values(base + q) = samples[q];
          }
      Kokkos::deep_copy(values, host_values);
      representation     = std::move(values);
      coefficient_layout = layout;
      n_q_points = layout == CoefficientLayout::quadrature ? n_values : 0;
    }

    /**
     * Return the starting cell index for each color, followed by the total
     * number of cells. These offsets match the numbering used by
     * Portable::FEEvaluation::get_current_cell_index(), including when a
     * color is empty.
     */
    static std::vector<unsigned int>
    portable_cell_offsets(const Portable::MatrixFree<dim, Number> &data)
    {
      const auto                n_colors = data.get_colored_graph().size();
      std::vector<unsigned int> offsets(n_colors + 1, 0);
      for (unsigned int color = 0; color < n_colors; ++color)
        offsets[color + 1] = offsets[color] + data.n_cells_per_color(color);
      return offsets;
    }

    std::variant<std::monostate, View, MatrixFreeCellData> representation;
    CoefficientLayout coefficient_layout = CoefficientLayout::cell;
    unsigned int      n_q_points         = 0;
  };

  template <typename MatrixFreeType, int dim, typename Number>
  MGLevelObject<Coefficient<dim, Number>>
  interpolate_cellwise_coefficients_to_mg(
    const MGLevelObject<std::shared_ptr<MatrixFreeType>> &level_matrix_free,
    const Coefficient<dim, Number>                       &finest_coefficient)
  {
    using VectorType = typename Coefficient<dim, Number>::VectorType;
    AssertThrow(level_matrix_free.n_levels() > 0, ExcNotInitialized());
    const unsigned int min_level = level_matrix_free.min_level();
    const unsigned int max_level = level_matrix_free.max_level();

    const FE_DGQ<dim>              fe(0);
    MGLevelObject<DoFHandler<dim>> dof_handlers(min_level, max_level);
    MGLevelObject<VectorType>      integrals(min_level, max_level);
    MGLevelObject<VectorType>      volumes(min_level, max_level);
    for (unsigned int level = min_level; level <= max_level; ++level)
      {
        const auto &backend = level_matrix_free[level];
        AssertThrow(backend != nullptr, ExcNotInitialized());
        auto &dof_handler = dof_handlers[level];
        dof_handler.reinit(backend->get_dof_handler().get_triangulation());
        dof_handler.distribute_dofs(fe);
        const auto partitioner =
          std::make_shared<const Utilities::MPI::Partitioner>(
            dof_handler.locally_owned_dofs(),
            DoFTools::extract_locally_relevant_dofs(dof_handler),
            dof_handler.get_mpi_communicator());
        integrals[level].reinit(partitioner);
        volumes[level].reinit(partitioner);
      }

    finest_coefficient.compute_cell_integrals(*level_matrix_free[max_level],
                                              dof_handlers[max_level],
                                              integrals[max_level],
                                              volumes[max_level]);

    // The transpose of DGQ0 prolongation sums the contributions of the fine
    // cells. Transfer the integrals and volumes before dividing so each fine
    // cell contributes in proportion to its volume at every coarser level.
    for (unsigned int level = max_level; level > min_level; --level)
      {
        MGTwoLevelTransfer<dim, VectorType> transfer;
        transfer.reinit(dof_handlers[level], dof_handlers[level - 1]);
        transfer.restrict_and_add(integrals[level - 1], integrals[level]);
        transfer.restrict_and_add(volumes[level - 1], volumes[level]);
      }

    MGLevelObject<Coefficient<dim, Number>> result(min_level, max_level);
    for (unsigned int level = min_level; level <= max_level; ++level)
      {
        auto &averages = integrals[level];
        for (unsigned int i = 0; i < averages.locally_owned_size(); ++i)
          {
            const Number volume = volumes[level].local_element(i);
            AssertThrow(volume > 0,
                        ExcMessage("The cell volume must be positive."));
            averages.local_element(i) /= volume;
          }
        averages.update_ghost_values();
        result[level].reinit(*level_matrix_free[level],
                             dof_handlers[level],
                             averages);
      }
    return result;
  }
} // namespace MatrixFreeTools

DEAL_II_NAMESPACE_CLOSE

#endif
