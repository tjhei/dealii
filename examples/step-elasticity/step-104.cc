/* -----------------------------------------------------------------------------
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
*/



// A matrix-free linear elasticity example with a manufactured solution.


#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/data_out.h>

#include <iomanip>
#include <iostream>
#include <string>
#include <variant>

using namespace dealii;

#if defined(DEAL_II_WITH_MPI) && defined(DEAL_II_WITH_P4EST)
template <int dim>
class ExampleTriangulation : public parallel::distributed::Triangulation<dim>
{
public:
  ExampleTriangulation()
    : parallel::distributed::Triangulation<dim>(MPI_COMM_WORLD)
  {}
};
#else
template <int dim>
using ExampleTriangulation = Triangulation<dim>;
#endif

namespace
{
  // Smooth, positive Lamé parameters used to initialize DGQ0 coefficient
  // fields. The continuous expressions are sampled at each cell center.
  template <int dim>
  class LambdaCoefficientFunction : public Function<dim>
  {
  public:
    double value(const Point<dim> &p,
                 const unsigned int = 0) const override
    {
      return 1. + p[0];
    }
  };

  template <int dim>
  class MuCoefficientFunction : public Function<dim>
  {
  public:
    double value(const Point<dim> &p,
                 const unsigned int = 0) const override
    {
      double coefficient = 1.;
      if constexpr (dim > 1)
        coefficient += p[1];
      if constexpr (dim > 2)
        coefficient += p[2];
      return coefficient;
    }
  };

  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution()
      : Function<dim>(dim)
    {}

    double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      AssertIndexRange(component, dim);
      double result = 1.;
      for (unsigned int d = 0; d < dim; ++d)
        result *= p[d] * (1. - p[d]);
      return result;
    }
  };


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

  /**
   * Pack a scalar finite element field for use in matrix-free cell loops.
   * Each call to reinit() replaces the stored values with a copy in the cell
   * order of the supplied MatrixFree or Portable::MatrixFree object. The
   * MatrixFree overloads pack cells into SIMD batches, and the
   * Portable::MatrixFree overloads copy the values to the default memory
   * space for use in device kernels.
   *
   * The source DoFHandler must describe a scalar field on the same
   * triangulation as the matrix-free object, with active DoF indices on the
   * cells being packed. The source vector must contain up-to-date ghost
   * entries for any cells read during packing. The coefficient owns the
   * packed values, so changes to the source field or the matrix-free cell
   * ordering require another call to reinit().
   *
   * In this example, the Lamé parameters are represented by FE_DGQ(0) fields.
   * The cell layout reads the single DoF value on each cell. The quadrature
   * layout evaluates the same field at each quadrature point, so both
   * layouts represent the same piecewise constant coefficients.
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
     * Pack one value per cell for @p data, including its ghost cell batches.
     * Read the single DoF on each cell of @p field_dof_handler directly from
     * @p field. Use a scalar, piecewise constant element such as FE_DGQ(0).
     */
    void
    reinit(const MatrixFree<dim, Number> &data,
           const DoFHandler<dim>        &field_dof_handler,
           const VectorType             &field)
    {
      coefficient_layout = CoefficientLayout::cell;
      n_q_points         = 0;
      Assert(field_dof_handler.get_fe().dofs_per_cell == 1,
             ExcMessage("Coefficient input must have one DoF per cell."));
      MatrixFreeCellData cell_data;
      data.initialize_cell_data_vector(cell_data);
      FEEvaluation<dim, -1, 0, 1, Number> fe_eval(data, 0, 0, 0);
      const unsigned int n_batches =
        data.n_cell_batches() + data.n_ghost_cell_batches();
      std::vector<types::global_dof_index> indices(1);
      for (unsigned int batch = 0; batch < n_batches; ++batch)
        {
          fe_eval.reinit(batch);
          CellArray local_values{};
          for (unsigned int lane = 0;
               lane < data.n_active_entries_per_cell_batch(batch);
               ++lane)
            {
              const auto cell =
                data.get_cell_iterator(batch, lane)
                  ->as_dof_handler_iterator(field_dof_handler);
              cell->get_dof_indices(indices);
              local_values[lane] = field(indices[0]);
            }
          fe_eval.set_cell_data(cell_data, local_values);
        }
      representation = std::move(cell_data);
    }

    /**
     * Evaluate @p field at each point of @p quadrature using @p mapping and
     * @p field_dof_handler, and pack the results in the cell batch order of
     * @p data, including its ghost cell batches. Use the mapping and the
     * quadrature points, in the same order, of the consuming FEEvaluation.
     */
    void
    reinit(const MatrixFree<dim, Number> &data,
           const Mapping<dim>           &mapping,
           const DoFHandler<dim>        &field_dof_handler,
           const VectorType             &field,
           const Quadrature<dim>        &quadrature)
    {
      coefficient_layout = CoefficientLayout::quadrature;
      n_q_points         = quadrature.size();
      const unsigned int n_batches =
        data.n_cell_batches() + data.n_ghost_cell_batches();
      MatrixFreeCellData cell_data(n_batches * n_q_points);
      FEValues<dim>       fe_values(mapping,
                              field_dof_handler.get_fe(),
                              quadrature,
                              update_values);
      std::vector<Number> q_values(n_q_points);
      for (unsigned int batch = 0; batch < n_batches; ++batch)
        {
          std::vector<CellArray> local_values(n_q_points);
          const unsigned int     n_lanes =
            data.n_active_entries_per_cell_batch(batch);
          for (unsigned int lane = 0; lane < n_lanes; ++lane)
            {
              const auto cell =
                data.get_cell_iterator(batch, lane)
                  ->as_dof_handler_iterator(field_dof_handler);
              fe_values.reinit(cell);
              fe_values.get_function_values(field, q_values);
              for (unsigned int q = 0; q < n_q_points; ++q)
                local_values[q][lane] = q_values[q];
            }
          if (n_lanes > 0)
            for (unsigned int lane = n_lanes;
                 lane < VectorizedArray<Number>::size();
                 ++lane)
              for (unsigned int q = 0; q < n_q_points; ++q)
                local_values[q][lane] = local_values[q][0];
          for (unsigned int q = 0; q < n_q_points; ++q)
            cell_data[batch * n_q_points + q] = local_values[q];
        }
      representation = std::move(cell_data);
    }

    /**
     * Pack one value per cell in the color order of @p data and copy the
     * result to the default memory space. Read the single DoF on each cell
     * of @p field_dof_handler directly from @p field. Use a scalar, piecewise
     * constant element such as FE_DGQ(0).
     */
    void
    reinit(const Portable::MatrixFree<dim, Number> &data,
           const DoFHandler<dim>                   &field_dof_handler,
           const VectorType                        &field)
    {
      coefficient_layout = CoefficientLayout::cell;
      n_q_points         = 0;
      Assert(field_dof_handler.get_fe().dofs_per_cell == 1,
             ExcMessage("Coefficient input must have one DoF per cell."));
      const auto cell_offsets = portable_cell_offsets(data);

      View values("cell coefficient data", cell_offsets.back());
      auto host_values = Kokkos::create_mirror_view(values);
      std::vector<types::global_dof_index> indices(1);

      for (unsigned int color = 0; color + 1 < cell_offsets.size(); ++color)
        for (unsigned int cell_index = 0;
             cell_index < data.n_cells_per_color(color);
             ++cell_index)
          {
            const auto cell = data.get_cell_iterator(color, cell_index)
                                ->as_dof_handler_iterator(field_dof_handler);
            cell->get_dof_indices(indices);
            host_values[cell_offsets[color] + cell_index] = field(indices[0]);
          }
      Kokkos::deep_copy(values, host_values);
      representation = std::move(values);
    }

    /**
     * Evaluate @p field at each point of @p quadrature using @p mapping and
     * @p field_dof_handler, and pack the results in the color order of
     * @p data before copying them to the default memory space. Use the
     * mapping and the quadrature points, in the same order, of the consuming
     * Portable::FEEvaluation.
     */
    void
    reinit(const Portable::MatrixFree<dim, Number> &data,
           const Mapping<dim>                      &mapping,
           const DoFHandler<dim>                   &field_dof_handler,
           const VectorType                        &field,
           const Quadrature<dim>                   &quadrature)
    {
      coefficient_layout     = CoefficientLayout::quadrature;
      n_q_points             = quadrature.size();
      const auto cell_offsets = portable_cell_offsets(data);

      View values("quadrature coefficient data",
                  cell_offsets.back() * n_q_points);
      auto host_values = Kokkos::create_mirror_view(values);
      FEValues<dim> fe_values(
        mapping, field_dof_handler.get_fe(), quadrature, update_values);
      std::vector<Number> q_values(n_q_points);

      for (unsigned int color = 0; color + 1 < cell_offsets.size(); ++color)
        for (unsigned int cell_index = 0;
             cell_index < data.n_cells_per_color(color);
             ++cell_index)
          {
            const auto cell = data.get_cell_iterator(color, cell_index)
                                ->as_dof_handler_iterator(field_dof_handler);
            fe_values.reinit(cell);
            fe_values.get_function_values(field, q_values);
            const auto base = (cell_offsets[color] + cell_index) * n_q_points;
            for (unsigned int q = 0; q < n_q_points; ++q)
              host_values[base + q] = q_values[q];
          }
      Kokkos::deep_copy(values, host_values);
      representation = std::move(values);
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
      return DeviceCoefficient<Number, requested>(std::get<View>(representation),
                                                  n_q_points);
    }

    /**
     * Return the coefficient at quadrature point @p q for all cells in the
     * batch selected by @p fe_eval. This function requires initialization
     * with the same MatrixFree object as @p fe_eval. In the cell layout,
     * the value on each cell is shared by all its quadrature points.
     */
    VectorizedArray<Number>
    value(const FEEvaluationData<dim, VectorizedArray<Number>, false> &fe_eval,
          const unsigned int q) const
    {
      const auto &cell_data = std::get<MatrixFreeCellData>(representation);
      const CellArray values =
        coefficient_layout == CoefficientLayout::cell ?
          fe_eval.read_cell_data(cell_data) :
          cell_data[fe_eval.get_current_cell_index() * n_q_points + q];
      VectorizedArray<Number> result;
      result.load(values.data());
      return result;
    }

  private:
    /**
     * Return the starting cell index for each color, followed by the total
     * number of cells. These offsets match the numbering used by
     * Portable::FEEvaluation::get_current_cell_index(), including when a
     * color is empty.
     */
    static std::vector<unsigned int>
    portable_cell_offsets(const Portable::MatrixFree<dim, Number> &data)
    {
      const auto n_colors =
        data.get_mg_level() == numbers::invalid_unsigned_int ?
          data.get_colored_graph().size() :
          data.get_colored_level_graph().size();
      std::vector<unsigned int> offsets(n_colors + 1, 0);
      for (unsigned int color = 0; color < n_colors; ++color)
        offsets[color + 1] = offsets[color] + data.n_cells_per_color(color);
      return offsets;
    }

    std::variant<std::monostate, View, MatrixFreeCellData> representation;
    CoefficientLayout coefficient_layout = CoefficientLayout::cell;
    unsigned int      n_q_points         = 0;
  };


  template <int dim, typename CoarseSequence>
  void
  initialize_cellwise_coefficient_field(
    MGLevelObject<DoFHandler<dim>> &dof_handlers,
    MGLevelObject<typename Coefficient<dim>::VectorType> &fields,
    const FE_DGQ<dim> &finite_element,
    const CoarseSequence &coarse,
    const Function<dim> &function)
  {
    const unsigned int max_level = coarse.size() - 1;
    dof_handlers.resize(0, max_level);
    fields.resize(0, max_level);
    for (unsigned int level = 0; level <= max_level; ++level)
      {
        dof_handlers[level].reinit(*coarse[level]);
        dof_handlers[level].distribute_dofs(finite_element);
        fields[level].reinit(std::make_shared<Utilities::MPI::Partitioner>(
          dof_handlers[level].locally_owned_dofs(),
          DoFTools::extract_locally_relevant_dofs(dof_handlers[level]),
          dof_handlers[level].get_mpi_communicator()));
        VectorTools::interpolate(dof_handlers[level], function, fields[level]);
        fields[level].update_ghost_values();
      }
  }


  template <int dim>
  void
  transfer_cellwise_coefficient_field(
    MGLevelObject<DoFHandler<dim>> &dof_handlers,
    MGLevelObject<typename Coefficient<dim>::VectorType> &fields)
  {
    const unsigned int min_level = fields.min_level();
    const unsigned int max_level = fields.max_level();
    using VectorType = typename Coefficient<dim>::VectorType;
    MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers(min_level,
                                                                  max_level);
    for (unsigned int level = min_level; level < max_level; ++level)
      transfers[level + 1].reinit(dof_handlers[level + 1],
                                  dof_handlers[level]);
    MGTransferMatrixFree<dim, double, MemorySpace::Host> transfer(
      transfers, [&](const auto level, auto &vector) {
        vector.reinit(fields[level].get_partitioner());
      });
    MGLevelObject<VectorType> level_fields(min_level, max_level);
    VectorType                source;
    source.reinit(fields[max_level].get_partitioner());
    source.import_elements(fields[max_level], VectorOperation::insert);
    transfer.interpolate_to_mg(level_fields, source);
    for (unsigned int level = min_level; level <= max_level; ++level)
      {
        fields[level].import_elements(level_fields[level],
                                      VectorOperation::insert);
        fields[level].update_ghost_values();
      }
  }


  template <int dim, typename VectorType>
  void
  write_graphical_output(const DoFHandler<dim> &dof_handler,
                         const VectorType &solution,
                         const DoFHandler<dim> &lambda_dof_handler,
                         const typename Coefficient<dim>::VectorType
                           &lambda_field,
                         const DoFHandler<dim> &mu_dof_handler,
                         const typename Coefficient<dim>::VectorType &mu_field,
                         const std::string &file_prefix)
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> displacement_names(dim);
    for (unsigned int d = 0; d < dim; ++d)
      displacement_names[d] = "displacement_" + std::to_string(d);
    data_out.add_data_vector(
      solution,
      displacement_names,
      DataOut<dim>::type_dof_data,
      std::vector<DataComponentInterpretation::DataComponentInterpretation>(
        dim, DataComponentInterpretation::component_is_part_of_vector));
    data_out.add_data_vector(lambda_dof_handler, lambda_field, "lambda");
    data_out.add_data_vector(mu_dof_handler, mu_field, "mu");
    data_out.build_patches();
    data_out.write_vtu_with_pvtu_record(
      "./", file_prefix, 0, MPI_COMM_WORLD, 4, 0);
  }
} // namespace


template <int dim, int fe_degree, CoefficientLayout layout, typename Number>
class ElasticityCellOperatorQuadrature
{
public:
  DEAL_II_HOST_DEVICE
  ElasticityCellOperatorQuadrature(
    const DeviceCoefficient<Number, layout> &mu_values,
    const DeviceCoefficient<Number, layout> &lambda_values)
    : mu_values(mu_values)
    , lambda_values(lambda_values)
  {}

  DEAL_II_HOST_DEVICE void
  operator()(
    Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, dim, Number> *fe_eval,
    const int q_point) const
  {
    const unsigned int cell_index = fe_eval->get_current_cell_index();
    const Tensor<2, dim, Number> gradient = fe_eval->get_gradient(q_point);
    Tensor<2, dim, Number> stress =
      mu_values(cell_index, q_point) * (gradient + transpose(gradient));
    const Number volumetric_stress =
      lambda_values(cell_index, q_point) * trace(gradient);
    for (unsigned int d = 0; d < dim; ++d)
      stress[d][d] += volumetric_stress;
    fe_eval->submit_gradient(stress, q_point);
  }

private:
  DeviceCoefficient<Number, layout> mu_values;
  DeviceCoefficient<Number, layout> lambda_values;
};

template <int dim, int degree, CoefficientLayout layout, typename Number>
class ElasticityCellOperator
{
public:
  static constexpr unsigned int n_q_points = Utilities::pow(degree + 1, dim);

  ElasticityCellOperator(const DeviceCoefficient<Number, layout> &mu_values,
                         const DeviceCoefficient<Number, layout> &lambda_values)
    : mu_values(mu_values)
    , lambda_values(lambda_values)
  {}

  DEAL_II_HOST_DEVICE
  void
  operator()(const typename Portable::MatrixFree<dim, Number>::Data *data,
             const Portable::DeviceVector<Number>                   &src,
             Portable::DeviceVector<Number>                         &dst) const
  {
    Portable::FEEvaluation<dim, degree, degree + 1, dim, Number> fe(data, 0);
    fe.read_dof_values(src);
    fe.evaluate(EvaluationFlags::gradients);

    ElasticityCellOperatorQuadrature<dim, degree, layout, Number> quad_operator(
      mu_values, lambda_values);
    data->for_each_quad_point(
      [&](const int q_point) { quad_operator(&fe, q_point); });

    fe.integrate(EvaluationFlags::gradients);
    fe.distribute_local_to_global(dst);
  }

private:
  DeviceCoefficient<Number, layout> mu_values;
  DeviceCoefficient<Number, layout> lambda_values;
};

template <int dim,
          int degree,
          CoefficientLayout layout,
          typename Number = double>
class ElasticityOperator : public EnableObserverPointer
{
public:
  using VectorType =
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

  ElasticityOperator() = default;

  void
  reinit(std::shared_ptr<Portable::MatrixFree<dim, Number>> data_in,
         const Coefficient<dim, Number> &lambda_coefficient,
         const Coefficient<dim, Number> &mu_coefficient)
  {
    data             = std::move(data_in);
    lambda_values    = &lambda_coefficient;
    mu_values        = &mu_coefficient;
  }

  void
  initialize_dof_vector(VectorType &v) const
  {
    data->initialize_dof_vector(v, 0);
  }

  types::global_dof_index
  m() const
  {
    return data->get_dof_handler(0).n_dofs();
  }

  double
  el(const types::global_dof_index row,
     const types::global_dof_index column) const
  {
    Assert(row == column, ExcNotImplemented());
    Assert(inverse_diagonal.get() != nullptr, ExcNotInitialized());
    return 1. / (*inverse_diagonal)(row, row);
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    dst = 0.;
    data->cell_loop(ElasticityCellOperator<dim, degree, layout, Number>(
                      mu_values->template portable_view<layout>(),
                      lambda_values->template portable_view<layout>()),
                    src,
                    dst);
    data->copy_constrained_values(src, dst, 0);
  }

  void
  Tvmult(VectorType &dst, const VectorType &src) const
  {
    vmult(dst, src);
  }

  void
  compute_diagonal()
  {
    inverse_diagonal     = std::make_shared<DiagonalMatrix<VectorType>>();
    VectorType &diagonal = inverse_diagonal->get_vector();
    data->initialize_dof_vector(diagonal, 0);
    ElasticityCellOperatorQuadrature<dim, degree, layout, Number> quad_operator(
      mu_values->template portable_view<layout>(),
      lambda_values->template portable_view<layout>());

    MatrixFreeTools::compute_diagonal<dim, degree, degree + 1, dim, Number>(
      *data,
      diagonal,
      quad_operator,
      EvaluationFlags::gradients,
      EvaluationFlags::gradients);

    double *values = diagonal.get_values();
    Kokkos::parallel_for(
      "invert diagonal",
      diagonal.locally_owned_size(),
      KOKKOS_LAMBDA(const int i) {
        values[i] = (values[i] == 0.0) ? 1.0 : 1. / values[i];
      });
  }

  std::shared_ptr<DiagonalMatrix<VectorType>>
  diagonal_inverse() const
  {
    return inverse_diagonal;
  }

private:
  std::shared_ptr<Portable::MatrixFree<dim, Number>> data;
  std::shared_ptr<DiagonalMatrix<VectorType>>        inverse_diagonal;
  const Coefficient<dim, Number>                    *lambda_values = nullptr;
  const Coefficient<dim, Number>                    *mu_values = nullptr;
};


template <int dim, int degree, typename Number = double>
class MatrixFreeElasticityOperator : public EnableObserverPointer
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Host>;
  void
  reinit(std::shared_ptr<MatrixFree<dim, Number>> data_in,
         const Coefficient<dim, Number> &lambda_coefficient,
         const Coefficient<dim, Number> &mu_coefficient)
  {
    data          = std::move(data_in);
    lambda_values = &lambda_coefficient;
    mu_values     = &mu_coefficient;
  }

  void
  initialize_dof_vector(VectorType &vector) const
  {
    data->initialize_dof_vector(vector);
  }

  types::global_dof_index
  m() const
  {
    return data->get_dof_handler().n_dofs();
  }

  double
  el(const types::global_dof_index row,
     const types::global_dof_index column) const
  {
    Assert(row == column, ExcNotImplemented());
    Assert(inverse_diagonal.get() != nullptr, ExcNotInitialized());
    return 1. / (*inverse_diagonal)(row, row);
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    data->cell_loop(&MatrixFreeElasticityOperator::local_apply,
                    this,
                    dst,
                    src,
                    true);
    data->get_affine_constraints().set_zero(dst);
  }

  void
  Tvmult(VectorType &dst, const VectorType &src) const
  {
    vmult(dst, src);
  }

  void
  compute_diagonal()
  {
    inverse_diagonal = std::make_shared<DiagonalMatrix<VectorType>>();
    VectorType &diagonal = inverse_diagonal->get_vector();
    initialize_dof_vector(diagonal);
    using Evaluation = FEEvaluation<dim, degree, degree + 1, dim, Number>;
    const std::function<void(Evaluation &)> cell_operator = [this](auto &fe) {
      fe.evaluate(EvaluationFlags::gradients);
      for (unsigned int q = 0; q < fe.n_q_points; ++q)
        (*this)(&fe, q);
      fe.integrate(EvaluationFlags::gradients);
    };
    MatrixFreeTools::compute_diagonal<dim,
                                      degree,
                                      degree + 1,
                                      dim,
                                      Number,
                                      VectorizedArray<Number>,
                                      VectorType>(
      *data,
      diagonal,
      cell_operator,
      0,
      0,
      0,
      0);
    for (unsigned int i = 0; i < diagonal.locally_owned_size(); ++i)
      diagonal.local_element(i) =
        diagonal.local_element(i) == 0. ? 1. : 1. / diagonal.local_element(i);
  }

  std::shared_ptr<DiagonalMatrix<VectorType>>
  diagonal_inverse() const
  {
    return inverse_diagonal;
  }

  void
  operator()(FEEvaluation<dim, degree, degree + 1, dim, Number> *fe,
             const int q_point) const
  {
    const auto lambda_value = lambda_values->value(*fe, q_point);
    const auto mu_value     = mu_values->value(*fe, q_point);
    const Tensor<2, dim, VectorizedArray<Number>> gradient =
      fe->get_gradient(q_point);
    Tensor<2, dim, VectorizedArray<Number>> stress =
      mu_value * (gradient + transpose(gradient));
    const auto volumetric_stress = lambda_value * trace(gradient);
    for (unsigned int d = 0; d < dim; ++d)
      stress[d][d] += volumetric_stress;
    fe->submit_gradient(stress, q_point);
  }

private:
  void
  local_apply(const MatrixFree<dim, Number> &,
              VectorType &dst,
              const VectorType &src,
              const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, degree, degree + 1, dim, Number> fe(*data);
    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        fe.reinit(cell);
        fe.read_dof_values(src);
        fe.evaluate(EvaluationFlags::gradients);
        for (unsigned int q = 0; q < fe.n_q_points; ++q)
          (*this)(&fe, q);
        fe.integrate(EvaluationFlags::gradients);
        fe.distribute_local_to_global(dst);
      }
  }

  std::shared_ptr<MatrixFree<dim, Number>> data;
  const Coefficient<dim, Number>          *lambda_values = nullptr;
  const Coefficient<dim, Number>          *mu_values = nullptr;
  std::shared_ptr<DiagonalMatrix<VectorType>> inverse_diagonal;
};


template <int dim, int degree, CoefficientLayout layout>
void
run_portable(const ConditionalOStream &pcout,
             const bool transfer_coefficients_from_finest)
{
  using VectorType =
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;
  using Operator = ElasticityOperator<dim, degree, layout>;
  using Smoother =
    PreconditionChebyshev<Operator, VectorType, DiagonalMatrix<VectorType>>;
  using Transfer = MGTransferMatrixFree<dim, double, MemorySpace::Default>;

  Timer setup_timer(MPI_COMM_WORLD);
  ExampleTriangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0., 1.);
  tria.refine_global(4);

  MappingQ1<dim>  mapping;
  const FE_Q<dim> scalar_fe(degree);
  FESystem<dim>   fe(scalar_fe, dim);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  pcout << "n_cells: " << tria.n_global_active_cells()
        << " n_dofs: " << dof_handler.n_dofs() << std::endl;
  pcout << "coefficient layout: "
        << (layout == CoefficientLayout::cell ? "cell" : "quadrature")
        << std::endl;

  AffineConstraints<double> constraints;
  constraints.reinit(dof_handler.locally_owned_dofs(),
                     DoFTools::extract_locally_relevant_dofs(dof_handler));
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
  constraints.close();

  auto data = std::make_shared<Portable::MatrixFree<dim, double>>();
  typename Portable::MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.mapping_update_flags = update_values | update_gradients;
  data->reinit(
    mapping, dof_handler, constraints, QGauss<1>(degree + 1), additional_data);

  const auto coarse =
    MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(tria);
  using CoefficientVector = typename Coefficient<dim>::VectorType;
  const FE_DGQ<dim> coefficient_fe(0);
  MGLevelObject<DoFHandler<dim>> lambda_dof_handlers, mu_dof_handlers;
  MGLevelObject<CoefficientVector> lambda_fields, mu_fields;
  initialize_cellwise_coefficient_field(lambda_dof_handlers,
                                        lambda_fields,
                                        coefficient_fe,
                                        coarse,
                                        LambdaCoefficientFunction<dim>());
  initialize_cellwise_coefficient_field(mu_dof_handlers,
                                        mu_fields,
                                        coefficient_fe,
                                        coarse,
                                        MuCoefficientFunction<dim>());
  if (transfer_coefficients_from_finest)
    {
      transfer_cellwise_coefficient_field(lambda_dof_handlers, lambda_fields);
      transfer_cellwise_coefficient_field(mu_dof_handlers, mu_fields);
    }
  const QGauss<dim> coefficient_quadrature(degree + 1);
  auto              pack_coefficient = [&](auto       &coefficient,
                              const auto &matrix_free,
                              const auto &dof_handler_level,
                              const auto &field) {
    if constexpr (layout == CoefficientLayout::cell)
      coefficient.reinit(matrix_free, dof_handler_level, field);
    else
      coefficient.reinit(matrix_free,
                         mapping,
                         dof_handler_level,
                         field,
                         coefficient_quadrature);
  };
  const unsigned int min_level = 0, max_level = coarse.size() - 1;
  MGLevelObject<Coefficient<dim>> lambda_coefficients(min_level, max_level);
  MGLevelObject<Coefficient<dim>> mu_coefficients(min_level, max_level);
  pack_coefficient(lambda_coefficients[max_level],
                   *data,
                   lambda_dof_handlers[max_level],
                   lambda_fields[max_level]);
  pack_coefficient(mu_coefficients[max_level],
                   *data,
                   mu_dof_handlers[max_level],
                   mu_fields[max_level]);

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host> exact_host;
  data->initialize_dof_vector(exact_host, 0);
  VectorTools::interpolate(dof_handler, ExactSolution<dim>(), exact_host);
  constraints.distribute(exact_host);

  VectorType rhs, solution;
  data->initialize_dof_vector(rhs, 0);
  data->initialize_dof_vector(solution, 0);
  solution.import_elements(exact_host, VectorOperation::insert);
  Operator op;
  op.reinit(data,
            lambda_coefficients[max_level],
            mu_coefficients[max_level]);
  // The discrete manufactured load is b = A I_h(u_exact), so the same
  // coefficient data and quadrature define the load and the operator.
  op.vmult(rhs, solution);
  solution = 0.;

  MGLevelObject<DoFHandler<dim>> mg_dofs(min_level, max_level);
  MGLevelObject<AffineConstraints<double>> mg_constraints(min_level, max_level);
  MGLevelObject<Operator>                  mg_ops(min_level, max_level);
  MGLevelObject<MGTwoLevelTransferCopyToHost<dim, VectorType>> transfers(
    min_level, max_level);
  std::vector<std::shared_ptr<Portable::MatrixFree<dim, double>>> level_data;
  for (unsigned int level = min_level; level <= max_level; ++level)
    {
      mg_dofs[level].reinit(*coarse[level]);
      mg_dofs[level].distribute_dofs(fe);
      mg_constraints[level].reinit(mg_dofs[level].locally_owned_dofs(),
                                   DoFTools::extract_locally_relevant_dofs(
                                     mg_dofs[level]));
      DoFTools::make_hanging_node_constraints(mg_dofs[level],
                                              mg_constraints[level]);
      DoFTools::make_zero_boundary_constraints(mg_dofs[level],
                                               mg_constraints[level]);
      mg_constraints[level].close();

      if (level == max_level)
        level_data.push_back(data);
      else
        {
          auto d = std::make_shared<Portable::MatrixFree<dim, double>>();
          d->reinit(mapping,
                    mg_dofs[level],
                    mg_constraints[level],
                    QGauss<1>(degree + 1),
                    additional_data);
          level_data.push_back(std::move(d));
        }
      if (level != max_level)
        {
          pack_coefficient(lambda_coefficients[level],
                           *level_data.back(),
                           lambda_dof_handlers[level],
                           lambda_fields[level]);
          pack_coefficient(mu_coefficients[level],
                           *level_data.back(),
                           mu_dof_handlers[level],
                           mu_fields[level]);
        }
      mg_ops[level].reinit(level_data.back(),
                           lambda_coefficients[level],
                           mu_coefficients[level]);
    }
  for (unsigned int level = min_level; level < max_level; ++level)
    transfers[level + 1].reinit(mg_dofs[level + 1],
                                mg_dofs[level],
                                mg_constraints[level + 1],
                                mg_constraints[level]);
  Transfer               transfer(transfers, [&](const auto level, auto &v) {
    mg_ops[level].initialize_dof_vector(v);
  });
  mg::Matrix<VectorType> mg_matrix(mg_ops);
  MGLevelObject<typename Smoother::AdditionalData> smoother_data(min_level,
                                                                 max_level);
  for (unsigned int level = min_level; level <= max_level; ++level)
    {
      mg_ops[level].compute_diagonal();
      smoother_data[level].preconditioner  = mg_ops[level].diagonal_inverse();
      smoother_data[level].smoothing_range = 20.;
      smoother_data[level].degree          = 5;
      smoother_data[level].eig_cg_n_iterations = 20;
      smoother_data[level].constraints.copy_from(mg_constraints[level]);
    }
  MGSmootherPrecondition<Operator, Smoother, VectorType> smoother;
  smoother.initialize(mg_ops, smoother_data);
  MGCoarseGridApplySmoother<VectorType> coarse_solver;
  coarse_solver.initialize(smoother);
  Multigrid<VectorType>                     mg(mg_matrix,
                           coarse_solver,
                           transfer,
                           smoother,
                           smoother,
                           min_level,
                           max_level);
  PreconditionMG<dim, VectorType, Transfer> preconditioner(dof_handler,
                                                           mg,
                                                           transfer);
  SolverControl                             control(100, 1e-10 * rhs.l2_norm());
  SolverCG<VectorType>                      solver(control);
  setup_timer.stop();
  pcout << "Setup and preconditioner: " << setup_timer.wall_time() << " s"
        << std::endl;
  Timer                                     solve_timer(MPI_COMM_WORLD);
  solver.solve(op, solution, rhs, preconditioner);
  solve_timer.stop();
  pcout << "Portable::MatrixFree solver converged in " << control.last_step()
        << " iterations in " << solve_timer.wall_time() << " s" << std::endl;

  Timer postprocess_timer(MPI_COMM_WORLD);
  LinearAlgebra::distributed::Vector<double, MemorySpace::Host> solution_host;
  data->initialize_dof_vector(solution_host, 0);
  solution_host.import_elements(solution, VectorOperation::insert);
  constraints.distribute(solution_host);
  solution_host.update_ghost_values();
  write_graphical_output(
    dof_handler,
    solution_host,
    lambda_dof_handlers[max_level],
    lambda_fields[max_level],
    mu_dof_handlers[max_level],
    mu_fields[max_level],
    (std::string(transfer_coefficients_from_finest ?
                   "elasticity_portable_transfer_" :
                   "elasticity_portable_evaluate_") +
     (layout == CoefficientLayout::cell ? "cell" : "quadrature")));
  Vector<double> cellwise_errors(tria.n_active_cells());
  VectorTools::integrate_difference(dof_handler,
                                    solution_host,
                                    ExactSolution<dim>(),
                                    cellwise_errors,
                                    QGauss<dim>(degree + 1),
                                    VectorTools::L2_norm);
  const double error = VectorTools::compute_global_error(tria,
                                                         cellwise_errors,
                                                         VectorTools::L2_norm);
  pcout << "Error: " << error << std::endl;
  postprocess_timer.stop();
  pcout << "Postprocessing: " << postprocess_timer.wall_time() << " s"
        << std::endl;
}


template <int dim, int degree>
void
run_matrixfree(const ConditionalOStream &pcout,
               const bool                transfer_coefficients_from_finest,
               const CoefficientLayout   layout)
{
  using VectorType =
    LinearAlgebra::distributed::Vector<double, MemorySpace::Host>;
  using Operator = MatrixFreeElasticityOperator<dim, degree>;
  using Smoother =
    PreconditionChebyshev<Operator, VectorType, DiagonalMatrix<VectorType>>;
  using Transfer = MGTransferMatrixFree<dim, double, MemorySpace::Host>;

  Timer setup_timer(MPI_COMM_WORLD);
  ExampleTriangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0., 1.);
  tria.refine_global(4);
  MappingQ1<dim> mapping;
  const FE_Q<dim> scalar_fe(degree);
  FESystem<dim> fe(scalar_fe, dim);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  pcout << "n_cells: " << tria.n_global_active_cells()
        << " n_dofs: " << dof_handler.n_dofs() << std::endl;
  pcout << "coefficient layout: "
        << (layout == CoefficientLayout::cell ? "cell" : "quadrature")
        << std::endl;

  AffineConstraints<double> constraints;
  constraints.reinit(dof_handler.locally_owned_dofs(),
                     DoFTools::extract_locally_relevant_dofs(dof_handler));
  DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
  constraints.close();

  auto data = std::make_shared<MatrixFree<dim, double>>();
  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.mapping_update_flags = update_values | update_gradients;
  data->reinit(mapping,
               dof_handler,
               constraints,
               QGauss<1>(degree + 1),
               additional_data);

  const auto coarse =
    MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(tria);
  using CoefficientVector = typename Coefficient<dim>::VectorType;
  const FE_DGQ<dim> coefficient_fe(0);
  MGLevelObject<DoFHandler<dim>> lambda_dof_handlers, mu_dof_handlers;
  MGLevelObject<CoefficientVector> lambda_fields, mu_fields;
  initialize_cellwise_coefficient_field(lambda_dof_handlers,
                                        lambda_fields,
                                        coefficient_fe,
                                        coarse,
                                        LambdaCoefficientFunction<dim>());
  initialize_cellwise_coefficient_field(mu_dof_handlers,
                                        mu_fields,
                                        coefficient_fe,
                                        coarse,
                                        MuCoefficientFunction<dim>());
  if (transfer_coefficients_from_finest)
    {
      transfer_cellwise_coefficient_field(lambda_dof_handlers, lambda_fields);
      transfer_cellwise_coefficient_field(mu_dof_handlers, mu_fields);
    }
  const QGauss<dim> coefficient_quadrature(degree + 1);
  auto              pack_coefficient = [&](auto       &coefficient,
                              const auto &matrix_free,
                              const auto &dof_handler_level,
                              const auto &field) {
    if (layout == CoefficientLayout::cell)
      coefficient.reinit(matrix_free, dof_handler_level, field);
    else
      coefficient.reinit(matrix_free,
                         mapping,
                         dof_handler_level,
                         field,
                         coefficient_quadrature);
  };
  const unsigned int min_level = 0, max_level = coarse.size() - 1;
  MGLevelObject<Coefficient<dim>> lambda_coefficients(min_level, max_level);
  MGLevelObject<Coefficient<dim>> mu_coefficients(min_level, max_level);
  pack_coefficient(lambda_coefficients[max_level],
                   *data,
                   lambda_dof_handlers[max_level],
                   lambda_fields[max_level]);
  pack_coefficient(mu_coefficients[max_level],
                   *data,
                   mu_dof_handlers[max_level],
                   mu_fields[max_level]);
  Operator op;
  op.reinit(data,
            lambda_coefficients[max_level],
            mu_coefficients[max_level]);

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host> exact_host;
  data->initialize_dof_vector(exact_host);
  VectorTools::interpolate(dof_handler, ExactSolution<dim>(), exact_host);
  constraints.distribute(exact_host);
  VectorType exact, rhs, solution;
  op.initialize_dof_vector(exact);
  op.initialize_dof_vector(rhs);
  op.initialize_dof_vector(solution);
  exact.import_elements(exact_host, VectorOperation::insert);
  // The discrete manufactured load is b = A I_h(u_exact).
  op.vmult(rhs, exact);

  MGLevelObject<DoFHandler<dim>> mg_dofs(min_level, max_level);
  MGLevelObject<AffineConstraints<double>> mg_constraints(min_level, max_level);
  MGLevelObject<Operator> mg_ops(min_level, max_level);
  MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers(min_level,
                                                              max_level);
  std::vector<std::shared_ptr<MatrixFree<dim, double>>> level_data;
  for (unsigned int level = min_level; level <= max_level; ++level)
    {
      mg_dofs[level].reinit(*coarse[level]);
      mg_dofs[level].distribute_dofs(fe);
      mg_constraints[level].reinit(mg_dofs[level].locally_owned_dofs(),
                                   DoFTools::extract_locally_relevant_dofs(
                                     mg_dofs[level]));
      DoFTools::make_zero_boundary_constraints(mg_dofs[level],
                                               mg_constraints[level]);
      mg_constraints[level].close();
      if (level == max_level)
        level_data.push_back(data);
      else
        {
          auto level_mf = std::make_shared<MatrixFree<dim, double>>();
          level_mf->reinit(mapping,
                           mg_dofs[level],
                           mg_constraints[level],
                           QGauss<1>(degree + 1),
                           additional_data);
          level_data.push_back(std::move(level_mf));
        }
      if (level != max_level)
        {
          pack_coefficient(lambda_coefficients[level],
                           *level_data.back(),
                           lambda_dof_handlers[level],
                           lambda_fields[level]);
          pack_coefficient(mu_coefficients[level],
                           *level_data.back(),
                           mu_dof_handlers[level],
                           mu_fields[level]);
        }
      mg_ops[level].reinit(level_data.back(),
                           lambda_coefficients[level],
                           mu_coefficients[level]);
    }
  for (unsigned int level = min_level; level < max_level; ++level)
    transfers[level + 1].reinit(mg_dofs[level + 1],
                                mg_dofs[level],
                                mg_constraints[level + 1],
                                mg_constraints[level]);
  Transfer transfer(transfers, [&](const auto level, auto &vector) {
    mg_ops[level].initialize_dof_vector(vector);
  });
  mg::Matrix<VectorType> mg_matrix(mg_ops);
  MGLevelObject<typename Smoother::AdditionalData> smoother_data(min_level,
                                                                 max_level);
  for (unsigned int level = min_level; level <= max_level; ++level)
    {
      mg_ops[level].compute_diagonal();
      smoother_data[level].preconditioner = mg_ops[level].diagonal_inverse();
      smoother_data[level].smoothing_range = 20.;
      smoother_data[level].degree = 5;
      smoother_data[level].eig_cg_n_iterations = 20;
      smoother_data[level].constraints.copy_from(mg_constraints[level]);
    }
  MGSmootherPrecondition<Operator, Smoother, VectorType> smoother;
  smoother.initialize(mg_ops, smoother_data);
  MGCoarseGridApplySmoother<VectorType> coarse_solver;
  coarse_solver.initialize(smoother);
  Multigrid<VectorType> mg(
    mg_matrix, coarse_solver, transfer, smoother, smoother, min_level, max_level);
  PreconditionMG<dim, VectorType, Transfer> preconditioner(dof_handler,
                                                           mg,
                                                           transfer);
  SolverControl control(100, 1e-10 * rhs.l2_norm());
  SolverCG<VectorType> solver(control);
  setup_timer.stop();
  pcout << "Setup and preconditioner: " << setup_timer.wall_time() << " s"
        << std::endl;
  Timer solve_timer(MPI_COMM_WORLD);
  solver.solve(op, solution, rhs, preconditioner);
  solve_timer.stop();
  pcout << "MatrixFree solver converged in " << control.last_step()
        << " iterations in " << solve_timer.wall_time() << " s" << std::endl;

  Timer postprocess_timer(MPI_COMM_WORLD);
  LinearAlgebra::distributed::Vector<double, MemorySpace::Host> solution_host;
  data->initialize_dof_vector(solution_host);
  solution_host.import_elements(solution, VectorOperation::insert);
  constraints.distribute(solution_host);
  solution_host.update_ghost_values();
  write_graphical_output(
    dof_handler,
    solution_host,
    lambda_dof_handlers[max_level],
    lambda_fields[max_level],
    mu_dof_handlers[max_level],
    mu_fields[max_level],
    (std::string(transfer_coefficients_from_finest ?
                   "elasticity_matrixfree_transfer_" :
                   "elasticity_matrixfree_evaluate_") +
     (layout == CoefficientLayout::cell ? "cell" : "quadrature")));
  Vector<double> cellwise_errors(tria.n_active_cells());
  VectorTools::integrate_difference(dof_handler,
                                    solution_host,
                                    ExactSolution<dim>(),
                                    cellwise_errors,
                                    QGauss<dim>(degree + 1),
                                    VectorTools::L2_norm);
  pcout << "Error: " << VectorTools::compute_global_error(
                         tria, cellwise_errors, VectorTools::L2_norm)
        << std::endl;
  postprocess_timer.stop();
  pcout << "Postprocessing: " << postprocess_timer.wall_time() << " s"
        << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);
  pcout << std::setprecision(6);
  const std::string backend = argc > 1 ? argv[1] : "portable";
  const std::string coefficient_policy = argc > 2 ? argv[2] : "evaluate";
  const std::string coefficient_layout = argc > 3 ? argv[3] : "cell";
  const bool transfer_coefficients_from_finest =
    coefficient_policy == "transfer";
  if (coefficient_policy != "evaluate" && coefficient_policy != "transfer")
    throw ExcMessage("Coefficient policy must be 'evaluate' or 'transfer'.");
  if (coefficient_layout != "cell" && coefficient_layout != "quadrature")
    throw ExcMessage("Coefficient layout must be 'cell' or 'quadrature'.");
  const CoefficientLayout layout = coefficient_layout == "quadrature" ?
                                     CoefficientLayout::quadrature :
                                     CoefficientLayout::cell;
  Timer timer(MPI_COMM_WORLD);
  if (backend == "portable")
    {
      if (layout == CoefficientLayout::quadrature)
        run_portable<3, 1, CoefficientLayout::quadrature>(
          pcout, transfer_coefficients_from_finest);
      else
        run_portable<3, 1, CoefficientLayout::cell>(
          pcout, transfer_coefficients_from_finest);
    }
  else if (backend == "matrixfree")
    run_matrixfree<3, 1>(pcout, transfer_coefficients_from_finest, layout);
  else
    throw ExcMessage("Backend must be 'portable' or 'matrixfree'.");
  timer.stop();
  pcout << "Total time: " << timer.wall_time() << " s" << std::endl;
}
