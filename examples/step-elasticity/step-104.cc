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


  // Keep the distributed FE fields as the canonical representation. The two
  // reinit() overloads pack them into the access formats used by each backend:
  // Kokkos views for Portable::MatrixFree and MatrixFree cell-batch data for
  // conventional MatrixFree.
  template <int dim, typename Number = double>
  class Coefficient
  {
  public:
    using VectorType =
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Host>;
    using View = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>;
    using CellArray = std::array<Number, VectorizedArray<Number>::size()>;

    Coefficient()
      : coefficient_fe(0)
    {}

    void
    initialize_fe_fields(const Triangulation<dim> &tria)
    {
      lambda_dof_handler.reinit(tria);
      mu_dof_handler.reinit(tria);
      lambda_dof_handler.distribute_dofs(coefficient_fe);
      mu_dof_handler.distribute_dofs(coefficient_fe);
      lambda_field.reinit(std::make_shared<Utilities::MPI::Partitioner>(
        lambda_dof_handler.locally_owned_dofs(),
        DoFTools::extract_locally_relevant_dofs(lambda_dof_handler),
        lambda_dof_handler.get_mpi_communicator()));
      mu_field.reinit(std::make_shared<Utilities::MPI::Partitioner>(
        mu_dof_handler.locally_owned_dofs(),
        DoFTools::extract_locally_relevant_dofs(mu_dof_handler),
        mu_dof_handler.get_mpi_communicator()));
      VectorTools::interpolate(lambda_dof_handler,
                               LambdaCoefficientFunction<dim>(),
                               lambda_field);
      VectorTools::interpolate(mu_dof_handler,
                               MuCoefficientFunction<dim>(),
                               mu_field);
      lambda_field.update_ghost_values();
      mu_field.update_ghost_values();

      refresh_cell_values();
    }

    const DoFHandler<dim> &
    get_lambda_dof_handler() const
    {
      return lambda_dof_handler;
    }

    const DoFHandler<dim> &
    get_mu_dof_handler() const
    {
      return mu_dof_handler;
    }

    const VectorType &
    get_lambda_field() const
    {
      return lambda_field;
    }

    const VectorType &
    get_mu_field() const
    {
      return mu_field;
    }

    void
    set_fields(const VectorType &lambda, const VectorType &mu)
    {
      lambda_field = lambda;
      mu_field     = mu;
      lambda_field.update_ghost_values();
      mu_field.update_ghost_values();
      refresh_cell_values();
    }

    void
    set_field_data(const DoFHandler<dim> &dof_handler,
                   const VectorType      &lambda,
                   const VectorType      &mu)
    {
      lambda_by_active_cell.resize(dof_handler.get_triangulation().n_active_cells());
      mu_by_active_cell.resize(dof_handler.get_triangulation().n_active_cells());
      std::vector<types::global_dof_index> indices(1);
      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned() || cell->is_ghost())
          {
            cell->get_dof_indices(indices);
            lambda_by_active_cell[cell->active_cell_index()] = lambda(indices[0]);
            mu_by_active_cell[cell->active_cell_index()] = mu(indices[0]);
          }
      values_initialized = true;
    }

    template <int degree>
    void
    reinit(const MatrixFree<dim, Number> &data)
    {
      if (!values_initialized)
        initialize_fe_fields(data.get_dof_handler(0).get_triangulation());
      data.initialize_cell_data_vector(lambda_cell_data);
      data.initialize_cell_data_vector(mu_cell_data);
      FEEvaluation<dim, degree, degree + 1, dim, Number> fe_eval(data);
      const unsigned int n_batches =
        data.n_cell_batches() + data.n_ghost_cell_batches();
      for (unsigned int batch = 0; batch < n_batches; ++batch)
        {
          fe_eval.reinit(batch);
          CellArray lambda_local{}, mu_local{};
          for (unsigned int lane = 0;
               lane < data.n_active_entries_per_cell_batch(batch);
               ++lane)
            {
              const auto cell = data.get_cell_iterator(batch, lane);
              lambda_local[lane] =
                lambda_by_active_cell[cell->active_cell_index()];
              mu_local[lane] = mu_by_active_cell[cell->active_cell_index()];
            }
          fe_eval.set_cell_data(lambda_cell_data, lambda_local);
          fe_eval.set_cell_data(mu_cell_data, mu_local);
        }
    }

    void
    reinit(const Portable::MatrixFree<dim, Number> &data)
    {
      if (!values_initialized)
        initialize_fe_fields(data.get_dof_handler(0).get_triangulation());
      const unsigned int padding = data.get_padding_length();
      unsigned int       n_cells = 0;
      for (unsigned int color = 0; data.n_cells_per_color(color) > 0; ++color)
        n_cells += ((data.n_cells_per_color(color) + padding - 1) / padding) *
                   padding;

      lambda_values = View("lambda cell data", n_cells);
      mu_values     = View("mu cell data", n_cells);
      auto lambda_host = Kokkos::create_mirror_view(lambda_values);
      auto mu_host     = Kokkos::create_mirror_view(mu_values);

      unsigned int offset = 0;
      for (unsigned int color = 0; data.n_cells_per_color(color) > 0; ++color)
        {
          for (unsigned int cell_index = 0;
               cell_index < data.n_cells_per_color(color);
               ++cell_index)
            {
              const auto cell = data.get_cell_iterator(color, cell_index);
              lambda_host[offset + cell_index] =
                lambda_by_active_cell[cell->active_cell_index()];
              mu_host[offset + cell_index] =
                mu_by_active_cell[cell->active_cell_index()];
            }
          offset += ((data.n_cells_per_color(color) + padding - 1) / padding) *
                    padding;
        }
      Kokkos::deep_copy(lambda_values, lambda_host);
      Kokkos::deep_copy(mu_values, mu_host);
    }

    View lambda_values;
    View mu_values;
    AlignedVector<CellArray> lambda_cell_data;
    AlignedVector<CellArray> mu_cell_data;

  private:
    void
    refresh_cell_values()
    {
      lambda_by_active_cell.resize(
        lambda_dof_handler.get_triangulation().n_active_cells());
      mu_by_active_cell.resize(
        mu_dof_handler.get_triangulation().n_active_cells());
      std::vector<types::global_dof_index> indices(1);
      for (const auto &cell : lambda_dof_handler.active_cell_iterators())
        if (cell->is_locally_owned() || cell->is_ghost())
          {
            cell->get_dof_indices(indices);
            lambda_by_active_cell[cell->active_cell_index()] =
              lambda_field(indices[0]);
          }
      for (const auto &cell : mu_dof_handler.active_cell_iterators())
        if (cell->is_locally_owned() || cell->is_ghost())
          {
            cell->get_dof_indices(indices);
            mu_by_active_cell[cell->active_cell_index()] = mu_field(indices[0]);
          }
      values_initialized = true;
    }

    FE_DGQ<dim> coefficient_fe;
    DoFHandler<dim> lambda_dof_handler;
    DoFHandler<dim> mu_dof_handler;
    VectorType lambda_field;
    VectorType mu_field;
    std::vector<Number> lambda_by_active_cell;
    std::vector<Number> mu_by_active_cell;
    bool values_initialized = false;
  };


  template <int dim, typename VectorType>
  void
  write_graphical_output(const DoFHandler<dim> &dof_handler,
                         const VectorType     &solution,
                         const Coefficient<dim> &coefficients,
                         const std::string      &file_prefix)
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
    data_out.add_data_vector(coefficients.get_lambda_dof_handler(),
                             coefficients.get_lambda_field(),
                             "lambda");
    data_out.add_data_vector(coefficients.get_mu_dof_handler(),
                             coefficients.get_mu_field(),
                             "mu");
    data_out.build_patches();
    data_out.write_vtu_with_pvtu_record(
      "./", file_prefix, 0, MPI_COMM_WORLD, 4, 0);
  }
} // namespace


template <int dim, int fe_degree, typename Number>
class ElasticityCellOperatorQuadrature
{
public:
  DEAL_II_HOST_DEVICE
  ElasticityCellOperatorQuadrature(const typename Coefficient<dim, Number>::View
                                     &mu_values,
                                   const typename Coefficient<dim, Number>::View
                                     &lambda_values)
    : mu_values(mu_values)
    , lambda_values(lambda_values)
  {}

  DEAL_II_HOST_DEVICE void
  operator()(
    Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, dim, Number> *fe_eval,
    const int q_point) const
  {
    const auto *data = fe_eval->get_matrix_free_data();
    const unsigned int cell_index =
      data->precomputed_data[0].row_start /
        data->precomputed_data[0].padding_length +
      fe_eval->get_current_cell_index();
    const Tensor<2, dim, Number> gradient = fe_eval->get_gradient(q_point);
    Tensor<2, dim, Number> stress =
      mu_values(cell_index) * (gradient + transpose(gradient));
    const Number volumetric_stress = lambda_values(cell_index) * trace(gradient);
    for (unsigned int d = 0; d < dim; ++d)
      stress[d][d] += volumetric_stress;
    fe_eval->submit_gradient(stress, q_point);
  }

private:
  typename Coefficient<dim, Number>::View mu_values;
  typename Coefficient<dim, Number>::View lambda_values;
};

template <int dim, int degree, typename Number>
class ElasticityCellOperator
{
public:
  static constexpr unsigned int n_q_points = Utilities::pow(degree + 1, dim);

  ElasticityCellOperator(const typename Coefficient<dim, Number>::View &mu_values,
                         const typename Coefficient<dim, Number>::View
                           &lambda_values)
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

    ElasticityCellOperatorQuadrature<dim, degree, Number> quad_operator(
      mu_values, lambda_values);
    data->for_each_quad_point(
      [&](const int q_point) { quad_operator(&fe, q_point); });

    fe.integrate(EvaluationFlags::gradients);
    fe.distribute_local_to_global(dst);
  }

private:
  typename Coefficient<dim, Number>::View mu_values;
  typename Coefficient<dim, Number>::View lambda_values;
};

template <int dim, int degree, typename Number = double>
class ElasticityOperator : public EnableObserverPointer
{
public:
  using VectorType =
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

  ElasticityOperator() = default;

  explicit ElasticityOperator(
    std::shared_ptr<Portable::MatrixFree<dim, Number>> data)
  {
    reinit(std::move(data));
  }

  void
  reinit(std::shared_ptr<Portable::MatrixFree<dim, Number>> data_in)
  {
    data = std::move(data_in);
    coefficients.reinit(*data);
  }

  void
  reinit(std::shared_ptr<Portable::MatrixFree<dim, Number>> data_in,
         const Coefficient<dim, Number> &coefficient_source)
  {
    coefficients.set_field_data(coefficient_source.get_lambda_dof_handler(),
                                coefficient_source.get_lambda_field(),
                                coefficient_source.get_mu_field());
    reinit(std::move(data_in));
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
    data->cell_loop(ElasticityCellOperator<dim, degree, Number>(
                      coefficients.mu_values, coefficients.lambda_values),
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
    ElasticityCellOperatorQuadrature<dim, degree, Number> quad_operator(
      coefficients.mu_values, coefficients.lambda_values);

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
  Coefficient<dim, Number>                          coefficients;
};


template <int dim, int degree, typename Number = double>
class MatrixFreeElasticityOperator : public EnableObserverPointer
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Host>;
  using CellArray = typename Coefficient<dim, Number>::CellArray;

  void
  reinit(std::shared_ptr<MatrixFree<dim, Number>> data_in)
  {
    data = std::move(data_in);
    coefficients.template reinit<degree>(*data);
  }

  void
  reinit(std::shared_ptr<MatrixFree<dim, Number>> data_in,
         const Coefficient<dim, Number> &coefficient_source)
  {
    coefficients.set_field_data(coefficient_source.get_lambda_dof_handler(),
                                coefficient_source.get_lambda_field(),
                                coefficient_source.get_mu_field());
    reinit(std::move(data_in));
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
    const auto lambda_array = fe->read_cell_data(coefficients.lambda_cell_data);
    const auto mu_array     = fe->read_cell_data(coefficients.mu_cell_data);
    VectorizedArray<Number> lambda_value, mu_value;
    for (unsigned int lane = 0; lane < VectorizedArray<Number>::size(); ++lane)
      {
        lambda_value[lane] = lambda_array[lane];
        mu_value[lane]     = mu_array[lane];
      }
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
  Coefficient<dim, Number>                 coefficients;
  std::shared_ptr<DiagonalMatrix<VectorType>> inverse_diagonal;
};


template <int dim, typename CoarseSequence>
void
initialize_coefficient_levels(MGLevelObject<Coefficient<dim>> &coefficients,
                              const CoarseSequence &coarse,
                              const bool transfer_from_finest)
{
  const unsigned int min_level = 0;
  const unsigned int max_level = coarse.size() - 1;
  coefficients.resize(min_level, max_level);
  for (unsigned int level = min_level; level <= max_level; ++level)
    coefficients[level].initialize_fe_fields(*coarse[level]);

  if (transfer_from_finest)
    {
      using VectorType = typename Coefficient<dim>::VectorType;
      MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers(min_level,
                                                                    max_level);
      for (unsigned int level = min_level; level < max_level; ++level)
        transfers[level + 1].reinit(
          coefficients[level + 1].get_lambda_dof_handler(),
          coefficients[level].get_lambda_dof_handler());
      MGTransferMatrixFree<dim, double, MemorySpace::Host> transfer(
        transfers, [&](const auto level, auto &vector) {
          vector.reinit(
            coefficients[level].get_lambda_field().get_partitioner());
        });
      MGLevelObject<VectorType> lambda_levels(min_level, max_level);
      MGLevelObject<VectorType> mu_levels(min_level, max_level);
      VectorType lambda_source, mu_source;
      lambda_source.reinit(
        coefficients[max_level].get_lambda_field().get_partitioner());
      mu_source.reinit(coefficients[max_level].get_mu_field().get_partitioner());
      lambda_source.import_elements(coefficients[max_level].get_lambda_field(),
                                    VectorOperation::insert);
      mu_source.import_elements(coefficients[max_level].get_mu_field(),
                                VectorOperation::insert);
      transfer.interpolate_to_mg(lambda_levels,
                                 lambda_source);
      transfer.interpolate_to_mg(mu_levels, mu_source);
      for (unsigned int level = min_level; level <= max_level; ++level)
        coefficients[level].set_fields(lambda_levels[level], mu_levels[level]);
    }
}

template <int dim, int degree>
void
run_portable(const ConditionalOStream &pcout,
             const bool transfer_coefficients_from_finest)
{
  using VectorType =
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;
  using Operator = ElasticityOperator<dim, degree>;
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
  MGLevelObject<Coefficient<dim>> coefficient_levels(0, coarse.size() - 1);
  initialize_coefficient_levels(coefficient_levels,
                                coarse,
                                transfer_coefficients_from_finest);

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host> exact_host;
  data->initialize_dof_vector(exact_host, 0);
  VectorTools::interpolate(dof_handler, ExactSolution<dim>(), exact_host);
  constraints.distribute(exact_host);

  VectorType rhs, solution;
  data->initialize_dof_vector(rhs, 0);
  data->initialize_dof_vector(solution, 0);
  solution.import_elements(exact_host, VectorOperation::insert);
  Operator op;
  op.reinit(data, coefficient_levels[coarse.size() - 1]);
  // The discrete manufactured load is b = A I_h(u_exact), so the same
  // coefficient data and quadrature define the load and the operator.
  op.vmult(rhs, solution);
  solution = 0.;

  const unsigned int             min_level = 0, max_level = coarse.size() - 1;
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
      mg_ops[level].reinit(level_data.back(), coefficient_levels[level]);
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
    coefficient_levels[max_level],
    transfer_coefficients_from_finest ? "elasticity_portable_transfer" :
                                        "elasticity_portable_evaluate");
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
               const bool transfer_coefficients_from_finest)
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
  MGLevelObject<Coefficient<dim>> coefficient_levels(0, coarse.size() - 1);
  initialize_coefficient_levels(coefficient_levels,
                                coarse,
                                transfer_coefficients_from_finest);
  Operator op;
  op.reinit(data, coefficient_levels[coarse.size() - 1]);

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

  const unsigned int min_level = 0, max_level = coarse.size() - 1;
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
      mg_ops[level].reinit(level_data.back(), coefficient_levels[level]);
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
    coefficient_levels[max_level],
    transfer_coefficients_from_finest ? "elasticity_matrixfree_transfer" :
                                        "elasticity_matrixfree_evaluate");
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
  const bool transfer_coefficients_from_finest =
    coefficient_policy == "transfer";
  if (coefficient_policy != "evaluate" && coefficient_policy != "transfer")
    throw ExcMessage("Coefficient policy must be 'evaluate' or 'transfer'.");
  Timer timer(MPI_COMM_WORLD);
  if (backend == "portable")
    run_portable<3, 1>(pcout, transfer_coefficients_from_finest);
  else if (backend == "matrixfree")
    run_matrixfree<3, 1>(pcout, transfer_coefficients_from_finest);
  else
    throw ExcMessage("Backend must be 'portable' or 'matrixfree'.");
  timer.stop();
  pcout << "Total time: " << timer.wall_time() << " s" << std::endl;
}
