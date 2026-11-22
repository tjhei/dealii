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

#include <deal.II/matrix_free/coefficient.h>
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
using MatrixFreeTools::Coefficient;
using MatrixFreeTools::CoefficientLayout;
using MatrixFreeTools::DeviceCoefficient;

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
    double value(const Point<dim> &p, const unsigned int = 0) const override
    {
      return 1. + p[0];
    }
  };

  template <int dim>
  class MuCoefficientFunction : public Function<dim>
  {
  public:
    double value(const Point<dim> &p, const unsigned int = 0) const override
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

    double value(const Point<dim>  &p,
                 const unsigned int component = 0) const override
    {
      AssertIndexRange(component, dim);
      double result = 1.;
      for (unsigned int d = 0; d < dim; ++d)
        result *= p[d] * (1. - p[d]);
      return result;
    }
  };


  template <int dim, typename CoarseSequence>
  void initialize_cellwise_coefficient_field(
    MGLevelObject<DoFHandler<dim>>                       &dof_handlers,
    MGLevelObject<typename Coefficient<dim>::VectorType> &fields,
    const FE_DGQ<dim>                                    &finite_element,
    const CoarseSequence                                 &coarse,
    const unsigned int                                    min_level,
    const Function<dim>                                  &function)
  {
    const unsigned int max_level = coarse.size() - 1;
    dof_handlers.resize(min_level, max_level);
    fields.resize(min_level, max_level);
    for (unsigned int level = min_level; level <= max_level; ++level)
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


  template <int dim, typename MatrixFreeType>
  MGLevelObject<Coefficient<dim>> initialize_coefficients(
    const MGLevelObject<std::shared_ptr<MatrixFreeType>>       &level_data,
    const MGLevelObject<DoFHandler<dim>>                       &dof_handlers,
    const MGLevelObject<typename Coefficient<dim>::VectorType> &fields,
    const Mapping<dim>                                         &mapping,
    const Quadrature<dim>                                      &quadrature,
    const CoefficientLayout                                     layout,
    const bool transfer_from_finest)
  {
    const unsigned int              min_level = level_data.min_level();
    const unsigned int              max_level = level_data.max_level();
    MGLevelObject<Coefficient<dim>> coefficients(min_level, max_level);
    for (unsigned int level = transfer_from_finest ? max_level : min_level;
         level <= max_level;
         ++level)
      if (layout == CoefficientLayout::cell)
        coefficients[level].reinit(*level_data[level],
                                   dof_handlers[level],
                                   fields[level]);
      else
        coefficients[level].reinit(*level_data[level],
                                   mapping,
                                   dof_handlers[level],
                                   fields[level],
                                   quadrature);

    if (transfer_from_finest)
      return MatrixFreeTools::interpolate_cellwise_coefficients_to_mg(
        level_data, coefficients[max_level]);
    return coefficients;
  }


  template <int dim, typename VectorType>
  void write_graphical_output(
    const DoFHandler<dim>                       &dof_handler,
    const VectorType                            &solution,
    const DoFHandler<dim>                       &lambda_dof_handler,
    const typename Coefficient<dim>::VectorType &lambda_field,
    const DoFHandler<dim>                       &mu_dof_handler,
    const typename Coefficient<dim>::VectorType &mu_field,
    const std::string                           &file_prefix)
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

  DEAL_II_HOST_DEVICE void operator()(
    Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, dim, Number> *fe_eval,
    const int q_point) const
  {
    const unsigned int           cell_index = fe_eval->get_current_cell_index();
    const Tensor<2, dim, Number> gradient   = fe_eval->get_gradient(q_point);
    Tensor<2, dim, Number>       stress =
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
  void operator()(const typename Portable::MatrixFree<dim, Number>::Data *data,
                  const Portable::DeviceVector<Number>                   &src,
                  Portable::DeviceVector<Number> &dst) const
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

template <int dim, int degree, typename Number = double>
class ElasticityOperator : public EnableObserverPointer
{
public:
  using VectorType =
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

  ElasticityOperator() = default;

  void reinit(std::shared_ptr<Portable::MatrixFree<dim, Number>> data_in,
              const Coefficient<dim, Number> &lambda_coefficient,
              const Coefficient<dim, Number> &mu_coefficient)
  {
    Assert(lambda_coefficient.get_layout() == mu_coefficient.get_layout(),
           ExcMessage("The coefficient layouts must match."));
    data          = std::move(data_in);
    lambda_values = &lambda_coefficient;
    mu_values     = &mu_coefficient;
  }

  void initialize_dof_vector(VectorType &v) const
  {
    data->initialize_dof_vector(v, 0);
  }

  types::global_dof_index m() const
  {
    return data->get_dof_handler(0).n_dofs();
  }

  double el(const types::global_dof_index row,
            const types::global_dof_index column) const
  {
    Assert(row == column, ExcNotImplemented());
    Assert(inverse_diagonal.get() != nullptr, ExcNotInitialized());
    return 1. / (*inverse_diagonal)(row, row);
  }

  void vmult(VectorType &dst, const VectorType &src) const
  {
    dst = 0.;
    if (lambda_values->get_layout() == CoefficientLayout::cell)
      apply<CoefficientLayout::cell>(dst, src);
    else
      apply<CoefficientLayout::quadrature>(dst, src);
    data->copy_constrained_values(src, dst, 0);
  }

  void Tvmult(VectorType &dst, const VectorType &src) const
  {
    vmult(dst, src);
  }

  void compute_diagonal()
  {
    inverse_diagonal     = std::make_shared<DiagonalMatrix<VectorType>>();
    VectorType &diagonal = inverse_diagonal->get_vector();
    data->initialize_dof_vector(diagonal, 0);
    if (lambda_values->get_layout() == CoefficientLayout::cell)
      compute_diagonal<CoefficientLayout::cell>(diagonal);
    else
      compute_diagonal<CoefficientLayout::quadrature>(diagonal);

    double *values = diagonal.get_values();
    Kokkos::parallel_for(
      "invert diagonal",
      diagonal.locally_owned_size(),
      KOKKOS_LAMBDA(const int i) {
        values[i] = (values[i] == 0.0) ? 1.0 : 1. / values[i];
      });
  }

  std::shared_ptr<DiagonalMatrix<VectorType>> diagonal_inverse() const
  {
    return inverse_diagonal;
  }

private:
  template <CoefficientLayout layout>
  void apply(VectorType &dst, const VectorType &src) const
  {
    data->cell_loop(ElasticityCellOperator<dim, degree, layout, Number>(
                      mu_values->template portable_view<layout>(),
                      lambda_values->template portable_view<layout>()),
                    src,
                    dst);
  }

  template <CoefficientLayout layout>
  void compute_diagonal(VectorType &diagonal) const
  {
    ElasticityCellOperatorQuadrature<dim, degree, layout, Number> quad_operator(
      mu_values->template portable_view<layout>(),
      lambda_values->template portable_view<layout>());

    MatrixFreeTools::compute_diagonal<dim, degree, degree + 1, dim, Number>(
      *data,
      diagonal,
      quad_operator,
      EvaluationFlags::gradients,
      EvaluationFlags::gradients);
  }

  std::shared_ptr<Portable::MatrixFree<dim, Number>> data;
  std::shared_ptr<DiagonalMatrix<VectorType>>        inverse_diagonal;
  const Coefficient<dim, Number>                    *lambda_values = nullptr;
  const Coefficient<dim, Number>                    *mu_values     = nullptr;
};


template <int dim, int degree, typename Number = double>
class MatrixFreeElasticityOperator : public EnableObserverPointer
{
public:
  using VectorType =
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Host>;
  void reinit(std::shared_ptr<MatrixFree<dim, Number>> data_in,
              const Coefficient<dim, Number>          &lambda_coefficient,
              const Coefficient<dim, Number>          &mu_coefficient)
  {
    data          = std::move(data_in);
    lambda_values = &lambda_coefficient;
    mu_values     = &mu_coefficient;
  }

  void initialize_dof_vector(VectorType &vector) const
  {
    data->initialize_dof_vector(vector);
  }

  types::global_dof_index m() const
  {
    return data->get_dof_handler().n_dofs();
  }

  double el(const types::global_dof_index row,
            const types::global_dof_index column) const
  {
    Assert(row == column, ExcNotImplemented());
    Assert(inverse_diagonal.get() != nullptr, ExcNotInitialized());
    return 1. / (*inverse_diagonal)(row, row);
  }

  void vmult(VectorType &dst, const VectorType &src) const
  {
    data->cell_loop(
      &MatrixFreeElasticityOperator::local_apply, this, dst, src, true);
    data->get_affine_constraints().set_zero(dst);
  }

  void Tvmult(VectorType &dst, const VectorType &src) const
  {
    vmult(dst, src);
  }

  void compute_diagonal()
  {
    inverse_diagonal     = std::make_shared<DiagonalMatrix<VectorType>>();
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
      *data, diagonal, cell_operator, 0, 0, 0, 0);
    for (unsigned int i = 0; i < diagonal.locally_owned_size(); ++i)
      diagonal.local_element(i) =
        diagonal.local_element(i) == 0. ? 1. : 1. / diagonal.local_element(i);
  }

  std::shared_ptr<DiagonalMatrix<VectorType>> diagonal_inverse() const
  {
    return inverse_diagonal;
  }

  void operator()(FEEvaluation<dim, degree, degree + 1, dim, Number> *fe,
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
              VectorType                                  &dst,
              const VectorType                            &src,
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

  std::shared_ptr<MatrixFree<dim, Number>>    data;
  const Coefficient<dim, Number>             *lambda_values = nullptr;
  const Coefficient<dim, Number>             *mu_values     = nullptr;
  std::shared_ptr<DiagonalMatrix<VectorType>> inverse_diagonal;
};


template <int dim, int degree>
void run_portable(const ConditionalOStream &pcout,
                  const bool                transfer_coefficients_from_finest,
                  const CoefficientLayout   layout)
{
  using VectorType =
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;
  using Operator = ElasticityOperator<dim, degree>;
  using Smoother =
    PreconditionChebyshev<Operator, VectorType, DiagonalMatrix<VectorType>>;
  using Transfer = MGTransferMatrixFree<dim, double, MemorySpace::Default>;

  Timer                     setup_timer(MPI_COMM_WORLD);
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
  pcout << "input coefficient layout: "
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
  additional_data.mapping_update_flags =
    update_values | update_gradients | update_JxW_values;
  data->reinit(
    mapping, dof_handler, constraints, QGauss<1>(degree + 1), additional_data);

  const auto coarse =
    MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(tria);
  const unsigned int min_level = 0, max_level = coarse.size() - 1;
  const unsigned int min_field_level =
    transfer_coefficients_from_finest ? max_level : min_level;
  using CoefficientVector = typename Coefficient<dim>::VectorType;
  const FE_DGQ<dim>                coefficient_fe(0);
  MGLevelObject<DoFHandler<dim>>   lambda_dof_handlers, mu_dof_handlers;
  MGLevelObject<CoefficientVector> lambda_fields, mu_fields;
  initialize_cellwise_coefficient_field(lambda_dof_handlers,
                                        lambda_fields,
                                        coefficient_fe,
                                        coarse,
                                        min_field_level,
                                        LambdaCoefficientFunction<dim>());
  initialize_cellwise_coefficient_field(mu_dof_handlers,
                                        mu_fields,
                                        coefficient_fe,
                                        coarse,
                                        min_field_level,
                                        MuCoefficientFunction<dim>());
  MGLevelObject<DoFHandler<dim>>           mg_dofs(min_level, max_level);
  MGLevelObject<AffineConstraints<double>> mg_constraints(min_level, max_level);
  MGLevelObject<MGTwoLevelTransferCopyToHost<dim, VectorType>> transfers(
    min_level, max_level);
  MGLevelObject<std::shared_ptr<Portable::MatrixFree<dim, double>>> level_data(
    min_level, max_level);
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
        level_data[level] = data;
      else
        {
          auto d = std::make_shared<Portable::MatrixFree<dim, double>>();
          d->reinit(mapping,
                    mg_dofs[level],
                    mg_constraints[level],
                    QGauss<1>(degree + 1),
                    additional_data);
          level_data[level] = std::move(d);
        }
    }

  const QGauss<dim> coefficient_quadrature(degree + 1);
  const auto        lambda_coefficients =
    initialize_coefficients(level_data,
                            lambda_dof_handlers,
                            lambda_fields,
                            mapping,
                            coefficient_quadrature,
                            layout,
                            transfer_coefficients_from_finest);
  const auto mu_coefficients =
    initialize_coefficients(level_data,
                            mu_dof_handlers,
                            mu_fields,
                            mapping,
                            coefficient_quadrature,
                            layout,
                            transfer_coefficients_from_finest);
  MGLevelObject<Operator> mg_ops(min_level, max_level);
  for (unsigned int level = min_level; level <= max_level; ++level)
    mg_ops[level].reinit(level_data[level],
                         lambda_coefficients[level],
                         mu_coefficients[level]);

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host> exact_host;
  data->initialize_dof_vector(exact_host, 0);
  VectorTools::interpolate(dof_handler, ExactSolution<dim>(), exact_host);
  constraints.distribute(exact_host);

  VectorType rhs, solution;
  data->initialize_dof_vector(rhs, 0);
  data->initialize_dof_vector(solution, 0);
  solution.import_elements(exact_host, VectorOperation::insert);
  Operator op;
  op.reinit(data, lambda_coefficients[max_level], mu_coefficients[max_level]);
  // The discrete manufactured load is b = A I_h(u_exact), so the same
  // coefficient data and quadrature define the load and the operator.
  op.vmult(rhs, solution);
  solution = 0.;

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
  Timer solve_timer(MPI_COMM_WORLD);
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
  write_graphical_output(dof_handler,
                         solution_host,
                         lambda_dof_handlers[max_level],
                         lambda_fields[max_level],
                         mu_dof_handlers[max_level],
                         mu_fields[max_level],
                         (std::string(transfer_coefficients_from_finest ?
                                        "elasticity_portable_transfer_" :
                                        "elasticity_portable_evaluate_") +
                          (layout == CoefficientLayout::cell ? "cell" :
                                                               "quadrature")));
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
void run_matrixfree(const ConditionalOStream &pcout,
                    const bool                transfer_coefficients_from_finest,
                    const CoefficientLayout   layout)
{
  using VectorType =
    LinearAlgebra::distributed::Vector<double, MemorySpace::Host>;
  using Operator = MatrixFreeElasticityOperator<dim, degree>;
  using Smoother =
    PreconditionChebyshev<Operator, VectorType, DiagonalMatrix<VectorType>>;
  using Transfer = MGTransferMatrixFree<dim, double, MemorySpace::Host>;

  Timer                     setup_timer(MPI_COMM_WORLD);
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
  pcout << "input coefficient layout: "
        << (layout == CoefficientLayout::cell ? "cell" : "quadrature")
        << std::endl;

  AffineConstraints<double> constraints;
  constraints.reinit(dof_handler.locally_owned_dofs(),
                     DoFTools::extract_locally_relevant_dofs(dof_handler));
  DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
  constraints.close();

  auto data = std::make_shared<MatrixFree<dim, double>>();
  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.mapping_update_flags =
    update_values | update_gradients | update_JxW_values;
  data->reinit(
    mapping, dof_handler, constraints, QGauss<1>(degree + 1), additional_data);

  const auto coarse =
    MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(tria);
  const unsigned int min_level = 0, max_level = coarse.size() - 1;
  const unsigned int min_field_level =
    transfer_coefficients_from_finest ? max_level : min_level;
  using CoefficientVector = typename Coefficient<dim>::VectorType;
  const FE_DGQ<dim>                coefficient_fe(0);
  MGLevelObject<DoFHandler<dim>>   lambda_dof_handlers, mu_dof_handlers;
  MGLevelObject<CoefficientVector> lambda_fields, mu_fields;
  initialize_cellwise_coefficient_field(lambda_dof_handlers,
                                        lambda_fields,
                                        coefficient_fe,
                                        coarse,
                                        min_field_level,
                                        LambdaCoefficientFunction<dim>());
  initialize_cellwise_coefficient_field(mu_dof_handlers,
                                        mu_fields,
                                        coefficient_fe,
                                        coarse,
                                        min_field_level,
                                        MuCoefficientFunction<dim>());
  MGLevelObject<DoFHandler<dim>>           mg_dofs(min_level, max_level);
  MGLevelObject<AffineConstraints<double>> mg_constraints(min_level, max_level);
  MGLevelObject<MGTwoLevelTransfer<dim, VectorType>>      transfers(min_level,
                                                                    max_level);
  MGLevelObject<std::shared_ptr<MatrixFree<dim, double>>> level_data(min_level,
                                                                     max_level);
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
        level_data[level] = data;
      else
        {
          auto level_mf = std::make_shared<MatrixFree<dim, double>>();
          level_mf->reinit(mapping,
                           mg_dofs[level],
                           mg_constraints[level],
                           QGauss<1>(degree + 1),
                           additional_data);
          level_data[level] = std::move(level_mf);
        }
    }

  const QGauss<dim> coefficient_quadrature(degree + 1);
  const auto        lambda_coefficients =
    initialize_coefficients(level_data,
                            lambda_dof_handlers,
                            lambda_fields,
                            mapping,
                            coefficient_quadrature,
                            layout,
                            transfer_coefficients_from_finest);
  const auto mu_coefficients =
    initialize_coefficients(level_data,
                            mu_dof_handlers,
                            mu_fields,
                            mapping,
                            coefficient_quadrature,
                            layout,
                            transfer_coefficients_from_finest);
  MGLevelObject<Operator> mg_ops(min_level, max_level);
  for (unsigned int level = min_level; level <= max_level; ++level)
    mg_ops[level].reinit(level_data[level],
                         lambda_coefficients[level],
                         mu_coefficients[level]);

  Operator op;
  op.reinit(data, lambda_coefficients[max_level], mu_coefficients[max_level]);

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

  for (unsigned int level = min_level; level < max_level; ++level)
    transfers[level + 1].reinit(mg_dofs[level + 1],
                                mg_dofs[level],
                                mg_constraints[level + 1],
                                mg_constraints[level]);
  Transfer transfer(transfers, [&](const auto level, auto &vector) {
    mg_ops[level].initialize_dof_vector(vector);
  });
  mg::Matrix<VectorType>                           mg_matrix(mg_ops);
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
  write_graphical_output(dof_handler,
                         solution_host,
                         lambda_dof_handlers[max_level],
                         lambda_fields[max_level],
                         mu_dof_handlers[max_level],
                         mu_fields[max_level],
                         (std::string(transfer_coefficients_from_finest ?
                                        "elasticity_matrixfree_transfer_" :
                                        "elasticity_matrixfree_evaluate_") +
                          (layout == CoefficientLayout::cell ? "cell" :
                                                               "quadrature")));
  Vector<double> cellwise_errors(tria.n_active_cells());
  VectorTools::integrate_difference(dof_handler,
                                    solution_host,
                                    ExactSolution<dim>(),
                                    cellwise_errors,
                                    QGauss<dim>(degree + 1),
                                    VectorTools::L2_norm);
  pcout << "Error: "
        << VectorTools::compute_global_error(tria,
                                             cellwise_errors,
                                             VectorTools::L2_norm)
        << std::endl;
  postprocess_timer.stop();
  pcout << "Postprocessing: " << postprocess_timer.wall_time() << " s"
        << std::endl;
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);
  pcout << std::setprecision(6);
  const std::string backend            = argc > 1 ? argv[1] : "portable";
  const std::string coefficient_policy = argc > 2 ? argv[2] : "evaluate";
  const std::string coefficient_layout = argc > 3 ? argv[3] : "cell";
  const bool        transfer_coefficients_from_finest =
    coefficient_policy == "transfer";
  if (coefficient_policy != "evaluate" && coefficient_policy != "transfer")
    throw ExcMessage("Coefficient policy must be 'evaluate' or 'transfer'.");
  if (coefficient_layout != "cell" && coefficient_layout != "quadrature")
    throw ExcMessage("Coefficient layout must be 'cell' or 'quadrature'.");
  const CoefficientLayout layout = coefficient_layout == "quadrature" ?
                                     CoefficientLayout::quadrature :
                                     CoefficientLayout::cell;
  Timer                   timer(MPI_COMM_WORLD);
  if (backend == "portable")
    run_portable<3, 1>(pcout, transfer_coefficients_from_finest, layout);
  else if (backend == "matrixfree")
    run_matrixfree<3, 1>(pcout, transfer_coefficients_from_finest, layout);
  else
    throw ExcMessage("Backend must be 'portable' or 'matrixfree'.");
  timer.stop();
  pcout << "Total time: " << timer.wall_time() << " s" << std::endl;
}
