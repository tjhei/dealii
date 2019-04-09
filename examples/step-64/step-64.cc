/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Authors: Fabian Castelli, Karlsruhe Institute of Technology (KIT)
 */


// First the typical inclusions of the library files.

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_c1.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

// Followed by common C++ headers for in and output.
#include <iostream>
#include <fstream>

/**
 * -----------------------------------------------------------------------------
 * Bratu problem:
 *  -\Delta u = f(u)     in \Omega,
 *          u = u^D      on \ptl\Omega,
 * with
 *       f(u) = exp(u),
 *        u^D = 0.
 * No exact solution is knwon to this nonlinear PDE.
 * The problem is numerically solved with a standard FEM of arbitrary polynomial
 * degree.
 * The resulting linear system for the Newton update is solved with a CG method
 * together with a multigrid preconditioner.
 * The matrix-free framework of deal.II is used.
 * -----------------------------------------------------------------------------
 */


namespace stepXX
{
  using namespace dealii;
  
  
  
  // As in the previous examples to the MatrixFree framework we define the
  // finite element degree and the space dimension to know them at compile time.
  const unsigned int degree_finite_element = 4;
  const unsigned int dimension = 2;
  
  
  
  // sect3{Matrix-free operators}
  
  // In the beginning we define two classes, the first for the application of
  // the Jacobian and the second for the evaluation of the residual.
  
  // We derive the JacobianOperator from the MatrixFreeOperator::Base class to
  // have the right interface for the lienar solver and the multilevel
  // framework. So we have to provide an implementation for the compute_diagonal
  // and the apply_add function.
  
  // Solving a nonlinear problem with Newton's method requieres in each
  // iteration step the Jacobian, evaluated in the old Newton step, as system
  // matrix for the linear system of the current newton update. For this reason
  // we need to store the information of the previous Newton step somewhere in
  // this class before we can apply the vmult function in the linear solver.
  // Hence we implement a function called evaluate_nonlinearity to store the
  // values of the Newton step in the quadrature points in a table like we did
  // it in step-37, where we stored the values of the coefficent function.
  
  // Further we overload the vmult function from the base class to make use of
  // the efficient zeroing of the destination vector by passing true to the
  // zero_dst_vector flag of the cell_loop fucntion. As private functions we
  // implement the local_apply and the local_compute_diagonal fucntion, which we
  // call in the cell_loop for computing the matrix-vector product or the
  // diagonal.
  template <int dim, int fe_degree, typename number>
  class JacobianOperator : public MatrixFreeOperators::Base<dim,LinearAlgebra::distributed::Vector<number> >
  {
  public:
	typedef number value_type;
	
	JacobianOperator();
	
	virtual
	void
	clear() override;
	
	void
	evaluate_nonlinearity(const LinearAlgebra::distributed::Vector<number> &src);
	
	virtual
	void
	compute_diagonal() override;
	
	void
	vmult(LinearAlgebra::distributed::Vector<number> &dst, const LinearAlgebra::distributed::Vector<number> &src) const;
	
  private:
	virtual
	void
	apply_add(LinearAlgebra::distributed::Vector<number> &dst, const LinearAlgebra::distributed::Vector<number> &src) const override;
	
	void
	local_apply(const MatrixFree<dim,number> &data, LinearAlgebra::distributed::Vector<number> &dst, const LinearAlgebra::distributed::Vector<number> &src, const std::pair<unsigned int,unsigned int> &cell_range) const;
	
	void
	local_compute_diagonal(const MatrixFree<dim,number> &data, LinearAlgebra::distributed::Vector<number> &dst, const unsigned int &dummy, const std::pair<unsigned int,unsigned int> &cell_range) const;
	
	Table<2, VectorizedArray<number> > nl_values;
  };
  
  
  
  template <int dim, int fe_degree, typename number>
  JacobianOperator<dim,fe_degree,number>::JacobianOperator()
  :
  MatrixFreeOperators::Base<dim,LinearAlgebra::distributed::Vector<number> >()
  {}
  
  
  
  template <int dim, int fe_degree, typename number>
  void
  JacobianOperator<dim,fe_degree,number>::clear()
  {
	nl_values.reinit(0, 0);
	MatrixFreeOperators::Base<dim,LinearAlgebra::distributed::Vector<number> >::clear();
  }
  
  
  
  template <int dim, int fe_degree, typename number>
  void
  JacobianOperator<dim,fe_degree,number>::evaluate_nonlinearity(const LinearAlgebra::distributed::Vector<number> &src)
  {
	const unsigned int n_cells = this->data->n_macro_cells();
	FEEvaluation<dim,fe_degree,fe_degree+1,1,number> phi(*this->data);
	
	nl_values.reinit(n_cells, phi.n_q_points);
	for(unsigned int cell=0; cell<n_cells; ++cell)
	{
	  phi.reinit(cell);
	  phi.gather_evaluate(src, true, false);
	  
	  for(unsigned int q=0; q<phi.n_q_points; ++q)
		nl_values(cell, q) = std::exp(phi.get_value(q));
	}
  }
  
  
  
  template <int dim, int fe_degree, typename number>
  void
  JacobianOperator<dim,fe_degree,number>::local_apply(const MatrixFree<dim,number> &data, LinearAlgebra::distributed::Vector<number> &dst, const LinearAlgebra::distributed::Vector<number> &src, const std::pair<unsigned int,unsigned int> &cell_range) const
  {
	FEEvaluation<dim,fe_degree,fe_degree+1,1,number> phi(data);
	
	for(unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
	{
	  AssertDimension(nl_values.size(0), data.n_macro_cells());
	  AssertDimension(nl_values.size(1), phi.n_q_points);
	  
	  phi.reinit(cell);
	  phi.gather_evaluate(src, true, true);
	  
	  for(unsigned int q=0; q<phi.n_q_points; ++q)
	  {
		phi.submit_value(-nl_values(cell,q)*phi.get_value(q), q);
		phi.submit_gradient(phi.get_gradient(q), q);
	  }
	  
	  phi.integrate_scatter(true, true, dst);
	}
  }
  
  
  
  template <int dim, int fe_degree, typename number>
  void
  JacobianOperator<dim,fe_degree,number>::apply_add(LinearAlgebra::distributed::Vector<number> &dst, const LinearAlgebra::distributed::Vector<number> &src) const
  {
	this->data->cell_loop(&JacobianOperator::local_apply, this, dst, src, true);
  }
  
  
  
  template <int dim, int fe_degree, typename number>
  void
  JacobianOperator<dim,fe_degree,number>::vmult(LinearAlgebra::distributed::Vector<number> &dst, const LinearAlgebra::distributed::Vector<number> &src) const
  {
	AssertDimension(dst.size(), src.size());
	
	// Is not possible to imit this call!
	dst = 0.0; // Avoid this and measure time difference
	
	this->preprocess_constraints(dst, src);
	apply_add(dst, src);
	this->postprocess_constraints(dst, src);
  }
  
  
  
  template <int dim, int fe_degree, typename number>
  void
  JacobianOperator<dim,fe_degree,number>::compute_diagonal()
  {
	this->inverse_diagonal_entries.reset(new DiagonalMatrix<LinearAlgebra::distributed::Vector<number> >());
	LinearAlgebra::distributed::Vector<number> &inverse_diagonal = this->inverse_diagonal_entries->get_vector();
	this->data->initialize_dof_vector(inverse_diagonal);
	
	unsigned int dummy = 0;
	
	this->data->cell_loop(&JacobianOperator::local_compute_diagonal, this, inverse_diagonal, dummy);
	
	this->set_constrained_entries_to_one(inverse_diagonal);
	
	for(unsigned int i=0; i<inverse_diagonal.local_size(); ++i)
	{
	  Assert(inverse_diagonal.local_element(i) > 0., ExcMessage("No diagonal entry in a positive definite operator should be zero"));
	  inverse_diagonal.local_element(i) = 1./inverse_diagonal.local_element(i);
	}
  }
  
  
  
  template <int dim, int fe_degree, typename number>
  void
  JacobianOperator<dim,fe_degree,number>::local_compute_diagonal(const MatrixFree<dim,number> &data, LinearAlgebra::distributed::Vector<number> &dst, const unsigned int &, const std::pair<unsigned int,unsigned int> &cell_range) const
  {
	FEEvaluation<dim,fe_degree,fe_degree+1,1,number> phi(data);
	
	AlignedVector<VectorizedArray<number> > diagonal(phi.dofs_per_cell);
	
	for(unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
	{
	  AssertDimension(nl_values.size(0), data.n_macro_cells());
	  AssertDimension(nl_values.size(1), phi.n_q_points);
	  
	  phi.reinit(cell);
	  for(unsigned int i=0; i<phi.dofs_per_cell; ++i)
	  {
		for(unsigned int j=0; j<phi.dofs_per_cell; ++j)
		  phi.submit_dof_value(VectorizedArray<number>(), j);
		phi.submit_dof_value(make_vectorized_array<number>(1.), i);
		
		phi.evaluate(true, true);
		for(unsigned int q=0; q<phi.n_q_points; ++q)
		{
		  phi.submit_value(-nl_values(cell, q)*phi.get_value(q), q);
		  phi.submit_gradient(phi.get_gradient(q), q);
		}
		phi.integrate(true, true);
		diagonal[i] = phi.get_dof_value(i);
	  }
	  for(unsigned int i=0; i<phi.dofs_per_cell; ++i)
		phi.submit_dof_value(diagonal[i], i);
	  phi.distribute_local_to_global(dst);
	}
  }
  
  
  
  // With the next class we implement the ResidualOperator, which we use to
  // assemble the right hand side  and to evaluate the residual in the Newton
  // method. Since we do not need an matrix interface we can directly implement
  // the class.
  template <int dim, int fe_degree>
  class ResidualOperator
  {
  public:
	ResidualOperator() = default;
	
	void
	initialize(std::shared_ptr<const MatrixFree<dim,double> > data_in);
	
	void
	apply(LinearAlgebra::distributed::Vector<double> &dst, const LinearAlgebra::distributed::Vector<double> &src) const;
	
  private:
	void
	local_apply(const MatrixFree<dim,double> &data, LinearAlgebra::distributed::Vector<double> &dst, const LinearAlgebra::distributed::Vector<double> &src, const std::pair<unsigned int,unsigned int> &cell_range) const;
	
	std::shared_ptr<const MatrixFree<dim,double> > data;
  };
  
  
  
  template <int dim, int fe_degree>
  void
  ResidualOperator<dim,fe_degree>::initialize(std::shared_ptr<const MatrixFree<dim,double> > data_in)
  {
	Assert(data_in, ExcNotInitialized());
	
	data = data_in;
  }
  
  
  
  template <int dim, int fe_degree>
  void
  ResidualOperator<dim, fe_degree>::apply(LinearAlgebra::distributed::Vector<double> &dst, const LinearAlgebra::distributed::Vector<double> &src) const
  {
	Assert(data, ExcNotInitialized());
	
	data->cell_loop(&ResidualOperator<dim,fe_degree>::local_apply, this, dst, src, true);
  }
  
  
  
  template <int dim, int fe_degree>
  void
  ResidualOperator<dim, fe_degree>::local_apply(const MatrixFree<dim> &data, LinearAlgebra::distributed::Vector<double> &dst, const LinearAlgebra::distributed::Vector<double> &src, const std::pair<unsigned int,unsigned int> &cell_range) const
  {
	FEEvaluation<dim,fe_degree> phi(data);
	
	for(unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
	{
	  phi.reinit(cell);
	  phi.gather_evaluate(src, true, true);
	  
	  for(unsigned int q=0; q<phi.n_q_points; ++q)
	  {
		phi.submit_value(-std::exp(phi.get_value(q)), q);
		phi.submit_gradient(phi.get_gradient(q), q);
	  }
	  
	  phi.integrate_scatter(true, true, dst);
	}
  }
  
  
  
  
  // Following comes the solver class for the Gelfand problem.
  template <int dim>
  class GelfandProblem 
  {
  public:
	GelfandProblem();
	
	void
	run();
	
  private:
	void
	make_grid();
	
	void
	setup_system();
	
	void
	assemble_right_hand_side();
	
	double
	compute_residual(const double alpha);
	
	void
	compute_update();
	
	void
	solve();
	
	double
	compute_solution_norm();
	
	void
	output_results() const;
	
	// The member objects related to the discretization are here.
	parallel::distributed::Triangulation<dim>	triangulation;
	const MappingC1<dim>						mapping;
	FE_Q<dim>									fe;
	DoFHandler<dim>								dof_handler;
	
	// Then, we have the matrices and vectors related to the global discrete
	// system.
	ConstraintMatrix											constraints;
	JacobianOperator<dim,degree_finite_element,double>			system_matrix;
	ResidualOperator<dim,degree_finite_element>					residual_operator;
	
	MGConstrainedDoFs											mg_constrained_dofs;
	typedef JacobianOperator<dim,degree_finite_element,float>	LevelMatrixType;
	MGLevelObject<LevelMatrixType>								mg_matrices;
	MGLevelObject<LinearAlgebra::distributed::Vector<float> >	mg_solution;
	
	LinearAlgebra::distributed::Vector<double>					solution;
	LinearAlgebra::distributed::Vector<double>					newton_update;
	LinearAlgebra::distributed::Vector<double>					system_rhs;
	
	unsigned int		linear_iterations;
	
	ConditionalOStream	pcout;
	
	TimerOutput			computing_timer;
  };
  
  
  
  template <int dim>
  GelfandProblem<dim>::GelfandProblem()
  :
  triangulation(MPI_COMM_WORLD, Triangulation<dim>::limit_level_difference_at_vertices, parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  fe(degree_finite_element),
  dof_handler(triangulation),
  pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0),
  computing_timer(MPI_COMM_WORLD, pcout, TimerOutput::summary, TimerOutput::wall_times)
  {}
  
  
  
  template <int dim>
  void
  GelfandProblem<dim>::make_grid()
  {
	TimerOutput::Scope t(computing_timer,"make grid");
	
	SphericalManifold<dim> boundary_manifold;
	TransfiniteInterpolationManifold<dim> inner_manifold;
	
	GridGenerator::hyper_ball(triangulation);
	
	triangulation.set_all_manifold_ids(1);
	triangulation.set_all_manifold_ids_on_boundary(0);
	
	triangulation.set_manifold(0, boundary_manifold);
	
	inner_manifold.initialize(triangulation);
	triangulation.set_manifold(1, inner_manifold);
	
	// Alternative: Square domain.
	// GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
	
	triangulation.refine_global(5);
  }
  
  
  
  template <int dim>
  void
  GelfandProblem<dim>::setup_system()
  {
	TimerOutput::Scope t(computing_timer, "setup system");
	
	system_matrix.clear();
	mg_matrices.clear_elements();
	
	dof_handler.distribute_dofs(fe);
	dof_handler.distribute_mg_dofs();
	
	IndexSet locally_relevant_dofs;
	DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
	
	constraints.clear();
	constraints.reinit(locally_relevant_dofs);
	DoFTools::make_hanging_node_constraints(dof_handler, constraints);
	VectorTools::interpolate_boundary_values(dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
	constraints.close();
	
	{
	  typename MatrixFree<dim,double>::AdditionalData additional_data;
	  additional_data.tasks_parallel_scheme = MatrixFree<dim,double>::AdditionalData::partition_color;
	  additional_data.mapping_update_flags = (update_values | update_gradients | update_JxW_values | update_quadrature_points);
	  // Cannot activate also the boudnary flags for example for inhomogeneous Neumann boundary data.
// 	  additional_data.mapping_update_flags_boundary_faces = (update_values | update_gradients | update_JxW_values | update_quadrature_points);
	  
	  std::shared_ptr<MatrixFree<dim,double> > system_mf_storage(new MatrixFree<dim,double>());
	  system_mf_storage->reinit(dof_handler, constraints, QGauss<1>(fe.degree+1), additional_data);
	  system_matrix.initialize(system_mf_storage);
	  residual_operator.initialize(system_mf_storage);
	}
	
	system_matrix.initialize_dof_vector(solution);
	system_matrix.initialize_dof_vector(newton_update);
	system_matrix.initialize_dof_vector(system_rhs);
	
	
	// Now initialize the multilevel objects. Therefore we define a new MatrixFree object for the level operators.
	const unsigned int nlevels = triangulation.n_global_levels();
	mg_matrices.resize(0, nlevels-1);
	mg_solution.resize(0, nlevels-1);
	
	std::set<types::boundary_id> dirichlet_boundary;
	dirichlet_boundary.insert(0);
	mg_constrained_dofs.initialize(dof_handler);
	mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary);
	
	for(unsigned int level=0; level<nlevels; ++level)
	{
	  IndexSet relevant_dofs;
	  DoFTools::extract_locally_relevant_level_dofs(dof_handler, level, relevant_dofs);
	  
	  ConstraintMatrix level_constraints;
	  level_constraints.reinit(relevant_dofs);
	  level_constraints.add_lines(mg_constrained_dofs.get_boundary_indices(level));
	  level_constraints.close();
	  
	  typename MatrixFree<dim,float>::AdditionalData additional_data;
	  additional_data.tasks_parallel_scheme = MatrixFree<dim,float>::AdditionalData::partition_color;
	  additional_data.mapping_update_flags = (update_values | update_gradients | update_JxW_values | update_quadrature_points);
	  additional_data.level_mg_handler = level;
	  std::shared_ptr<MatrixFree<dim,float> > mg_mf_storage_level(new MatrixFree<dim,float>());
	  mg_mf_storage_level->reinit(dof_handler, level_constraints, QGauss<1>(fe.degree+1), additional_data);
	  
	  mg_matrices[level].initialize(mg_mf_storage_level, mg_constrained_dofs, level);
	  mg_matrices[level].initialize_dof_vector(mg_solution[level]);
	}
  }
  
  
  
  template <int dim>
  void
  GelfandProblem<dim>::assemble_right_hand_side()
  {
	TimerOutput::Scope t(computing_timer, "assemble right hand side");
	
	residual_operator.apply(system_rhs, solution);
	
	system_rhs *= -1.0;
  }
  
  
  
  // The constraints object passed to the MatrixFree object takes care about
  // the boundary dofs to set the values on the Dirichlet boundary to zero.
  template <int dim>
  double
  GelfandProblem<dim>::compute_residual(const double alpha)
  {
	TimerOutput::Scope t(computing_timer, "compute residual");
	
	LinearAlgebra::distributed::Vector<double> residual;
	LinearAlgebra::distributed::Vector<double> evaluation_point;
	
	system_matrix.initialize_dof_vector(residual);
	system_matrix.initialize_dof_vector(evaluation_point);
	
	evaluation_point = solution;
	if(alpha > 1e-12)
	  evaluation_point.add(alpha, newton_update);
	
	residual_operator.apply(residual, evaluation_point);
	
	return residual.l2_norm();
  }
  
  
  
  template <int dim>
  void
  GelfandProblem<dim>::compute_update()
  {
	TimerOutput::Scope t(computing_timer, "compute update");
	
	
	solution.update_ghost_values();
	
	system_matrix.evaluate_nonlinearity(solution);
	
	
	MGTransferMatrixFree<dim,float> mg_transfer(mg_constrained_dofs);
	mg_transfer.build(dof_handler);
	
	// New point.
	mg_transfer.interpolate_to_mg(dof_handler, mg_solution, solution);
	
	typedef PreconditionChebyshev<LevelMatrixType,LinearAlgebra::distributed::Vector<float> > SmootherType;
	mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::Vector<float> > mg_smoother;
	MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
	smoother_data.resize(0, triangulation.n_global_levels()-1);
	for(unsigned int level = 0; level<triangulation.n_global_levels(); ++level)
	{
	  if(level > 0)
	  {
		smoother_data[level].smoothing_range = 15.;
		smoother_data[level].degree = 4;
		smoother_data[level].eig_cg_n_iterations = 10;
	  }
	  else
	  {
		smoother_data[0].smoothing_range = 1e-3;
		smoother_data[0].degree = numbers::invalid_unsigned_int;
		smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
	  }
	  
	  mg_matrices[level].evaluate_nonlinearity(mg_solution[level]);
	  mg_matrices[level].compute_diagonal();
	  
	  smoother_data[level].preconditioner = mg_matrices[level].get_matrix_diagonal_inverse();
	}
	mg_smoother.initialize(mg_matrices, smoother_data);
	
	MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<float> > mg_coarse;
	mg_coarse.initialize(mg_smoother);
	
	
	mg::Matrix<LinearAlgebra::distributed::Vector<float> > mg_matrix(mg_matrices);
	
	MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType> > mg_interface_matrices;
	mg_interface_matrices.resize(0, triangulation.n_global_levels()-1);
	for(unsigned int level=0; level<triangulation.n_global_levels(); ++level)
	  mg_interface_matrices[level].initialize(mg_matrices[level]);
	mg::Matrix<LinearAlgebra::distributed::Vector<float> > mg_interface(mg_interface_matrices);
	
	Multigrid<LinearAlgebra::distributed::Vector<float> > mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
	mg.set_edge_matrices(mg_interface, mg_interface);
	
	PreconditionMG<dim, LinearAlgebra::distributed::Vector<float>, MGTransferMatrixFree<dim,float> > preconditioner(dof_handler, mg, mg_transfer);
	
	
	SolverControl solver_control(100, 1.e-12);
	SolverCG<LinearAlgebra::distributed::Vector<double> > cg(solver_control);
	
	constraints.set_zero(solution);
	
	cg.solve(system_matrix, newton_update, system_rhs, preconditioner);
	
	constraints.distribute(newton_update);
	
	linear_iterations = solver_control.last_step();
  }
  
  
  
  template<int dim>
  void
  GelfandProblem<dim>::solve()
  {
	TimerOutput::Scope t(computing_timer, "solve");
	
	const unsigned int	itmax = 10;
	const double		TOLf = 1e-12;
	const double		TOLx = 1e-10;
	
	
	Timer solver_timer;
	solver_timer.start();
	
	for(unsigned int newton_step=1; newton_step<=itmax; ++newton_step)
	{
	  assemble_right_hand_side();
	  
	  compute_update();
	  
	  const double ERRx = newton_update.l2_norm();
	  const double ERRf = compute_residual(1.0);
	  
	  solution.add(1.0, newton_update);
	  
	  pcout << "   Nstep " << newton_step << ", errf = " << ERRf << ", errx = " << ERRx << ", it = " << linear_iterations << std::endl;
	  
	  if(ERRf < TOLf || ERRx < TOLx)
	  {
		solver_timer.stop();
		
		pcout << "Convergence step " << newton_step
		<< " value " << ERRf
		<< " (used wall time: " << solver_timer.wall_time() << " s)" << std::endl;
		
		break;
	  }
	  else if(newton_step==itmax)
	  {
		solver_timer.stop();
		pcout << "WARNING: No convergence of Newton's method after " << newton_step << " steps." << std::endl;
		
		break;
	  }
	}
  }
  
  
  
  template <int dim>
  double
  GelfandProblem<dim>::compute_solution_norm()
  {
	TimerOutput::Scope t(computing_timer, "compute solution norm");
	
	solution.update_ghost_values();
	
	Vector<float> integral_per_cell(triangulation.n_active_cells());
	
	VectorTools::integrate_difference(mapping, dof_handler, solution, Functions::ZeroFunction<dim>(), integral_per_cell, QGauss<dim>(fe.degree+2), VectorTools::H1_seminorm);
	
	return VectorTools::compute_global_error(triangulation, integral_per_cell, VectorTools::H1_seminorm);
  }
  
  
  
  template <int dim>
  void
  GelfandProblem<dim>::output_results() const
  {
	if(dof_handler.n_dofs() > 1e6 || triangulation.n_global_active_cells() > 1e6)
	  return;
	
	solution.update_ghost_values();
	
	DataOut<dim> data_out;
	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(solution, "solution");
	
	Vector<float> subdomain(triangulation.n_active_cells());
	for(unsigned int i=0; i<subdomain.size(); ++i)
	{
	  subdomain(i) = triangulation.locally_owned_subdomain();
	}
	data_out.add_data_vector(subdomain, "subdomain");
	
	data_out.build_patches();
	std::ofstream output("solution." + Utilities::to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD), 2) + ".vtu");
	data_out.write_vtu(output);
	
	if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
	{
	  std::vector<std::string> filenames;
	  for(unsigned int i=0; i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
	  {
		filenames.emplace_back("solution." + Utilities::to_string(i, 2) + ".vtu");
	  }
	  std::ofstream master_output("solution.pvtu");
	  data_out.write_pvtu_record(master_output, filenames);
	}
  }
  
  
  
  template <int dim>
  void
  GelfandProblem<dim>::run() 
  {
	// General output when the program starts
	{
	  const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
	  const unsigned int n_vect_doubles = VectorizedArray<double>::n_array_elements;
	  const unsigned int n_vect_bits = 8*sizeof(double)*n_vect_doubles;
	  
	  std::string DAT_header = "START DATE: " + Utilities::System::get_date() + ", TIME: " + Utilities::System::get_time();
	  std::string MPI_header = "Running with " + std::to_string(n_ranks) + " MPI process" + (n_ranks>1 ? "es" : "");
	  std::string VEC_header = "Vectorization over " + std::to_string(n_vect_doubles) + " doubles = " + std::to_string(n_vect_bits) + " bits (" + Utilities::System::get_current_vectorization_level() + "), VECTORIZATION_LEVEL=" + std::to_string(DEAL_II_COMPILER_VECTORIZATION_LEVEL);
	  std::string SOL_header = "Finite element space: " + fe.get_name();
	  
	  pcout << std::string(80, '=') << std::endl;
	  pcout << DAT_header << std::endl;
	  pcout << std::string(80, '-') << std::endl;
	  
	  pcout << MPI_header << std::endl;
	  pcout << VEC_header << std::endl;
	  pcout << SOL_header << std::endl;
	  
	  pcout << std::string(80, '=') << std::endl;
	}
	
	
	// Generate the triangulation and set the manifold objects to refine correctly
	make_grid();
	
	
	
	// Initialize a timer to measure the computational time for this section and
	// compare it for a different number of MPI ranks.
	Timer timer;
	
	
	// Setup the system, i.e., define the MatrixFree object and initialize the system matrix as well as the residual operator. Initialize the vectors and the multilevel object.
	pcout << "Setup system..." << std::endl;
	setup_system();
	
	
	// Some output about the discretization.
	pcout << "   Triangulation: " << triangulation.n_global_active_cells() << " cells" << std::endl;
	pcout << "   DoFHandler:    " << dof_handler.n_dofs() << " DoFs" << std::endl;
	pcout << std::endl;
	
	
	// Start the Newton iteration to solve the nonlinear problem. 
	pcout << "Start Newton iteration..." << std::endl;
	solve();
	pcout << std::endl;
	
	
	// Stop the time measurement and give some text output.
	timer.stop();
	pcout << "Time for setup+solve (CPU/Wall) " << timer.cpu_time() << "/" << timer.wall_time() << " s" << std::endl;
	pcout << std::endl;
	
	
	// Compute the norm of the solution and write it to the output.
	const double norm = compute_solution_norm();
	pcout << "H1 seminorm of the solution: " << norm << std::endl;
	pcout << std::endl;
	
	
	// Generate the graphical output files.
	pcout << "Output results..." << std::endl;
	pcout << std::endl;
	output_results();
	
	
	// General output when the program ends
	{
	  pcout << std::string(80, '=') << std::endl;
	  pcout << "END DATE: " << Utilities::System::get_date() << ", TIME: " << Utilities::System::get_time() << std::endl;
	  pcout << std::string(80, '=') << std::endl;
	}
  }
}



int
main(int argc, char *argv[]) 
{
  try
  {    
	using namespace stepXX;
	
	Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
	
	GelfandProblem<dimension> gelfand_problem;
	gelfand_problem.run();
  }
  catch(std::exception &exc)
  {
	std::cerr << std::endl << std::endl
	<< "----------------------------------------------------"
	<< std::endl;
	std::cerr << "Exception on processing: " << std::endl
	<< exc.what() << std::endl
	<< "Aborting!" << std::endl
	<< "----------------------------------------------------"
	<< std::endl;
	return 1;
  }
  catch(...)
  {
	std::cerr << std::endl << std::endl
	<< "----------------------------------------------------"
	<< std::endl;
	std::cerr << "Unknown exception!" << std::endl
	<< "Aborting!" << std::endl
	<< "----------------------------------------------------"
	<< std::endl;
	return 1;
  }
  
  return 0;
}
