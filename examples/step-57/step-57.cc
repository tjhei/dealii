/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2008 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Liang Zhao, Clemson University, 2016
 */

// @sect3{Include files}

// As usual, we start by including some well-known files:
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

// To transfer solutions between meshes, this file is included.
#include <deal.II/numerics/solution_transfer.h>

// This file includes UMFPACK: the direct solver.
#include <deal.II/lac/sparse_direct.h>

// And the one for ILU preconditioner
#include <deal.II/lac/sparse_ilu.h>

// C++
#include <fstream>
#include <iostream>
#include <sstream>

// As in all programs, the namespace dealii is included
namespace Step57
{
  using namespace dealii;

  // @sect3{The <code>StokesProblem</code> class template}

  // As explained in introduction, what we obtain at each step is the Newton's update term instead of the real solution, so
  // we define two variables: the present solution and the update term. By Newton's iteration, the new solution can be written
  // as $x_{new} = x_{old} + x_{update}$
  // In this program, we do not know the exact solution, thus we check its convergence through
  // the norm of residual. A sparse matrix for the mass matrix of pressure is created for the operator of
  // a block Schur complement preconditioner. We use one ConstraintMatrix for implementing Dirichlet boundary conditions at
  // the initial step and a zero ConstraintMatrix for the Newton's update term.

  template <int dim>
  class Navier_Stokes_Newton
  {
  public:
    Navier_Stokes_Newton(const unsigned int degree);
    ~Navier_Stokes_Newton();
    void run();

  private:
    void setup_system();
    void assemble_NavierStokes_system(const bool initial_step);
    void solve(bool initial_step);
    void refine_mesh();
    void process_solution();
    void output_results (const unsigned int refinement_cycle) const;
    void set_viscosity(double nu);
    void search_initial_guess(double step_size);
    double compute_residual(const double alpha);


    double viscosity;
    double gamma;
    const unsigned int           degree;

    Triangulation<dim>           triangulation;
    FESystem<dim>                fe;
    DoFHandler<dim>              dof_handler;

    ConstraintMatrix             zero_constraints;
    ConstraintMatrix             nonzero_constraints;

    BlockSparsityPattern         sparsity_pattern;
    BlockSparseMatrix<double>    system_matrix;
    SparseMatrix<double>         pressure_mass_matrix;

    BlockVector<double>          present_solution;
    BlockVector<double>          newton_update;
    BlockVector<double>          system_rhs;
    BlockVector<double>          residual;

  };

  // @sect3{Boundary values and right hand side}
  // In this problem we set the velocity along the upper surface of the cavity to be one on and ones on the other three
  // boundaries to be zero,
  // and the right hand side function is a ZeroFunction. The dimension of the boundary function is dim+1 that implies the
  // pressure is included. In practice, the boundary values are applied to our solution through ConstraintMatrix which is
  // obtained by using
  // VectorTools::interpolate_boundary_values. The components of boundary value functions should be according with the dimension
  // of finite element space. Therefore we have to define the boundary values of pressure even though we actually do not need
  // it. While creating the ConstraintMatrix, we use a ComponentMask to discard the pressure component in boundary values.

  // The following function represents the boundary values:

  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues() : Function<dim>(dim+1) {}
    virtual double value(const Point<dim> &p,
                         const unsigned int component) const;

    virtual void   vector_value(const Point <dim>    &p,
                                Vector<double> &values) const;
  };

  template <int dim>
  double BoundaryValues<dim>::value(const Point<dim> &p,
                                   const unsigned int component) const
  {
	Assert (component < this->n_components,
			ExcIndexRange (component, 0, this->n_components));
    if (component == 0)
      {
        if (p[1]>1.0-1e-5)
          return 1.0;
        else
          return 0.0;
      }

    return 0;
  }

  template <int dim>
  void BoundaryValues<dim>::vector_value (const Point<dim> &p,
                                         Vector<double>   &values) const
  {
    for (unsigned int c = 0; c < this->n_components; ++c)
      values(c) = BoundaryValues<dim>::value (p, c);
  }


  // @sect3{BlockSchurPreconditioner for Navier Stokes equations}
  // In this part we define the block Schur complement preconditioner. AS discussed in introduction,
  // the preconditioner in Krylov iterative methods is implemented as a matrix-vector product operator.
  // In practice, the Schur complement preconditioner is decomposed as a product of three matrices(as presented
  // in the first section). The tilde-A inverse in the first factor involves a solver for the linear system
  // $\tilde{A}x=b$. Here we solve this system via a direct solver for simplicity. The computation involved
  // in the second factor is simple matrix-vector multiplication. The Schur complement $\tilde{S}$
  // can be well approximated by the mass matrix of pressure and its inverse can be obtained through an inexact
  // solver. Because of the symmetry of pressure mass matrix we use an iterative method, the CG method, to
  // solve the corresponding linear system.
  //
  // In summary, the preconditioner is defined by a combination of operators: a direct solver for the first
  // factor, matrix-vector multiplication for the second factor and a iterative method for the last factor.
  // $\gamma$ is the coefficient of the Augmented Lagrangian term and $\nu$ is the viscosity.


  template <class PreconditionerMp>
  class BlockSchurPreconditioner : public Subscriptor
  {
  public:
	  BlockSchurPreconditioner (double                                     gamma,
			  	  	  	  	  	double                                     viscosity,
			  	  	  	  	  	const BlockSparseMatrix<double>            &S,
	                            const SparseMatrix<double>                 &P,
	                            const PreconditionerMp                     &Mppreconditioner
							    );

	     void vmult (BlockVector<double>       &dst,
	                 const BlockVector<double> &src) const;

  private:
	     const double gamma;
	     const double viscosity;
	     const BlockSparseMatrix<double> &stokes_matrix;
	     const SparseMatrix<double>      &pressure_mass_matrix;
	     const PreconditionerMp          &mp_preconditioner;
  };

  template <class PreconditionerMp>
  BlockSchurPreconditioner<PreconditionerMp>::
  BlockSchurPreconditioner (double                           gamma,
		  	  	  	  	  	double                           viscosity,
		  	  	  	  	  	const BlockSparseMatrix<double>  &S,
                            const SparseMatrix<double>       &P,
                            const PreconditionerMp           &Mppreconditioner)
    :
	gamma                (gamma),
	viscosity            (viscosity),
    stokes_matrix        (S),
    pressure_mass_matrix (P),
    mp_preconditioner    (Mppreconditioner)
  {}

  template <class PreconditionerMp>
   void
   BlockSchurPreconditioner<PreconditionerMp>::
   vmult (BlockVector<double>       &dst,
          const BlockVector<double> &src) const
   {
     Vector<double> utmp(src.block(0));

     {
       SolverControl solver_control(1000, 1e-6 * src.block(1).l2_norm());
       SolverCG<>    cg (solver_control);

       dst.block(1) = 0.0;
       cg.solve(pressure_mass_matrix,
                dst.block(1), src.block(1),
                mp_preconditioner);
       dst.block(1) *= -1.0/(viscosity+gamma);
     }

     {
       stokes_matrix.block(0,1).vmult(utmp, dst.block(1));
       utmp*=-1.0;
       utmp+=src.block(0);
     }

     SparseDirectUMFPACK  A_direct;
     A_direct.initialize(stokes_matrix.block(0,0));
     A_direct.vmult (dst.block(0), utmp);
   }

  // @sect3{Navier_Stokes_Newton class implementation}
  // @sect4{Navier_Stokes_Newton::Navier_Stokes_Newton}
  // The constructor of this class looks very similar to the one in step-22. The only difference is the
  // viscosity and the AL coefficient. In test case, we set the viscosity to be a small number that implies
  // the nonlinear term is dominant so that we have to figure out a way to linearize the system before solving.

  template <int dim>
  Navier_Stokes_Newton<dim>::Navier_Stokes_Newton(const unsigned int degree)
    :

    viscosity(1.0/10000.0),
	gamma(1.0),
    degree(degree),
    triangulation(Triangulation<dim>::maximum_smoothing),
    fe(FE_Q<dim>(degree+1), dim,
       FE_Q<dim>(degree),   1),
    dof_handler(triangulation)
  {}


  template <int dim>
  Navier_Stokes_Newton<dim>::~Navier_Stokes_Newton()
  {
    dof_handler.clear();
  }

  // @sect4{Navier_Stokes_Newton::setup_system}
  // All structures are set up in this part.

  template <int dim>
  void Navier_Stokes_Newton<dim>::setup_system()
  {
    system_matrix.clear();
    pressure_mass_matrix.clear();

    // The first step is to associate DoFs with a given mesh. Here it is done as in step-22
    dof_handler.distribute_dofs (fe);
    DoFRenumbering::Cuthill_McKee (dof_handler);

    // In Navier Stokes velocity and pressure are both what we want to solve so a block structure of size dim+1 is created:
    // dim for velocity and 1 for pressure.
    std::vector<unsigned int> block_component(dim+1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise (dof_handler, block_component);

    // In Newton's scheme, we first apply the boundary condition on the solution obtained from the initial step.
    // To make sure boundary condition satisfied, zero boundary condition is used for the Newton's update term.
    // Therefore we set up two constraints for the two situations: nonzero constraints for the initial step and
    // zero constrains for the update term.
    FEValuesExtractors::Vector velocities(0);
    {
      nonzero_constraints.clear();

      DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               BoundaryValues<dim>(),
                                               nonzero_constraints,
                                               fe.component_mask(velocities));
    }
    nonzero_constraints.close();

    {
      zero_constraints.clear();

      DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints);
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               ZeroFunction<dim>(dim+1),
                                               zero_constraints,
                                               fe.component_mask(velocities));
    }
     zero_constraints.close();

  // Finally, block matrices and block vectors are set up. In Newton's scheme, the solution is computed through
  // x_new = x_old + update_term. Correspondingly two block vectors are created: we use present_solution to store
  // the solution from last step and compute the newton_update to obtain a new solution. Then
  // present_solution is replaced by the new one. The residual is used in linear search and we will discuss it in details later.

    std::vector<types::global_dof_index> dofs_per_block (2);
    DoFTools::count_dofs_per_block (dof_handler, dofs_per_block, block_component);
    const unsigned int n_u = dofs_per_block[0],
                       n_p = dofs_per_block[1];

    {
      BlockDynamicSparsityPattern dsp (2,2);
      dsp.block(0,0).reinit (n_u, n_u);
      dsp.block(1,0).reinit (n_p, n_u);
      dsp.block(0,1).reinit (n_u, n_p);
      dsp.block(1,1).reinit (n_p, n_p);
      dsp.collect_sizes();

      DoFTools::make_sparsity_pattern (dof_handler, dsp, nonzero_constraints);
      sparsity_pattern.copy_from (dsp);
    }

    system_matrix.reinit (sparsity_pattern);


    present_solution.reinit (2);
    present_solution.block(0).reinit (n_u);
    present_solution.block(1).reinit (n_p);
    present_solution.collect_sizes ();

    newton_update.reinit (2);
    newton_update.block(0).reinit (n_u);
    newton_update.block(1).reinit (n_p);
    newton_update.collect_sizes ();

    system_rhs.reinit (2);
    system_rhs.block(0).reinit (n_u);
    system_rhs.block(1).reinit (n_p);
    system_rhs.collect_sizes ();

    residual.reinit (2);
    residual.block(0).reinit (n_u);
    residual.block(1).reinit (n_p);
    residual.collect_sizes ();

  }


  // @sect4{Navier_Stokes_Newton::assemble_NavierStokes_system}
  // This function builds the system matrix and right hand side that we actually work on. We can see the function contains
  // one argument: initial_step. This is because, as we discussed above, the constraints are not the same when the local data
  // is transfered to the global. If initial_step is true, nonzero constraint works, or we use zero constraint.
  // Similar to step-22, extractors are set up to handle vector component and pressure component.
  template <int dim>
  void Navier_Stokes_Newton<dim>::assemble_NavierStokes_system(const bool initial_step)
  {
    system_matrix = 0;
    system_rhs    = 0;

    QGauss<dim>   quadrature_formula(degree+2);

    FEValues<dim> fe_values (fe,
                             quadrature_formula,
                             update_values |
                             update_quadrature_points |
                             update_JxW_values |
                             update_gradients );

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);

    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       local_rhs    (dofs_per_cell);

    std::vector<Vector<double>>   rhs_values(n_q_points, Vector<double>(dim+1));

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    // For the linearized system, we create temporary storages for present velocity and its divergence and gradient, and
    // present pressure. In practice, they are all obtained through their sharp functions at quadrature points. This work
    // can be done by FEValues::get_function_values, FEValues::get_function_gradients and FEValues::get_function_divergences.

    std::vector<Tensor<1, dim>>   present_velocity_values    (n_q_points);
    std::vector<Tensor<2, dim>>   present_velocity_gradients (n_q_points);
    std::vector<double>           present_velocity_divergence(n_q_points);
    std::vector<double>           present_pressure_values    (n_q_points);

    std::vector<double>           div_phi_u                 (dofs_per_cell);
    std::vector<Tensor<1, dim>>   phi_u                     (dofs_per_cell);
    std::vector<Tensor<2, dim>>   grad_phi_u                (dofs_per_cell);
    std::vector<double>           phi_p                     (dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

    for (; cell!=endc; ++cell)
      {
        fe_values.reinit(cell);
        local_matrix = 0;
        local_rhs    = 0;

        fe_values[velocities].get_function_values(present_solution,
                                                  present_velocity_values);

        fe_values[velocities].get_function_gradients(present_solution,
                                                     present_velocity_gradients);

        fe_values[velocities].get_function_divergences(present_solution,
                                                       present_velocity_divergence);

        fe_values[pressure].get_function_values(present_solution,
                                                present_pressure_values);

        // Then we do the same as in step-22 to assemble the matrix with nonlinear term linearized by present solutions.
        // An additional term with gamma as coefficient is the Lagrangian Argument(AL). As we discussed in introduction, the
        // 1,1-block of system matrix should be zero. Since the mass matrix of pressure is used while creating
        // the preconditioner, we assemble it here and set it to be zero when solve linear system.

        for (unsigned int q=0; q<n_q_points; ++q)
          {
            for (unsigned int k=0; k<dofs_per_cell; ++k)
              {
                div_phi_u[k]  =  fe_values[velocities].divergence (k, q);
                grad_phi_u[k] =  fe_values[velocities].gradient(k, q);
                phi_u[k]      =  fe_values[velocities].value(k, q);
                phi_p[k]      =  fe_values[pressure]  .value(k, q);
              }

            for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                  {
                    local_matrix(i, j) += (  viscosity*scalar_product(grad_phi_u[j], grad_phi_u[i])      //(gradU, gradV_u)
                                             + present_velocity_gradients[q]*phi_u[j]*phi_u[i]   //(d_U*grad_U_old, V_u)
                                             + grad_phi_u[j]*present_velocity_values[q]*phi_u[i] //(U_old*grad_d_U, V_u)
                                             - div_phi_u[i]*phi_p[j]                             //-(d_P, div_V_u)
                                             - phi_p[i]*div_phi_u[j]                             //-(div_d_U, V_p)
											 + gamma*div_phi_u[j]*div_phi_u[i]                   //grad-div (div_u_j, div_u_i)
                                             + phi_p[i]*phi_p[j] )                               //Mp-mass matrix of pressure
                                          * fe_values.JxW(q);

                  }

                //  f-(-laplace_U + U*grad_U + grad_p)
                //   -div_U
                const unsigned int component_i = fe.system_to_component_index(i).first;
                local_rhs(i) += (
                                  fe_values.shape_value(i, q)*rhs_values[q](component_i)     // (f, V)
                                  -viscosity*scalar_product(present_velocity_gradients[q],grad_phi_u[i]) // -(gradU_old, gradV_u)
                                  -present_velocity_gradients[q]*present_velocity_values[q]*phi_u[i]
                                  // -(U_old*gradU_old, V_u)
                                  +present_pressure_values[q]*div_phi_u[i] // +(P_old, div_V_u)
                                  +present_velocity_divergence[q]*phi_p[i]   // +(div_U_oid, V_p)
								  -gamma*present_velocity_divergence[q]*div_phi_u[i]
                                )*fe_values.JxW(q);

              }
          }

        cell-> get_dof_indices (local_dof_indices);

        if (initial_step)
          {

            nonzero_constraints.distribute_local_to_global(local_matrix,
                                                           local_rhs,
                                                           local_dof_indices,
                                                           system_matrix,
                                                           system_rhs);
          }
        else
          {
            zero_constraints.distribute_local_to_global(local_matrix,
                                                        local_rhs,
                                                        local_dof_indices,
                                                        system_matrix,
                                                        system_rhs);
          }
      }

      pressure_mass_matrix.reinit(sparsity_pattern.block(1,1));
      pressure_mass_matrix.copy_from(system_matrix.block(1,1));
      system_matrix.block(1,1) = 0;

  }

  // @sect4{Navier_Stokes_Newton::solve}
  // In this function, we use FGMRES together with the block preconditioner, which is defined at the beginning of the program,
  // to solve the linear system. What we obtain at this step is the coefficient vector. If this is the initial step, the
  // coefficient vector gives us an initial guess of Navier Stokes equations. For the initial step, nonzero constrain is
  // applied in order to make sure boundary condition satisfied. In the following steps, we will actually solve for the Newton's
  // update term so zero constraint is distributed to the coefficient vector.
  //

  template <int dim>
  void Navier_Stokes_Newton<dim>::solve (bool initial_step)
  {
     SolverControl solver_control (system_matrix.m(),1e-4*system_rhs.l2_norm(), true);

     SolverFGMRES<BlockVector<double> >::AdditionalData gmres_data;

     SolverFGMRES<BlockVector<double> > gmres(solver_control,gmres_data);

     SparseILU<double> pmass_preconditioner;
     pmass_preconditioner.initialize (pressure_mass_matrix,
                                      SparseILU<double>::AdditionalData());

     const BlockSchurPreconditioner<SparseILU<double>>
         preconditioner (gamma,
        		         viscosity,
        		 	 	 system_matrix,
                         pressure_mass_matrix,
                         pmass_preconditioner);

     gmres.solve (system_matrix,
                  newton_update,
                  system_rhs,
                  preconditioner);

     std::cout << " ****FGMRES steps: " << solver_control.last_step() << std::endl;

     if(initial_step)
     {
    	 nonzero_constraints.distribute(newton_update);
     }

     else
     {
    	 zero_constraints.distribute(newton_update);
     }

  }

  // @sect4{Navier_Stokes_Newton::compute_residual}
  // This function deals with the line search. Recalling Newton's update term is obtained under low tolerance and this
  // implies the update direction is probably not good enough. As discussed in introduction, we use line search to guarantee
  // the solution we obtained at each step is at least better than last one. In other words, the residual is no larger than
  // previous one. In this function, alpha is the weight of Newton's update term.

  template <int dim>
  double Navier_Stokes_Newton<dim>::compute_residual(const double alpha)
  {
    residual = 0;

    BlockVector<double> evaluation_point;
    evaluation_point = present_solution;
    evaluation_point.add(alpha, newton_update);

    QGauss<dim>   quadrature_formula(degree+2);

    FEValues<dim> fe_values (fe,
                             quadrature_formula,
                             update_values |
                             update_quadrature_points |
                             update_JxW_values |
                             update_gradients );

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);

    Vector<double>       local_res    (dofs_per_cell);

    std::vector<Vector<double>>   rhs_values(n_q_points, Vector<double>(dim+1));

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    std::vector<Tensor<1, dim>>   present_velocity_values    (n_q_points);
    std::vector<Tensor<2, dim>>   present_velocity_gradients (n_q_points);
    std::vector<double>           present_velocity_divergence(n_q_points);
    std::vector<double>           present_pressure_values    (n_q_points);

    std::vector<double>           div_phi_u                 (dofs_per_cell);
    std::vector<Tensor<1, dim>>   phi_u                     (dofs_per_cell);
    std::vector<Tensor<2, dim>>   grad_phi_u                (dofs_per_cell);
    std::vector<double>           phi_p                     (dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

    for (; cell!=endc; ++cell)
      {
        fe_values.reinit(cell);
        local_res    = 0;

        fe_values[velocities].get_function_values(evaluation_point,
                                                  present_velocity_values);

        fe_values[velocities].get_function_gradients(evaluation_point,
                                                     present_velocity_gradients);

        fe_values[velocities].get_function_divergences(evaluation_point,
                                                       present_velocity_divergence);

        fe_values[pressure].get_function_values(evaluation_point,
                                                present_pressure_values);


        for (unsigned int q=0; q<n_q_points; ++q)
          {
            for (unsigned int k=0; k<dofs_per_cell; ++k)
              {
                div_phi_u[k]  =  fe_values[velocities].divergence (k, q);
                grad_phi_u[k] =  fe_values[velocities].gradient(k, q);
                phi_u[k]      =  fe_values[velocities].value(k, q);
                phi_p[k]      =  fe_values[pressure]  .value(k, q);
              }

            for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                //  f-(-laplace_U + U*grad_U + grad_p)
                //   -div_U
                const unsigned int component_i = fe.system_to_component_index(i).first;
                local_res(i) += (
                                  fe_values.shape_value(i, q)*rhs_values[q](component_i)     // (f, V)
                                  -viscosity*scalar_product(present_velocity_gradients[q],grad_phi_u[i]) // -(gradU_old, gradV_u)
                                  -present_velocity_gradients[q]*present_velocity_values[q]*phi_u[i]
                                  // -(U_old*gradU_old, V_u)
                                  +present_pressure_values[q]*div_phi_u[i] // +(P_old, div_V_u)
                                  +present_velocity_divergence[q]*phi_p[i]   // +(div_U_oid, V_p)
								  -gamma*present_velocity_divergence[q]*div_phi_u[i]
                                )*fe_values.JxW(q);

              }
          }

        cell-> get_dof_indices (local_dof_indices);

        nonzero_constraints.distribute_local_to_global(local_res,
                                                       local_dof_indices,
                                                       residual);
      }
    nonzero_constraints.set_zero(residual); // why?
    return residual.l2_norm();
  }

  // @sect4{Navier_Stokes_Newton::set_viscosity}
  // This function sets the viscosity to be nu.
  template <int dim>
  void Navier_Stokes_Newton<dim>::set_viscosity(double nu)
  {
	  viscosity = nu;
  }

  // @sect4{Navier_Stokes_Newton::refine_mesh}
  // After finding out a good initial guess on coarse mesh, we hope to decrease the error through refining the mesh.
  // Here we do the adaptive refinement. The first part is almost the same as in step-15: we tag the cells that need
  // to be refined and do the refinement.
  template <int dim>
  void Navier_Stokes_Newton<dim>::refine_mesh()
  {

		Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
		FEValuesExtractors::Vector velocity(0);
		KellyErrorEstimator<dim>::estimate (dof_handler,
		                                    QGauss<dim-1>(degree+1),
		                                    typename FunctionMap<dim>::type(),
		                                    present_solution,
		                                    estimated_error_per_cell,
		                                    fe.component_mask(velocity));

		GridRefinement::refine_and_coarsen_fixed_number (triangulation,
		                                                 estimated_error_per_cell,
		                                                 0.3, 0.0);


		  triangulation.prepare_coarsening_and_refinement();
		  SolutionTransfer<dim, BlockVector<double>> solution_transfer(dof_handler);
		  solution_transfer.prepare_for_coarsening_and_refinement(present_solution);
		  triangulation.execute_coarsening_and_refinement ();

          //  Create a temporary vector "tmp", whose size is according with the solution in refined mesh,
          //  to receive the solution transfered from last mesh.

		  dof_handler.distribute_dofs (fe);
		  DoFRenumbering::Cuthill_McKee (dof_handler);
		  std::vector<unsigned int> block_component(dim+1, 0);
		  block_component[dim] = 1;
		  DoFRenumbering::component_wise (dof_handler, block_component);
		  std::vector<types::global_dof_index> dofs_per_block (2);
		  DoFTools::count_dofs_per_block (dof_handler, dofs_per_block, block_component);
		  const unsigned int n_u = dofs_per_block[0],
				             n_p = dofs_per_block[1];

		  BlockVector<double> tmp;
		  tmp.reinit (2);
		  tmp.block(0).reinit (n_u);
		  tmp.block(1).reinit (n_p);
		  tmp.collect_sizes ();

          //  Transfer solution from coarse to fine mesh and apply boundary value constraints
          //  to the new transfered solution. Then set it to be the initial guess on the
          //  fine mesh.
		  solution_transfer.interpolate(present_solution, tmp);
		  setup_system();
		  nonzero_constraints.distribute(tmp);
		  present_solution = tmp;
  }

  // @sect4{Navier_Stokes_Newton::search_initial_guess}
  // As we discussed in introduction, the solution to Stokes equations will not be a good approximation to Navier Stokes
  // equations so we have to use the solution to another Navier Stokes whose viscosity is a little larger than the original
  // one as an initial guess. In practice we set up a series of auxiliary Navier Stokes equations working like a staircase:
  // from Stokes to the original Navier Stokes. By experiment, the solution to Stokes is good enough to be the initial guess
  // of Navier Stokes with viscosity 1000 so we let the first stair be 1000. To make sure the solution from previous Navier
  // Stokes is cloeser enough to next Navier Stokes, the step size must be small enough or we will lose convergence.
  template <int dim>
  void Navier_Stokes_Newton<dim>::search_initial_guess(double step_size)
  {
	  const double target_Re = 1.0/viscosity;

	  double stair_Re  = 1000.0;
	  double diffe_Re  = target_Re-stair_Re;
	  bool first_step  = true;

	  while (diffe_Re >= step_size)
	  {
		  set_viscosity(1/stair_Re);
		  std::cout << "*****************************************" << std::endl;
		  std::cout << " Searching for initial guess with Re = " << stair_Re << std::endl;
		  std::cout << "*****************************************" << std::endl;

		  double current_res = 1.0;
		  double last_res = 1.0;
		  unsigned int outer_iteration = 0;

	      while ((first_step || (current_res > 1e-12)) && outer_iteration < 50)
	     {
	    	std::cout << " ** Viscosity = " << viscosity << std::endl;
	    	++outer_iteration;

	    	if (first_step)
	    	{
	    	  setup_system();
	    	  assemble_NavierStokes_system(first_step);
	    	  solve(first_step);
	    	  present_solution.add(1, newton_update);
	    	  nonzero_constraints.distribute(present_solution);
	    	  current_res = compute_residual(0);
	    	  first_step = false;
	    	  std::cout << "******************************" << std::endl;
	    	  std::cout << " The residual of initial guess is " << current_res << std::endl;
	          std::cout << " Initialization complete!  " << std::endl;
	        }

	       else
	       {
	    	  assemble_NavierStokes_system(first_step);
	          last_res = system_rhs.l2_norm();
	          solve(first_step);
	          double alpha = 1.0;
	          for (alpha = 1.0; alpha > 1e-5; alpha *= 0.5)
	          {
	            current_res = compute_residual(alpha);
	            std::cout << " alpha = " << std::setw(6) << alpha << std::setw(0)
	                      << " res = " << current_res << std::endl;
	            if (current_res < last_res)
	              break;
	           }

              {
	            last_res = current_res;
	            present_solution.add(alpha, newton_update);
	            nonzero_constraints.distribute(present_solution);
	            std::cout << " ----The " << outer_iteration << "th iteration. ---- " << std::endl;
	            std::cout << " ----Residual: " << current_res << std::endl;
	          }
	        }
	      }
	      stair_Re += step_size;
	      diffe_Re -= step_size;
	  }

  }

  // @sect4{Navier_Stokes_Newton::output_results}
  // This function is all the same as in step-22.

  template <int dim>
  void Navier_Stokes_Newton<dim>::output_results (const unsigned int refinement_cycle)  const
  {
    std::vector<std::string> solution_names (dim, "velocity");

    solution_names.push_back ("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation
    (dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation
    .push_back (DataComponentInterpretation::component_is_scalar);
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (present_solution, solution_names,
                              DataOut<dim>::type_dof_data,
                              data_component_interpretation);
    data_out.build_patches ();

    std::ostringstream filename;
    filename << "solution-"
             << Utilities::int_to_string (refinement_cycle, 2)
             << ".vtk";

    std::ofstream output (filename.str().c_str());
    data_out.write_vtk (output);
  }

  // @sect4{Navier_Stokes_Newton::process_solution}
  // In our test case, we do not know the analytic solution to Navier Stokes equations, so another numerical result is
  // considered to be the "real" solution. This function outputs horizontal velocity alone x=0.5 and y from 0 to 1.
  template <int dim>
  void Navier_Stokes_Newton<dim>::process_solution()
  {
	  std::ofstream f("line.txt");
      f << "# y u_x u_y" << std::endl;

      Point<dim> p;
      p(0)= 0.5;
      p(1)= 0.5;

      f << std::scientific;

      for (unsigned int i=0; i<=100; ++i)
        {

          p(dim-1) = i/100.0;

          Vector<double> tmp_vector(dim+1);
          VectorTools::point_value(dof_handler, present_solution, p, tmp_vector);
          f << p(dim-1);

          for (int j=0; j<dim; j++)
            f << " " << tmp_vector(j);
            f << std::endl;
        }
  }


  // @sect4{Navier_Stokes_Newton::run}
  // This is the last step of this program. In this part, we generate the grid and run the other functions respectively.
  // We first generate a square of size $8 \times 8$ which is relatively coarse mesh, and the initial guess will be
  // determined on this mesh via the "staircase". If the viscosity is small, such as 1/7500 or 1/10000, the initial guess
  // is expected to be close to the real solution. In that case, the mesh for initial guess should also be finer and the step
  // size(in Navier_Stokes_Newton::search_initial_guess) should be small.

  template <int dim>
  void Navier_Stokes_Newton<dim>::run()
  {

    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(5);

    const double Reynold =  1.0/viscosity;

    // When the viscosity is larger than 1/1000, the solution to Stokes equations is good enough as an initial guess. If so,
    // we do not need to search for the initial guess via staircase. Newton's iteration can be started directly.
    if (Reynold <= 1000)
    {
    double current_res = 1;
    double last_res = 1;
    bool   first_step = true;
    unsigned int max_outer_iter = 50;

    // The starting mesh is $8 \times 8$, then we do 5 steps of adaptive refinement. After each refinement, the solution from
    // previous mesh will be transfered to the new one. On each mesh, Newton's iteration runs at most 15 times or the residual
    // is smaller than tolerance.

    for (unsigned int refinement = 0; refinement < 1; ++refinement)
    {
    	unsigned int outer_iteration = 0;
    	std::cout << "*****************************************" << std::endl;
    	std::cout << "************  refinement = " << refinement << " ************ " << std::endl;
    	std::cout << "viscosity= " << viscosity << std::endl;
    	std::cout << "*****************************************" << std::endl;

    	while ((first_step || (current_res > 1e-12)) && outer_iteration < max_outer_iter)
    	{
    		++outer_iteration;
    		if (first_step)
    		{
    		  setup_system();
    		  assemble_NavierStokes_system(first_step);
    		  solve(first_step);
    		  present_solution.add(1, newton_update);
    		  nonzero_constraints.distribute(present_solution);
    	      process_solution();
    		  current_res = compute_residual(0);
    		  first_step = false;
    		  std::cout << "******************************" << std::endl;
    		  std::cout << " The residual of initial guess is " << current_res << std::endl;
    	      std::cout << " Initialization complete!  " << std::endl;
       	    }

    	  else
          {
    		assemble_NavierStokes_system(first_step);
            last_res = system_rhs.l2_norm();
            solve(first_step);
            double alpha = 1.0;
            // Because of the rough computation of update term, the Newton's iteration, $x_{new} = x_{old} + x_{update}$,
            // probably fails to show us an improved solution at present step. Hence the weight of update term is moderated
            // to let the present solution be better. Starting with 1, we take its half if the current residual is not
            // better than the previous one until an appropriate weight is found. In this case, the Newton's iteration
            // acts as $x_{new} = x_{old} + \alpha x_{update}$, where $\alpha$ is less than 1.

            for (alpha = 1.0; alpha > 1e-5; alpha *= 0.5)
              {
                current_res = compute_residual(alpha);
                std::cout << " alpha = " << std::setw(6) << alpha << std::setw(0)
                          << " res = " << current_res << std::endl;
                if (current_res < last_res)
                  break;
              }

            {
              last_res = current_res;
              present_solution.add(alpha, newton_update);
              nonzero_constraints.distribute(present_solution);
              process_solution();
              std::cout << " ----The " << outer_iteration << "th iteration. ---- " << std::endl;
              std::cout << " ----Residual: " << current_res << std::endl;
            }

          }
    	output_results (50*refinement+outer_iteration);
      }
    	refine_mesh();
    	current_res =1;
     }
    }

    // If the viscosity is smaller than 1/1000, we have to first search for an initial guess via "staircase". What we
    // should notice is the search is always on the initial mesh, that is the $8 \times 8$ mesh in this program.
    // After the searching part, we just do the same as we did when viscosity is larger than 1/1000: run Newton's iteration,
    // refine the mesh, transfer solutions, and again.
    else
    {
    	std::cout << "       Searching for initial guess ... " << std::endl;
    	search_initial_guess(2000.0);

    	std::cout << "       Computing solution with target viscosity ..." <<std::endl;
    	std::cout << "       Reynold = " << Reynold << std::endl;
    	set_viscosity(1.0/Reynold);

   	    double current_res = 1;
   	    double last_res = 1;
   	    unsigned int max_outer_iter = 50;

   	    for (unsigned int refinement = 0; refinement < 5; ++refinement)
   	    {
   	    	unsigned int outer_iteration = 0;
   	    	std::cout << "*****************************************" << std::endl;
   	    	std::cout << "************  refinement = " << refinement << " ************ " << std::endl;
   	    	std::cout << "*****************************************" << std::endl;

//   	    	if(refinement == 4)
//   	    	{
//   	    		max_outer_iter = 10;
//   	    	}

   	    	while (current_res > 1e-12 && outer_iteration < max_outer_iter)
   	    	{
    	    	++outer_iteration;
        		assemble_NavierStokes_system(false);
                last_res = system_rhs.l2_norm();
                solve(false);
    	        double alpha = 1.0;
                for (alpha = 1.0; alpha > 1e-5; alpha *= 0.5)
    	        {
   	                current_res = compute_residual(alpha);
   	                std::cout << " alpha = " << std::setw(6) << alpha << std::setw(0)
   	                          << " res = " << current_res << std::endl;
   	                if (current_res < last_res)
   	                  break;
   	            }

   	            {
   	              last_res = current_res;
   	              present_solution.add(alpha, newton_update);
   	              nonzero_constraints.distribute(present_solution);
   	              process_solution();
   	              std::cout << " ----The " << outer_iteration << "th iteration. ---- " << std::endl;
   	              std::cout << " ----Residual: " << current_res << std::endl;
   	            }

   	           output_results (50*refinement+outer_iteration);
   	      }
   	      refine_mesh();
   	      current_res =1;
    }
   }

  }

}

int main()
{
  using namespace dealii;
  using namespace Step57;

  deallog.depth_console(0);

  Navier_Stokes_Newton<2> flow(1);
  flow.run();
}









