/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2016 by the deal.II authors
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

 */

// @sect3{Include files}

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

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

#include <deal.II/lac/sparse_direct.h>

#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/grid/grid_out.h>

#include <fstream>
#include <sstream>

// We need to include this class for all of the timings between ILU and Multigrid
#include <deal.II/base/timer.h>

// This includes the files necessary for us to use Multigrid
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

namespace Step55
{
  using namespace dealii;

  // @sect3{Solution Function}

  template <int dim>
  class Solution : public Function<dim>
  {
  public:
    Solution () : Function<dim>(dim+1) {}
        virtual double value (const Point<dim> &p,
                               const unsigned int component) const;
//        virtual Tensor<1,dim> gradient (const Point<dim> &p,
//                                    const unsigned int component = 0) const;
  };

  template <int dim>
    double
  Solution<dim>::value (const Point<dim> &p,
                      const unsigned int component) const
  {
	    using numbers::PI;
	    double x = p(0);
	    double y = p(1);
	    if (component == 0)
	    	return 2 * sin (PI * x);
	    if (component == 1)
	    	return - PI * y * cos(PI * x);
	    if (dim==2)
	    	if (component == 2) // this is different in 2d and 3d because its either u_z or pressure
	    		return sin (PI * x) * sin (PI * y);
	    if (dim ==3)
	    {
		    double z = p(2);
	    	if (component == 2) // this is different in 2d and 3d because its either u_z or pressure
	    		    		return - PI * z * cos(PI * x);
	    	if (component == 3)
	    		    		return sin (PI * x) * sin (PI * y) * sin (PI * z);
	    }
	    else
	    	return 0; // Timo: Or assert?
  }

//  template <int dim>
//  Tensor<1,dim>
//  Solution<dim>::gradient (const Point<dim> &p,  // Timo: But okay for this to be a Tensor?
//                         const unsigned int component) const  // component and you'd return like values
//  {
//	    using numbers::PI;
//	    Tensor<1,dim> return_value;
//	    return_value[0] = 2 * PI * cos (PI * p(0));
//	    return_value[1] = - PI * cos(PI * p(0)) + PI * PI * p(1) * sin(PI * p(0));
//	    if (dim == 3)
//	    	return_value[2] = - PI * cos(PI * p(0)) + PI * PI * p(2) * sin(PI * p(0));
//	    return return_value;
//  }


  // @sect3{ASPECT BlockSchurPreconditioner}

  // Implement the block Schur preconditioner for the Stokes system.
  template <class PreconditionerA, class PreconditionerMp>
  class BlockSchurPreconditioner : public Subscriptor
  {
  public:
    // Constructor of the BlockSchurPreconditioner.
    //
    // S is The entire Stokes matrix.
    //
    // Spre is the matrix whose blocks are used in the definition of
    // the preconditioning of the Stokes matrix, i.e. containing approximations
    // of the A and S blocks.
    //
    // Mppreconditioner is the Preconditioner object for the Schur complement,
    //     typically chosen as the mass matrix.
    //
    // Apreconditioner is the Preconditioner object for the matrix A.
    //
    // do_solve_A is a flag indicating whether we should actually solve with
    //     the matrix $A$, or only apply one preconditioner step with it.
    BlockSchurPreconditioner (const BlockSparseMatrix<double>                 &S,
                              const SparseMatrix<double>                 &P,
                              const PreconditionerMp                     &Mppreconditioner,
                              const PreconditionerA                      &Apreconditioner,
                              const bool                                do_solve_A);


    // Matrix vector product with this preconditioner object.
    void vmult (BlockVector<double>       &dst,
                const BlockVector<double> &src) const;

    unsigned int n_iterations_A() const;
    unsigned int n_iterations_S() const;

  private:

    // References to the various matrix object this preconditioner works on.
    const BlockSparseMatrix<double> &stokes_matrix;
    const SparseMatrix<double> &pressure_mass_matrix;
    const PreconditionerMp                    &mp_preconditioner;
    const PreconditionerA                     &a_preconditioner;


    // Whether to actually invert the $\tilde A$ part of the preconditioner matrix
    // or to just apply a single preconditioner step with it.
    const bool do_solve_A;
    mutable unsigned int n_iterations_A_;
    mutable unsigned int n_iterations_S_;
  };

  template <class PreconditionerA, class PreconditionerMp>
  BlockSchurPreconditioner<PreconditionerA, PreconditionerMp>::
  BlockSchurPreconditioner (const BlockSparseMatrix<double>  &S,
                            const SparseMatrix<double>                &P,
                            const PreconditionerMp                     &Mppreconditioner,
                            const PreconditionerA                      &Apreconditioner,
                            const bool                                  do_solve_A)
    :
    stokes_matrix     (S),
    pressure_mass_matrix (P),
    mp_preconditioner (Mppreconditioner),
    a_preconditioner  (Apreconditioner),
    do_solve_A        (do_solve_A),
    n_iterations_A_(0),
    n_iterations_S_(0)
  {}

  template <class PreconditionerA, class PreconditionerMp>
  unsigned int
  BlockSchurPreconditioner<PreconditionerA, PreconditionerMp>::
  n_iterations_A() const
  {
    return n_iterations_A_;
  }
  template <class PreconditionerA, class PreconditionerMp>
  unsigned int
  BlockSchurPreconditioner<PreconditionerA, PreconditionerMp>::
  n_iterations_S() const
  {
    return n_iterations_S_;
  }


  template <class PreconditionerA, class PreconditionerMp>
  void
  BlockSchurPreconditioner<PreconditionerA, PreconditionerMp>::
  vmult (BlockVector<double>       &dst,
         const BlockVector<double> &src) const
  {
    Vector<double> utmp(src.block(0));

    // First solve with the bottom left block, which we have built
    // as a mass matrix with the inverse of the viscosity
    {
      SolverControl solver_control(1000, 1e-6 * src.block(1).l2_norm());
      SolverCG<>    cg (solver_control);


      // This takes care of the mass matrix
      dst.block(1) = 0.0;
      cg.solve(pressure_mass_matrix,
               dst.block(1), src.block(1),
               mp_preconditioner);
      n_iterations_S_ += solver_control.last_step();


      dst.block(1) *= -1.0;
    }

    // Apply the top right block
    {
      stokes_matrix.block(0,1).vmult(utmp, dst.block(1)); //B^T
      utmp*=-1.0;
      utmp.add(src.block(0));
    }

    // Now either solve with the top left block (if do_solve_A==true)
    // or just apply one preconditioner sweep (for the first few
    // iterations of our two-stage outer GMRES iteration)
    if (do_solve_A == true)
      {
        SolverControl solver_control(10000, utmp.l2_norm()*1e-2);
        SolverCG<>    cg (solver_control);

        dst.block(0) = 0.0;
        cg.solve(stokes_matrix.block(0,0), dst.block(0), utmp,
                 a_preconditioner);
        n_iterations_A_ += solver_control.last_step();
      }
    else
      {
        a_preconditioner.vmult (dst.block(0), utmp);
        n_iterations_A_ += 1;
      }
  }

  // This structure is similar to Step-22 except the 2D and 3D case both use ILU
  template <int dim>
  struct InnerPreconditioner;

  template <>
  struct InnerPreconditioner<2>
  {
    typedef SparseILU<double> type;
  };

  template <>
  struct InnerPreconditioner<3>
  {
    typedef SparseILU<double> type;
  };

  template <int dim>
  class StokesProblem
  {
  public:
    StokesProblem (const unsigned int degree);
    void run ();

  private:
    void setup_dofs (bool use_multigrid);
    void assemble_system (bool use_multigrid);
    void assemble_multigrid ();
    void solve (bool use_multigrid);
    void process_solution ();
    void output_results (const unsigned int refinement_cycle) const;
    void refine_mesh ();

    const unsigned int   degree;

    Triangulation<dim>   triangulation;
    FESystem<dim>        fe;
    FESystem<dim>        velocity_fe;
    DoFHandler<dim>      dof_handler;
    DoFHandler<dim>      velocity_dof_handler;

    // The following are related to constraints associated with the hanging nodes for both the
    // entire system as well as just the A block.  Here we allocate space for an object constraints
    // that will hold a list of these constraints.
    ConstraintMatrix     constraints;
    ConstraintMatrix     velocity_constraints;

    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;

    BlockVector<double> solution;
    BlockVector<double> system_rhs;

    MGLevelObject<SparsityPattern>        mg_sparsity_patterns;
    MGLevelObject<SparseMatrix<double> > mg_matrices;
    MGLevelObject<SparseMatrix<double> > mg_interface_matrices;
    MGConstrainedDoFs                     mg_constrained_dofs;

    TimerOutput computing_timer;

    unsigned int its_A;
    unsigned int its_S;
  };


  template <int dim>
  class BoundaryValuesForVelocity : public Function<dim>
  {
  public:
    BoundaryValuesForVelocity () : Function<dim>(dim) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };


  template <int dim>
  double
  BoundaryValuesForVelocity<dim>::value (const Point<dim>  &p,
                                         const unsigned int component) const
  {
    Assert (component < this->n_components,
            ExcIndexRange (component, 0, this->n_components));

    using numbers::PI;
	    double x = p(0);
	    double y = p(1);
	    if (component == 0)
	    	return 2 * sin (PI * x);
	    if (component == 1)
	    	return - PI * y * cos(PI * x);
	    if (component == 2)
	    	{
	    		double z = p(2);
	    		return - PI * z * cos(PI * x);
	    	}
	    else
	    	return 0;
  }

  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide () : Function<dim>(dim+1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };


  template <int dim>
  double
  RightHandSide<dim>::value (const Point<dim>  & p,   // Timo: This can be done using template specialization
                             const unsigned int component) const
  {
	    using numbers::PI;
	    double x = p(0);
	    double y = p(1);
        if (component == 0)
	    {
	    	if (dim==2)
	    	  return - 2 * PI * PI * sin(PI * x) + PI * cos(PI * x) * sin(PI * y);
	    	else if (dim==3)
	    	{
	    		double z = p(2);
	    		return - 2 * PI * PI * sin(PI * x) + PI * cos(PI * y) * sin(PI * x) * sin(PI * z);
	    	}
	    	else
	    		return 0; // Timo: Assert?
	    }
	    if (component == 1)
	    {
	    	if (dim==2)
	    	  return PI * PI * PI * y * cos(PI * x) + PI * cos(PI * y)*sin(PI * x);
	    	else if (dim==3)
	    	{
	    	  double z = p(2);
	    	  return  PI * PI * PI * y * cos (PI * x) + PI * cos(PI * y)*sin(PI * x)*sin(PI * z);
	    	}
	    	else
	    		return 0; // Timo: Assert?
	    }
	    if (dim==3)
	    {
	    	double z = p(2);
			if (component == 2)
				return PI * PI * PI * z * cos (PI * x) + PI * cos(PI * z)*sin(PI * x)*sin(PI * y);
			if (component == 3) // Ryan: this can go (just here as sanity check)
				return 0;
	    }
	    else
	    	return 0; // must be component = 2 in 2D
  }

  template <int dim>
  StokesProblem<dim>::StokesProblem (const unsigned int degree)
    :
    degree (degree),
    triangulation (Triangulation<dim>::maximum_smoothing),
    fe (FE_Q<dim>(degree+1), dim, // Finite element for whole system
        FE_Q<dim>(degree), 1),
    velocity_fe (FE_Q<dim>(degree+1), dim), // Finite element for velocity-only
    dof_handler (triangulation),
    velocity_dof_handler (triangulation),
    computing_timer (std::cout, TimerOutput::summary,
                     TimerOutput::wall_times)
  {}

// @sect4{StokesProblem::setup_dofs}

// This function sets up things differently based on if you want to use ILU or GMG as a preconditioner.
  template <int dim>
  void StokesProblem<dim>::setup_dofs (bool use_multigrid)
  {
    system_matrix.clear ();
    // We don't need the multigrid dofs for whole problem finite element
    dof_handler.distribute_dofs(fe);

    // This first creates and array (0,0,1) which means that it first does everything with index 0 and then 1
    std::vector<unsigned int> block_component (dim+1,0);
    block_component[dim] = 1;

    // This always knows how to use the dim (start at 0 one)
    FEValuesExtractors::Vector velocities(0);

    if (use_multigrid==false)
      {
        DoFRenumbering::Cuthill_McKee (dof_handler);

        DoFRenumbering::component_wise (dof_handler, block_component);
      }
    else
      {
        // Distribute only the dofs for velocity finite element
        velocity_dof_handler.distribute_dofs(velocity_fe);

        // Multigrid only needs the dofs for velocity
        velocity_dof_handler.distribute_mg_dofs(velocity_fe);

        DoFRenumbering::component_wise (dof_handler, block_component);

        velocity_constraints.clear ();

        // For adaptivity, these must be taken care of (correctly)
        DoFTools::make_hanging_node_constraints (velocity_dof_handler, velocity_constraints);

        VectorTools::interpolate_boundary_values (velocity_dof_handler,
                                                  0,
                                                  BoundaryValuesForVelocity<dim>(),
                                                  velocity_constraints);

        velocity_constraints.close ();

        typename FunctionMap<dim>::type      boundary_condition_function_map;
        BoundaryValuesForVelocity<dim>                velocity_boundary_condition;
        boundary_condition_function_map[1] = &velocity_boundary_condition;

        mg_constrained_dofs.clear();
        mg_constrained_dofs.initialize(velocity_dof_handler, boundary_condition_function_map);
        const unsigned int n_levels = triangulation.n_levels();

        mg_interface_matrices.resize(0, n_levels-1);
        mg_interface_matrices.clear ();
        mg_matrices.resize(0, n_levels-1);
        mg_matrices.clear ();
        mg_sparsity_patterns.resize(0, n_levels-1);

        for (unsigned int level=0; level<n_levels; ++level)
          {
            DynamicSparsityPattern csp;
            csp.reinit(velocity_dof_handler.n_dofs(level), // Generates sparsity pattern, matrices, interface matrices for each matrix on each level
                       velocity_dof_handler.n_dofs(level));
            MGTools::make_sparsity_pattern(velocity_dof_handler, csp, level);

            mg_sparsity_patterns[level].copy_from (csp);

            mg_matrices[level].reinit(mg_sparsity_patterns[level]);
            //std::cout << "mg_matrices[" << level << "] has size " <<  mg_matrices[level].m() << " by " << mg_matrices[level].n() << std::endl;
            mg_interface_matrices[level].reinit(mg_sparsity_patterns[level]);
          }
      }

    {
      constraints.clear ();
      // This is further explained in vector valued dealii step-20
      DoFTools::make_hanging_node_constraints (dof_handler, constraints);
      VectorTools::interpolate_boundary_values (dof_handler,
                                                0,
                                                Solution<dim>(),
                                                constraints,
                                                fe.component_mask(velocities));
    }
    constraints.close ();

    std::vector<types::global_dof_index> dofs_per_block (2);
    DoFTools::count_dofs_per_block (dof_handler, dofs_per_block, block_component);
    const unsigned int n_u = dofs_per_block[0],
                       n_p = dofs_per_block[1];

    std::cout << "   Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "   Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << " (" << n_u << '+' << n_p << ')'
              << std::endl;

    //Assert (n_u == velocity_dof_handler.n_dofs(), ExcMessage ("Numbers of degrees of freedom must match for the velocity part of the main dof handler and the whole velocity dof handler."));

    {
      BlockDynamicSparsityPattern csp (2,2);

      csp.block(0,0).reinit (n_u, n_u);
      csp.block(1,0).reinit (n_p, n_u);
      csp.block(0,1).reinit (n_u, n_p);
      csp.block(1,1).reinit (n_p, n_p);

      csp.collect_sizes();

      DoFTools::make_sparsity_pattern (dof_handler, csp, constraints, false);
      sparsity_pattern.copy_from (csp);

    }
    system_matrix.reinit (sparsity_pattern);

    solution.reinit (2);
    solution.block(0).reinit (n_u);
    solution.block(1).reinit (n_p);
    solution.collect_sizes ();

    system_rhs.reinit (2);
    system_rhs.block(0).reinit (n_u);
    system_rhs.block(1).reinit (n_p);
    system_rhs.collect_sizes ();
  }


// @sect4{StokesProblem::assemble_system}

// In this function, the system matrix is assembled the same regardless of using ILU and GMG
  template <int dim>
  void StokesProblem<dim>::assemble_system (bool use_multigrid)
  {
    system_matrix=0;
    system_rhs=0;

    QGauss<dim>   quadrature_formula(degree+2);

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values    |
                             update_quadrature_points  |
                             update_JxW_values |
                             update_gradients);

    const unsigned int   dofs_per_cell   = fe.dofs_per_cell;

    const unsigned int   n_q_points      = quadrature_formula.size();

    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       local_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    const RightHandSide<dim>          right_hand_side;
    std::vector<Vector<double> >      rhs_values (n_q_points,
                                                  Vector<double>(dim+1));

    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);

    std::vector<SymmetricTensor<2,dim> > symgrad_phi_u (dofs_per_cell);
    std::vector<double>                  div_phi_u   (dofs_per_cell);
    std::vector<double>                  phi_p       (dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        fe_values.reinit (cell);
        local_matrix = 0;
        local_rhs = 0;

        right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                          rhs_values);

        for (unsigned int q=0; q<n_q_points; ++q)
          {
            for (unsigned int k=0; k<dofs_per_cell; ++k)
              {
                symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient (k, q);
                div_phi_u[k]     = fe_values[velocities].divergence (k, q);
                phi_p[k]         = fe_values[pressure].value (k, q);
              }

            for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                for (unsigned int j=0; j<=i; ++j)
                  {
                    local_matrix(i,j) += (2 * (symgrad_phi_u[i] * symgrad_phi_u[j])
                                          - div_phi_u[i] * phi_p[j]
                                          - phi_p[i] * div_phi_u[j]
                                          + phi_p[i] * phi_p[j])
                                         * fe_values.JxW(q);

                  }

                const unsigned int component_i =
                  fe.system_to_component_index(i).first;
                local_rhs(i) += fe_values.shape_value(i,q) *
                                rhs_values[q](component_i) *
                                fe_values.JxW(q);
              }
          }

        for (unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int j=i+1; j<dofs_per_cell; ++j)
            local_matrix(i,j) = local_matrix(j,i);

        cell->get_dof_indices (local_dof_indices);
        constraints.distribute_local_to_global (local_matrix, local_rhs,
                                                local_dof_indices,
                                                system_matrix, system_rhs);
      }
  }

  // @sect4{StokesProblem::assemble_multigrid

  // Here, like step-40, we have a function that assembles everything necessary for the multigrid preconditioner

  template <int dim>
  void StokesProblem<dim>::assemble_multigrid ()
  {
    mg_matrices = 0;

    QGauss<dim>   quadrature_formula(degree+2);

    FEValues<dim> fe_values (velocity_fe, quadrature_formula,
                             update_values    |
                             update_quadrature_points  |
                             update_JxW_values |
                             update_gradients);

    const unsigned int   dofs_per_cell   = velocity_fe.dofs_per_cell;

    const unsigned int   n_q_points      = quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    const RightHandSide<dim>          right_hand_side;
    std::vector<Vector<double> >      rhs_values (n_q_points,
                                                  Vector<double>(dim+1));

    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);

    std::vector<SymmetricTensor<2,dim> > symgrad_phi_u (dofs_per_cell);
    std::vector<double>                  div_phi_u   (dofs_per_cell);
    std::vector<double>                  phi_p       (dofs_per_cell);

    std::vector<std::vector<bool> > interface_dofs
      = mg_constrained_dofs.get_refinement_edge_indices ();
    std::vector<std::vector<bool> > boundary_interface_dofs
      = mg_constrained_dofs.get_refinement_edge_boundary_indices ();

    std::vector<ConstraintMatrix> boundary_constraints (triangulation.n_levels());
    std::vector<ConstraintMatrix> boundary_interface_constraints (triangulation.n_levels());
    for (unsigned int level=0; level<triangulation.n_levels(); ++level)
      {
        boundary_constraints[level].add_lines (interface_dofs[level]);
        boundary_constraints[level].add_lines (mg_constrained_dofs.get_boundary_indices()[level]);
        boundary_constraints[level].close ();

        boundary_interface_constraints[level]
        .add_lines (boundary_interface_dofs[level]);
        boundary_interface_constraints[level].close ();
      }

    // This iterator goes over all cells (not just active)
    typename DoFHandler<dim>::cell_iterator cell = velocity_dof_handler.begin(),
                                            endc = velocity_dof_handler.end();

    for (; cell!=endc; ++cell)
      {
        fe_values.reinit (cell);
        cell_matrix = 0;

        right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                          rhs_values);

        for (unsigned int q=0; q<n_q_points; ++q)
          {
            for (unsigned int k=0; k<dofs_per_cell; ++k)
              {
                symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient (k, q);
              }

            for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                for (unsigned int j=0; j<=i; ++j)
                  {
                    cell_matrix(i,j) += (symgrad_phi_u[i]
                                         * symgrad_phi_u[j])
                                        * fe_values.JxW(q);
                  }
              }
          }


        for (unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int j=i+1; j<dofs_per_cell; ++j)
            cell_matrix(i,j) = cell_matrix(j,i);

        cell->get_mg_dof_indices (local_dof_indices);

        boundary_constraints[cell->level()]
        .distribute_local_to_global (cell_matrix,
                                     local_dof_indices,
                                     mg_matrices[cell->level()]);

        for (unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            if ( !(interface_dofs[cell->level()][local_dof_indices[i]]==true &&
                   interface_dofs[cell->level()][local_dof_indices[j]]==false))
              cell_matrix(i,j) = 0;

        boundary_interface_constraints[cell->level()]
        .distribute_local_to_global (cell_matrix,
                                     local_dof_indices,
                                     mg_interface_matrices[cell->level()]);
      }
  }

// @sect4{StokesProblem::solve}

// This function sets up things differently based on if you want to use ILU or GMG as a preconditioner.  Both methods share
// the same solver (GMRES) but require a different preconditioner to be assembled.  Here we time not only the entire solve
// function, but we separately time the set-up of the preconditioner as well as the GMRES solve.
  template <int dim>
  void StokesProblem<dim>::solve (bool use_multigrid)
  {
    SolverControl solver_control (system_matrix.m(),
                                  1e-6*system_rhs.l2_norm());

    GrowingVectorMemory<BlockVector<double> > vector_memory;
    SolverGMRES<BlockVector<double> >::AdditionalData gmres_data;
    gmres_data.max_n_tmp_vectors = 100;

    SolverGMRES<BlockVector<double> > gmres(solver_control, vector_memory,
                                            gmres_data);

    SparseMatrix<double> pressure_mass_matrix;  // Timo: This block has trouble going to assembly for some reason.. even if I make pressure_mass_matrix global
    pressure_mass_matrix.reinit(sparsity_pattern.block(1,1));
    pressure_mass_matrix.copy_from(system_matrix.block(1,1));
    system_matrix.block(1,1) = 0;

    if (use_multigrid==false)
      {
        computing_timer.enter_subsection ("Solve - Set-up Preconditioner");

        std::cout << "   Computing preconditioner..." << std::endl << std::flush;

        SparseILU<double>       A_preconditioner;
        A_preconditioner.initialize (system_matrix.block(0,0),SparseILU<double>::AdditionalData());

        SparseILU<double> pmass_preconditioner;
        pmass_preconditioner.initialize (pressure_mass_matrix,
                                         SparseILU<double>::AdditionalData());

        bool use_expensive = true;

        // If this cheaper solver is not desired, then simply short-cut
        // the attempt at solving with the cheaper preconditioner that consists
        // of only a single V-cycle
        const BlockSchurPreconditioner<typename InnerPreconditioner<dim>::type,
              SparseILU<double>>
              preconditioner (system_matrix,
                              pressure_mass_matrix,
                              pmass_preconditioner, A_preconditioner,
                              use_expensive);
        computing_timer.leave_subsection();
        computing_timer.enter_subsection ("Solve - GMRES");
        gmres.solve (system_matrix,
                     solution,
                     system_rhs,
                     preconditioner);
        computing_timer.leave_subsection();

        its_A = preconditioner.n_iterations_A();
        its_S = preconditioner.n_iterations_S();

        constraints.distribute (solution);

        std::cout << " "
                  << solver_control.last_step()
                  << " block GMRES iterations";
      }
    else
      {
        computing_timer.enter_subsection ("Solve - Set-up Preconditioner");
        // Transfer operators between levels
        MGTransferPrebuilt<Vector<double> > mg_transfer(constraints, mg_constrained_dofs);
        mg_transfer.build_matrices(velocity_dof_handler);

        // Coarse grid solver
        FullMatrix<double> coarse_matrix;
        coarse_matrix.copy_from (mg_matrices[0]); //TODO: coarse_grid_solver.initialize(mg_matrices[0])  --- failed
        MGCoarseGridHouseholder<> coarse_grid_solver;
        coarse_grid_solver.initialize (coarse_matrix);
        //coarse_grid_solver.initialize(mg_matrices[0]);

        typedef PreconditionSOR<SparseMatrix<double> > Smoother;
        mg::SmootherRelaxation<Smoother, Vector<double> > mg_smoother;
        mg_smoother.initialize(mg_matrices);
        mg_smoother.set_steps(2);

        // Multigrid, when used as a preconditioner for CG, expects the smoother to be symmetric, and this takes care of that
        mg_smoother.set_symmetric(true);

        mg::Matrix<Vector<double> > mg_matrix(mg_matrices);
        mg::Matrix<Vector<double> > mg_interface_up(mg_interface_matrices);
        mg::Matrix<Vector<double> > mg_interface_down(mg_interface_matrices);

        // Now, we are ready to set up the V-cycle operator and the multilevel preconditioner.
        Multigrid<Vector<double> > mg(velocity_dof_handler,
                                      mg_matrix,
                                      coarse_grid_solver,
                                      mg_transfer,
                                      mg_smoother,
                                      mg_smoother);
        mg.set_edge_matrices(mg_interface_down, mg_interface_up);

        // The velocity_dof_handler takes care of fact that we want block(0,0) here, as that is A.
        PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double> > >
        A_Multigrid(velocity_dof_handler, mg, mg_transfer);

        SparseILU<double> pmass_preconditioner;
        pmass_preconditioner.initialize (pressure_mass_matrix,
                                         SparseILU<double>::AdditionalData());

        bool use_expensive = false;

        // If this cheaper solver is not desired, then simply short-cut
        // the attempt at solving with the cheaper preconditioner that
        // consists of only a single V-cycle
        const BlockSchurPreconditioner<PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double> > >,
              SparseILU<double>>
              preconditioner (system_matrix, pressure_mass_matrix,
                              pmass_preconditioner, A_Multigrid,
                              use_expensive);

        computing_timer.leave_subsection();
        computing_timer.enter_subsection ("Solve - GMRES");
        gmres.solve (system_matrix,
                     solution,
                     system_rhs,
                     preconditioner);
        computing_timer.leave_subsection();

        its_A += preconditioner.n_iterations_A();
        its_S += preconditioner.n_iterations_S();

        constraints.distribute (solution);

        std::cout << " "
                  << solver_control.last_step()
                  << " block GMRES iterations";
      }
  }

  // @sect4{StokesProblem::process_solution}

  template <int dim>
  void StokesProblem<dim>::process_solution ()
  {
    FEValuesExtractors::Vector velocities(0);

   const ComponentSelectFunction<dim> // Timo: Step-20 showed me this but errors seem to be the same
   pressure_mask (dim, dim+1);
   const ComponentSelectFunction<dim>
   velocity_mask(std::make_pair(0, dim), dim+1);

    Vector<float> difference_per_cell (triangulation.n_active_cells());

    VectorTools::integrate_difference (dof_handler,
                                       solution,
                                       Solution<dim>(),
                                       difference_per_cell,
                                       QGauss<dim>(3),
                                       VectorTools::L2_norm),
                                       &velocity_mask;

   const double Velocity_L2_error = difference_per_cell.l2_norm();

   VectorTools::integrate_difference (dof_handler,
                                      solution,
                                      Solution<dim>(),
                                      difference_per_cell,
                                      QGauss<dim>(3),
                                      VectorTools::L2_norm),
                                      &pressure_mask;

  const double Pressure_L2_error = difference_per_cell.l2_norm();

//    VectorTools::integrate_difference (dof_handler,
//                                       solution,
//                                       Solution<dim>(),
//                                       difference_per_cell,
//                                       QGauss<dim>(3),
//                                       VectorTools::H1_norm,
//                                       fe.component_mask(velocities)); // TODO: need to add mask because only want to check velocity or pressure

//    const double H1_error = difference_per_cell.l2_norm();

    const QTrapez<1> q_trapez; // Timo: Shall we use something better than QTrapez?

    std::cout << std::endl << "Velocity L2 Error: " << Velocity_L2_error
        << std::endl
        << "Pressure L2 Error: " << Pressure_L2_error
                << std::endl
//        << "H1 Error: "
//        << H1_error
        << std::endl;

    const double Linfty_error = difference_per_cell.linfty_norm();
    const unsigned int n_active_cells=triangulation.n_active_cells();
    const unsigned int n_dofs=dof_handler.n_dofs();

    std::cout //<< "Cycle " << cycle << ':'
        << std::endl
        << " Number of active cells: "
        << n_active_cells
        << std::endl
        << " Number of degrees of freedom: "
        << n_dofs
        << std::endl;
  }


  // @sect4{StokesProblem::output_results}

  template <int dim>
  void
  StokesProblem<dim>::output_results (const unsigned int refinement_cycle)  const
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
    data_out.add_data_vector (solution, solution_names,
                              DataOut<dim>::type_dof_data,
                              data_component_interpretation);
    data_out.build_patches ();

    std::ostringstream filename;
    filename << "solution-"
             << Utilities::int_to_string (refinement_cycle, 2)
             << ".vtk";

    std::ofstream output (filename.str().c_str());
    data_out.write_vtk (output);

    std::cout << std::endl
              << "Number of iterations used for approximation of A inverse: " << its_A << std::endl
              << "Number of iterations used for approximation of S inverse: " << its_S << std::endl
              << std::endl;
  }


  // @sect4{StokesProblem::refine_mesh}

  template <int dim>
  void
  StokesProblem<dim>::refine_mesh ()
  {
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

    FEValuesExtractors::Scalar pressure(dim);
    KellyErrorEstimator<dim>::estimate (dof_handler,
                                        QGauss<dim-1>(degree+1),
                                        typename FunctionMap<dim>::type(),
                                        solution,
                                        estimated_error_per_cell,
                                        fe.component_mask(pressure));

    GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                     estimated_error_per_cell,
                                                     0.3, 0.0);
    triangulation.execute_coarsening_and_refinement ();
  }

  // @sect4{StokesProblem::run}

  // The last step in the Stokes class is, as usual, the function that
  // generates the initial grid and calls the other functions in the
  // respective order.
  template <int dim>
  void StokesProblem<dim>::run ()
  {
    GridGenerator::hyper_cube (triangulation);

    triangulation.refine_global (6-dim);

    for (unsigned int refinement_cycle = 0; refinement_cycle<3;
         ++refinement_cycle)
      {
        bool use_multigrid=false;
        do
          {
            std::cout << "Refinement cycle " << refinement_cycle << std::endl;

            if (refinement_cycle > 0)
              triangulation.refine_global (1);
            if (use_multigrid == false)
              std::cout << "Now running with ILU" << std::endl;
            if (use_multigrid == true)
              std::cout << "Now running with Multigrid" << std::endl;
            computing_timer.enter_subsection ("Setup");
            setup_dofs(use_multigrid);
            computing_timer.leave_subsection();
            std::cout << "   Assembling..." << std::endl << std::flush;
            computing_timer.enter_subsection ("Assemble");
            assemble_system (use_multigrid);
            computing_timer.leave_subsection();
            if (use_multigrid == true)
              {
                std::cout << "   Assembling Multigrid..." << std::endl << std::flush;
                computing_timer.enter_subsection ("Assemble Multigrid");
                assemble_multigrid();
                computing_timer.leave_subsection();
              }
            std::cout << "   Solving..." << std::flush;
            computing_timer.enter_subsection ("Solve");
            solve (use_multigrid);
            computing_timer.leave_subsection();

            process_solution ();

            computing_timer.print_summary ();
            computing_timer.reset ();
            output_results (refinement_cycle);

            use_multigrid = !use_multigrid;
          }
        while (use_multigrid);
      }
  }
}

// @sect3{The main function}

int main ()
{
  try
    {
      using namespace dealii;
      using namespace Step55;

      deallog.depth_console(1); // Timo: Need this or else there is too much output

      StokesProblem<2> flow_problem(1);

      flow_problem.run ();
    }
  catch (std::exception &exc)
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
  catch (...)
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
