/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2009 - 2016 by the deal.II authors
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
 * Author: Timo Heister, Clemson University
 */

/*
 * work in progress p-laplacian with PETSc SNES and maybe Trilinos NOX
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>

// uncomment the following #define if you have PETSc and Trilinos installed
// and you prefer using Trilinos in this example:
// #define FORCE_USE_OF_TRILINOS

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
}


#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <fstream>
#include <iostream>


const double eps = 1e-2;

namespace Step40
{
  using namespace dealii;

  template <int dim>
  class LaplaceProblem
  {
  public:
    LaplaceProblem ();
    ~LaplaceProblem ();

    void run ();
    void residual_function (LA::MPI::Vector &x,
                            LA::MPI::Vector &f);
      void update_jacobian_function (LA::MPI::SparseMatrix &mat, LA::MPI::Vector &x);

  private:
    void setup_system ();
    void assemble_residual (LA::MPI::Vector &residual,
                            const LA::MPI::Vector &solution);
    void assemble_system (LA::MPI::SparseMatrix &mat);
    void solve ();
    void refine_grid ();
    void output_results (const unsigned int cycle) const;

    MPI_Comm                                  mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    DoFHandler<dim>                           dof_handler;
    FE_Q<dim>                                 fe;

    IndexSet                                  locally_owned_dofs;
    IndexSet                                  locally_relevant_dofs;

    ConstraintMatrix                          constraints;

    LA::MPI::SparseMatrix                     system_matrix;
    LA::MPI::Vector                           locally_relevant_solution;
    LA::MPI::Vector                           x;
    LA::MPI::Vector                           residual;
    LA::MPI::Vector                           system_rhs;

    ConditionalOStream                        pcout;
    TimerOutput                               computing_timer;
  };

  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide () : Function<dim>(1) {}

    virtual double value (const Point<dim> &p,
                          const unsigned int  component = 0) const;

  };


  template <int dim>
  double
  RightHandSide<dim>::value (const Point<dim> &p,
                             const unsigned int /*component*/) const
  {
    const double x = p[0];
    const double y = p[1];
    //return 1*x*(1-x)*y*(1-y);
    return -(1.0*sin(x)*pow(sin(y), 2.0)*cos(x) - 1.0*sin(x)*cos(x)*pow(cos(y), 2.0))*pow(pow(pow(sin(x), 2.0)*pow(sin(y), 2.0) + pow(cos(x), 2.0)*pow(cos(y), 2.0), 1.0) + 0.0001, -0.5)*cos(x)*cos(y) + (1.0*pow(sin(x), 2.0)*sin(y)*cos(y) - 1.0*sin(y)*pow(cos(x), 2.0)*cos(y))*pow(pow(pow(sin(x), 2.0)*pow(sin(y), 2.0) + pow(cos(x), 2.0)*pow(cos(y), 2.0), 1.0) + 0.0001, -0.5)*sin(x)*sin(y) + 2*sqrt(pow(pow(sin(x), 2.0)*pow(sin(y), 2.0) + pow(cos(x), 2.0)*pow(cos(y), 2.0), 1.0) + 0.0001)*sin(x)*cos(y);
  }

  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution () : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };


  template <int dim>
  double
  ExactSolution<dim>::value (const Point<dim> &p,
                             const unsigned int  /*component*/) const
  {
    const double x = p[0];
    const double y = p[1];
    return sin(x)*cos(y);
  }

  // @sect3{The <code>LaplaceProblem</code> class implementation}

  // @sect4{Constructors and destructors}

  // Constructors and destructors are rather trivial. In addition to what we
  // do in step-6, we set the set of processors we want to work on to all
  // machines available (MPI_COMM_WORLD); ask the triangulation to ensure that
  // the mesh remains smooth and free to refined islands, for example; and
  // initialize the <code>pcout</code> variable to only allow processor zero
  // to output anything. The final piece is to initialize a timer that we
  // use to determine how much compute time the different parts of the program
  // take:
  template <int dim>
  LaplaceProblem<dim>::LaplaceProblem ()
    :
    mpi_communicator (MPI_COMM_WORLD),
    triangulation (mpi_communicator,
                   typename Triangulation<dim>::MeshSmoothing
                   (Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening)),
    dof_handler (triangulation),
    fe (1),
    pcout (std::cout,
           (Utilities::MPI::this_mpi_process(mpi_communicator)
            == 0)),
    computing_timer (mpi_communicator,
                     pcout,
                     TimerOutput::summary,
                     TimerOutput::wall_times)
  {}



  template <int dim>
  LaplaceProblem<dim>::~LaplaceProblem ()
  {
    dof_handler.clear ();
  }


  // @sect4{LaplaceProblem::setup_system}

  // The following function is, arguably, the most interesting one in the
  // entire program since it goes to the heart of what distinguishes %parallel
  // step-40 from sequential step-6.
  //
  // At the top we do what we always do: tell the DoFHandler object to
  // distribute degrees of freedom. Since the triangulation we use here is
  // distributed, the DoFHandler object is smart enough to recognize that on
  // each processor it can only distribute degrees of freedom on cells it
  // owns; this is followed by an exchange step in which processors tell each
  // other about degrees of freedom on ghost cell. The result is a DoFHandler
  // that knows about the degrees of freedom on locally owned cells and ghost
  // cells (i.e. cells adjacent to locally owned cells) but nothing about
  // cells that are further away, consistent with the basic philosophy of
  // distributed computing that no processor can know everything.
  template <int dim>
  void LaplaceProblem<dim>::setup_system ()
  {
    TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs (fe);

    // The next two lines extract some information we will need later on,
    // namely two index sets that provide information about which degrees of
    // freedom are owned by the current processor (this information will be
    // used to initialize solution and right hand side vectors, and the system
    // matrix, indicating which elements to store on the current processor and
    // which to expect to be stored somewhere else); and an index set that
    // indicates which degrees of freedom are locally relevant (i.e. live on
    // cells that the current processor owns or on the layer of ghost cells
    // around the locally owned cells; we need all of these degrees of
    // freedom, for example, to estimate the error on the local cells).
    locally_owned_dofs = dof_handler.locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs (dof_handler,
                                             locally_relevant_dofs);

    // Next, let us initialize the solution and right hand side vectors. As
    // mentioned above, the solution vector we seek does not only store
    // elements we own, but also ghost entries; on the other hand, the right
    // hand side vector only needs to have the entries the current processor
    // owns since all we will ever do is write into it, never read from it on
    // locally owned cells (of course the linear solvers will read from it,
    // but they do not care about the geometric location of degrees of
    // freedom).
    locally_relevant_solution.reinit (locally_owned_dofs,
                                      locally_relevant_dofs, mpi_communicator);

    system_rhs.reinit (locally_owned_dofs, mpi_communicator);
    residual.reinit (locally_owned_dofs, mpi_communicator);
    x.reinit (locally_owned_dofs, mpi_communicator);

    // The next step is to compute hanging node and boundary value
    // constraints, which we combine into a single object storing all
    // constraints.
    //
    // As with all other things in %parallel, the mantra must be that no
    // processor can store all information about the entire universe. As a
    // consequence, we need to tell the constraints object for which degrees
    // of freedom it can store constraints and for which it may not expect any
    // information to store. In our case, as explained in the @ref distributed
    // module, the degrees of freedom we need to care about on each processor
    // are the locally relevant ones, so we pass this to the
    // ConstraintMatrix::reinit function. As a side note, if you forget to
    // pass this argument, the ConstraintMatrix class will allocate an array
    // with length equal to the largest DoF index it has seen so far. For
    // processors with high MPI process number, this may be very large --
    // maybe on the order of billions. The program would then allocate more
    // memory than for likely all other operations combined for this single
    // array.
    constraints.clear ();
    constraints.reinit (locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints (dof_handler, constraints);
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              ExactSolution<dim>(),
                                              constraints);
    constraints.close ();

    // The last part of this function deals with initializing the matrix with
    // accompanying sparsity pattern. As in previous tutorial programs, we use
    // the DynamicSparsityPattern as an intermediate with which we
    // then initialize the PETSc matrix. To do so we have to tell the sparsity
    // pattern its size but as above there is no way the resulting object will
    // be able to store even a single pointer for each global degree of
    // freedom; the best we can hope for is that it stores information about
    // each locally relevant degree of freedom, i.e. all those that we may
    // ever touch in the process of assembling the matrix (the @ref
    // distributed_paper "distributed computing paper" has a long discussion
    // why one really needs the locally relevant, and not the small set of
    // locally active degrees of freedom in this context).
    //
    // So we tell the sparsity pattern its size and what DoFs to store
    // anything for and then ask DoFTools::make_sparsity_pattern to fill it
    // (this function ignores all cells that are not locally owned, mimicking
    // what we will do below in the assembly process). After this, we call a
    // function that exchanges entries in these sparsity pattern between
    // processors so that in the end each processor really knows about all the
    // entries that will exist in that part of the finite element matrix that
    // it will own. The final step is to initialize the matrix with the
    // sparsity pattern.
    DynamicSparsityPattern dsp (locally_relevant_dofs);

    DoFTools::make_sparsity_pattern (dof_handler, dsp,
                                     constraints, false);
    SparsityTools::distribute_sparsity_pattern (dsp,
                                                dof_handler.n_locally_owned_dofs_per_processor(),
                                                mpi_communicator,
                                                locally_relevant_dofs);

    system_matrix.reinit (locally_owned_dofs,
                          locally_owned_dofs,
                          dsp,
                          mpi_communicator);
  }

  template <int dim>
  void LaplaceProblem<dim>::assemble_residual (LA::MPI::Vector &residual, const LinearAlgebraPETSc::MPI::Vector &solution)
  {
    TimerOutput::Scope t(computing_timer, "residual");
    residual = 0.;

    const QGauss<dim>  quadrature_formula(3);

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values    |  update_gradients |
                             update_quadrature_points |
                             update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    std::vector<Tensor<1,dim> > velocity_gradients(n_q_points);

    RightHandSide<dim> right_hand_side;
    std::vector<double> rhs_values(n_q_points);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          cell_rhs = 0;

          fe_values.reinit (cell);
          fe_values.get_function_gradients(solution, velocity_gradients);
          right_hand_side.value_list(fe_values.get_quadrature_points(),
                                     rhs_values);


          for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            {

              // = (eps+nablau^2^ 1/2
              double nu = pow(eps + velocity_gradients[q_point].norm_square(), 0.5);

              for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                  cell_rhs(i) += (rhs_values[q_point]
                                  * fe_values.shape_value(i,q_point)
                                  - nu *
                                  velocity_gradients[q_point] *
                                  fe_values.shape_grad(i,q_point)
                                 )
                                 * fe_values.JxW(q_point);
                }
            }

          cell->get_dof_indices (local_dof_indices);
          constraints.distribute_local_to_global (cell_rhs,
                                                  local_dof_indices,
                                                  residual);
        }
    residual.compress (VectorOperation::add);
  }



  template <int dim>
  void LaplaceProblem<dim>::assemble_system (LA::MPI::SparseMatrix &mat)
  {
    mat = 0;
    system_rhs = 0;
    TimerOutput::Scope t(computing_timer, "assembly");

    const QGauss<dim>  quadrature_formula(3);

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values    |  update_gradients |
                             update_quadrature_points |
                             update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    std::vector<Tensor<1,dim> > velocity_gradients(n_q_points);

    RightHandSide<dim> right_hand_side;
    std::vector<double> rhs_values(n_q_points);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          cell_matrix = 0;
          cell_rhs = 0;

          fe_values.reinit (cell);
          fe_values.get_function_gradients(locally_relevant_solution, velocity_gradients);
          right_hand_side.value_list(fe_values.get_quadrature_points(),
                                     rhs_values);

          for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            {
              double nu = pow(eps + velocity_gradients[q_point].norm_square(), 0.5);


              for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                  for (unsigned int j=0; j<dofs_per_cell; ++j)
                    cell_matrix(i,j) += (nu * fe_values.shape_grad(i,q_point) *
                                         fe_values.shape_grad(j,q_point) *
                                         fe_values.JxW(q_point));

                  cell_rhs(i) += (rhs_values[q_point] *
                                  fe_values.shape_value(i,q_point) *
                                  fe_values.JxW(q_point));
                }
            }

          cell->get_dof_indices (local_dof_indices);
          constraints.distribute_local_to_global (cell_matrix,
                                                  cell_rhs,
                                                  local_dof_indices,
                                                  mat,
                                                  system_rhs);
        }

    // Notice that the assembling above is just a local operation. So, to
    // form the "global" linear system, a synchronization between all
    // processors is needed. This could be done by invoking the function
    // compress(). See @ref GlossCompress  "Compressing distributed objects"
    // for more information on what is compress() designed to do.
    mat.compress (VectorOperation::add);
    system_rhs.compress (VectorOperation::add);
  }


  template<int dim>
  void LaplaceProblem<dim>::residual_function (LA::MPI::Vector &x,
                                               LA::MPI::Vector &f)
  {

    locally_relevant_solution = x;
    assemble_residual(f, locally_relevant_solution);
    pcout << "residual: " << f.l2_norm() << std::endl;
  }

  template<int dim>
  void LaplaceProblem<dim>::update_jacobian_function (LA::MPI::SparseMatrix &mat, LA::MPI::Vector &x)
  {
    pcout << "Jacobian... ||x||=" << x.l2_norm() << std::endl;
    locally_relevant_solution = x;
    assemble_system (mat);
  }

  template<int dim>
  PetscErrorCode ResidualFunc(SNES snes, Vec x, Vec f, void *ctx)
  {
    // compute f = residual(x)
    const char *name;
    PetscObjectGetName((PetscObject)x, &name);
    std::cout << " x=" << name;
    PetscObjectGetName((PetscObject)f, &name);
    std::cout << " f=" << name;
    std::cout << std::endl;

    LaplaceProblem<dim> *ptr = static_cast<LaplaceProblem<dim>*>(ctx);
    LA::MPI::Vector v_x(x);
    LA::MPI::Vector v_f(f);
    ptr->residual_function(v_x, v_f);

    return 0;
  }

  template<int dim>
  PetscErrorCode JacobianFunc(SNES snes, Vec x, Mat jac, Mat B, void *ctx)
  {
    std::cout << "JacobianFunc" << std::endl;
    const char *name;
    PetscObjectGetName((PetscObject)x, &name);
    std::cout << " x=" << name << std::endl;
    PetscObjectGetName((PetscObject)jac, &name);
    std::cout << " jac=" << name << std::endl;
    PetscObjectGetName((PetscObject)B, &name);
    std::cout << " B=" << name << std::endl;

    LA::MPI::Vector v_x(x);
    LaplaceProblem<dim> *ptr = static_cast<LaplaceProblem<dim>*>(ctx);
    LA::MPI::SparseMatrix mat((Mat)B);
    ptr->update_jacobian_function(mat, v_x);

    return 0;
  }

  template <int dim>
  void LaplaceProblem<dim>::solve ()
  {
    TimerOutput::Scope t(computing_timer, "solve");

    SNES snes;
    SNESCreate(MPI_COMM_WORLD, &snes);

    const char *name;
    PetscObjectGetName((PetscObject)(Vec)x, &name);
    std::cout << " x=" << name;
    PetscObjectGetName((PetscObject)(Vec)system_rhs, &name);
    std::cout << " system_rhs=" << name;
    PetscObjectGetName((PetscObject)(Vec)residual, &name);
    std::cout << " residual=" << name;
    PetscObjectGetName((PetscObject)(Vec)locally_relevant_solution, &name);
    std::cout << " locally_relevant_solution=" << name;
    std::cout << std::endl;


    SNESSetFunction(snes, residual, ResidualFunc<dim>, this);
    SNESSetJacobian(snes, system_matrix, system_matrix, JacobianFunc<dim>, this);

    KSP ksp;
    PC pc;
    SNESGetKSP(snes,&ksp);
    KSPGetPC(ksp,&pc);
    PCSetType(pc,PCNONE);
    KSPSetTolerances(ksp,1.e-8,PETSC_DEFAULT,PETSC_DEFAULT,20);


    //SNESGetKSP;
    //  SNESSetType
//        SNESSetUp

    assemble_system(system_matrix);
    

    //x=1;

    //constraints.distribute(x);


    SNESSetFromOptions(snes);

    SNESSolve(snes, (Vec)system_rhs, (Vec)x);
    int its;
    SNESGetIterationNumber(snes,&its);

    pcout << its << " Newton iterations!" << std::endl;

    SNESDestroy(&snes);

    /*
        SolverControl solver_control (dof_handler.n_dofs(), 1e-12);

    #ifdef USE_PETSC_LA
        LA::SolverCG solver(solver_control, mpi_communicator);
    #else
        LA::SolverCG solver(solver_control);
    #endif

        LA::MPI::PreconditionAMG preconditioner;

        LA::MPI::PreconditionAMG::AdditionalData data;

    #ifdef USE_PETSC_LA
        data.symmetric_operator = true;
    #else

    #endif
        preconditioner.initialize(system_matrix, data);

        solver.solve (system_matrix, completely_distributed_solution, system_rhs,
                      preconditioner);

        pcout << "   Solved in " << solver_control.last_step()
              << " iterations." << std::endl;
    */

//  VectorTools::interpolate(dof_handler,
//                           ExactSolution<dim>(),
//                           x);
    constraints.distribute (x);

    locally_relevant_solution = x;
  }



  // @sect4{LaplaceProblem::refine_grid}

  // The function that estimates the error and refines the grid is again
  // almost exactly like the one in step-6. The only difference is that the
  // function that flags cells to be refined is now in namespace
  // parallel::distributed::GridRefinement -- a namespace that has functions
  // that can communicate between all involved processors and determine global
  // thresholds to use in deciding which cells to refine and which to coarsen.
  //
  // Note that we didn't have to do anything special about the
  // KellyErrorEstimator class: we just give it a vector with as many elements
  // as the local triangulation has cells (locally owned cells, ghost cells,
  // and artificial ones), but it only fills those entries that correspond to
  // cells that are locally owned.
  template <int dim>
  void LaplaceProblem<dim>::refine_grid ()
  {
    TimerOutput::Scope t(computing_timer, "refine");

    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate (dof_handler,
                                        QGauss<dim-1>(3),
                                        typename FunctionMap<dim>::type(),
                                        locally_relevant_solution,
                                        estimated_error_per_cell);
    parallel::distributed::GridRefinement::
    refine_and_coarsen_fixed_number (triangulation,
                                     estimated_error_per_cell,
                                     0.3, 0.03);
    triangulation.execute_coarsening_and_refinement ();
  }



  // @sect4{LaplaceProblem::output_results}

  // Compared to the corresponding function in step-6, the one here is a tad
  // more complicated. There are two reasons: the first one is that we do not
  // just want to output the solution but also for each cell which processor
  // owns it (i.e. which "subdomain" it is in). Secondly, as discussed at
  // length in step-17 and step-18, generating graphical data can be a
  // bottleneck in parallelizing. In step-18, we have moved this step out of
  // the actual computation but shifted it into a separate program that later
  // combined the output from various processors into a single file. But this
  // doesn't scale: if the number of processors is large, this may mean that
  // the step of combining data on a single processor later becomes the
  // longest running part of the program, or it may produce a file that's so
  // large that it can't be visualized any more. We here follow a more
  // sensible approach, namely creating individual files for each MPI process
  // and leaving it to the visualization program to make sense of that.
  //
  // To start, the top of the function looks like always. In addition to
  // attaching the solution vector (the one that has entries for all locally
  // relevant, not only the locally owned, elements), we attach a data vector
  // that stores, for each cell, the subdomain the cell belongs to. This is
  // slightly tricky, because of course not every processor knows about every
  // cell. The vector we attach therefore has an entry for every cell that the
  // current processor has in its mesh (locally owned ones, ghost cells, and
  // artificial cells), but the DataOut class will ignore all entries that
  // correspond to cells that are not owned by the current processor. As a
  // consequence, it doesn't actually matter what values we write into these
  // vector entries: we simply fill the entire vector with the number of the
  // current MPI process (i.e. the subdomain_id of the current process); this
  // correctly sets the values we care for, i.e. the entries that correspond
  // to locally owned cells, while providing the wrong value for all other
  // elements -- but these are then ignored anyway.
  template <int dim>
  void LaplaceProblem<dim>::output_results (const unsigned int cycle) const
  {
    {
      Vector<double> cellwise_errors (triangulation.n_active_cells());
      QGauss<dim> quadrature (3);


      VectorTools::integrate_difference (dof_handler,
                                         locally_relevant_solution,
                                         ExactSolution<dim>(),
                                         cellwise_errors,
                                         quadrature,
                                         VectorTools::L2_norm);

      const double error_u_l2
        = VectorTools::compute_global_error(triangulation, cellwise_errors, VectorTools::L2_norm);
      pcout << "error= " << error_u_l2 << std::endl;
    }



    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (locally_relevant_solution, "u");

    Vector<float> subdomain (triangulation.n_active_cells());
    for (unsigned int i=0; i<subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector (subdomain, "subdomain");



    data_out.build_patches ();

    // The next step is to write this data to disk. We choose file names of
    // the form <code>solution-XX.PPPP.vtu</code> where <code>XX</code>
    // indicates the refinement cycle, <code>PPPP</code> refers to the
    // processor number (enough for up to 10,000 processors, though we hope
    // that nobody ever tries to generate this much data -- you would likely
    // overflow all file system quotas), and <code>.vtu</code> indicates the
    // XML-based Visualization Toolkit (VTK) file format.
    const std::string filename = ("solution-" +
                                  Utilities::int_to_string (cycle, 2) +
                                  "." +
                                  Utilities::int_to_string
                                  (triangulation.locally_owned_subdomain(), 4));
    std::ofstream output ((filename + ".vtu").c_str());
    data_out.write_vtu (output);

    // The last step is to write a "master record" that lists for the
    // visualization program the names of the various files that combined
    // represents the graphical data for the entire domain. The
    // DataOutBase::write_pvtu_record does this, and it needs a list of
    // filenames that we create first. Note that only one processor needs to
    // generate this file; we arbitrarily choose processor zero to take over
    // this job.
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i=0;
             i<Utilities::MPI::n_mpi_processes(mpi_communicator);
             ++i)
          filenames.push_back ("solution-" +
                               Utilities::int_to_string (cycle, 2) +
                               "." +
                               Utilities::int_to_string (i, 4) +
                               ".vtu");

        std::ofstream master_output (("solution-" +
                                      Utilities::int_to_string (cycle, 2) +
                                      ".pvtu").c_str());
        data_out.write_pvtu_record (master_output, filenames);
      }
  }




  template <int dim>
  void LaplaceProblem<dim>::run ()
  {
    const unsigned int n_cycles = 1;
    for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
      {
        pcout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
          {
            GridGenerator::hyper_cube (triangulation);
            triangulation.refine_global (6);
          }
        else
          //refine_grid ();
          triangulation.refine_global(1);

        setup_system ();
        //output_results (1);

        pcout << "   Number of active cells:       "
              << triangulation.n_global_active_cells()
              << std::endl
              << "   Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;

        //assemble_system ();
        solve ();
//        VectorTools::interpolate(dof_handler,
//                                 ExactSolution<dim>(),
//                                 x);
//        x=0;

        locally_relevant_solution = x;
        assemble_residual(residual, locally_relevant_solution);
        pcout << "residual: " << residual.l2_norm() << std::endl;

        if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
          {
            TimerOutput::Scope t(computing_timer, "output");
            output_results (cycle);
          }

        //computing_timer.print_summary ();
        computing_timer.reset ();

        pcout << std::endl;
      }
  }
}



// @sect4{main()}

// The final function, <code>main()</code>, again has the same structure as in
// all other programs, in particular step-6. Like in the other programs that
// use PETSc, we have to initialize and finalize PETSc, which is done using the
// helper object MPI_InitFinalize.
//
// Note how we enclose the use the use of the LaplaceProblem class in a pair
// of braces. This makes sure that all member variables of the object are
// destroyed by the time we destroy the mpi_initialization object. Not doing
// this will lead to strange and hard to debug errors when
// <code>PetscFinalize</code> first deletes all PETSc vectors that are still
// around, and the destructor of the LaplaceProblem class then tries to delete
// them again.
int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace Step40;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      LaplaceProblem<2> laplace_problem_2d;
      laplace_problem_2d.run ();
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
