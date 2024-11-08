/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2010 - 2024 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 *
 * Authors: Wolfgang Bangerth, Texas A&M University, 2009, 2010
 *          Timo Heister, University of Goettingen, 2009, 2010
 */


// @sect3{Include files}
//
// Most of the include files we need for this program have already been
// discussed in previous programs. In particular, all of the following should
// already be familiar friends:
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/linear_operator_tools.h>

// This program can use either PETSc or Trilinos for its parallel
// algebra needs. By default, if deal.II has been configured with
// PETSc, it will use PETSc. Otherwise, the following few lines will
// check that deal.II has been configured with Trilinos and take that.
//
// But there may be cases where you want to use Trilinos, even though
// deal.II has *also* been configured with PETSc, for example to
// compare the performance of these two libraries. To do this,
// add the following \#define to the source code:
// @code
// #define FORCE_USE_OF_TRILINOS
// @endcode
//
// Using this logic, the following lines will then import either the
// PETSc or Trilinos wrappers into the namespace `LA` (for linear
// algebra). In the former case, we are also defining the macro
// `USE_PETSC_LA` so that we can detect if we are using PETSc (see
// solve() for an example where this is necessary).
namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA


#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

// The following, however, will be new or be used in new roles. Let's walk
// through them. The first of these will provide the tools of the
// Utilities::System namespace that we will use to query things like the
// number of processors associated with the current MPI universe, or the
// number within this universe the processor this job runs on is:
#include <deal.II/base/utilities.h>
// The next one provides a class, ConditionOStream that allows us to write
// code that would output things to a stream (such as <code>std::cout</code>)
// on every processor but throws the text away on all but one of them. We
// could achieve the same by simply putting an <code>if</code> statement in
// front of each place where we may generate output, but this doesn't make the
// code any prettier. In addition, the condition whether this processor should
// or should not produce output to the screen is the same every time -- and
// consequently it should be simple enough to put it into the statements that
// generate output itself.
#include <deal.II/base/conditional_ostream.h>
// After these preliminaries, here is where it becomes more interesting. As
// mentioned in the @ref distributed topic, one of the fundamental truths of
// solving problems on large numbers of processors is that there is no way for
// any processor to store everything (e.g. information about all cells in the
// mesh, all degrees of freedom, or the values of all elements of the solution
// vector). Rather, every processor will <i>own</i> a few of each of these
// and, if necessary, may <i>know</i> about a few more, for example the ones
// that are located on cells adjacent to the ones this processor owns
// itself. We typically call the latter <i>ghost cells</i>, <i>ghost nodes</i>
// or <i>ghost elements of a vector</i>. The point of this discussion here is
// that we need to have a way to indicate which elements a particular
// processor owns or need to know of. This is the realm of the IndexSet class:
// if there are a total of $N$ cells, degrees of freedom, or vector elements,
// associated with (non-negative) integral indices $[0,N)$, then both the set
// of elements the current processor owns as well as the (possibly larger) set
// of indices it needs to know about are subsets of the set $[0,N)$. IndexSet
// is a class that stores subsets of this set in an efficient format:
#include <deal.II/base/index_set.h>
// The next header file is necessary for a single function,
// SparsityTools::distribute_sparsity_pattern. The role of this function will
// be explained below.
#include <deal.II/lac/sparsity_tools.h>
// The final two, new header files provide the class
// parallel::distributed::Triangulation that provides meshes distributed
// across a potentially very large number of processors, while the second
// provides the namespace parallel::distributed::GridRefinement that offers
// functions that can adaptively refine such distributed meshes:
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <fstream>
#include <iostream>

namespace Step40
{
  using namespace dealii;

  // @sect3{The <code>LaplaceProblem</code> class template}

  // Next let's declare the main class of this program. Its structure is
  // almost exactly that of the step-6 tutorial program. The only significant
  // differences are:
  // - The <code>mpi_communicator</code> variable that
  //   describes the set of processors we want this code to run on. In practice,
  //   this will be MPI_COMM_WORLD, i.e. all processors the batch scheduling
  //   system has assigned to this particular job.
  // - The presence of the <code>pcout</code> variable of type ConditionOStream.
  // - The obvious use of parallel::distributed::Triangulation instead of
  // Triangulation.
  // - The presence of two IndexSet objects that denote which sets of degrees of
  //   freedom (and associated elements of solution and right hand side vectors)
  //   we own on the current processor and which we need (as ghost elements) for
  //   the algorithms in this program to work.
  // - The fact that all matrices and vectors are now distributed. We use
  //   either the PETSc or Trilinos wrapper classes so that we can use one of
  //   the sophisticated preconditioners offered by Hypre (with PETSc) or ML
  //   (with Trilinos). Note that as part of this class, we store a solution
  //   vector that does not only contain the degrees of freedom the current
  //   processor owns, but also (as ghost elements) all those vector elements
  //   that correspond to "locally relevant" degrees of freedom (i.e. all
  //   those that live on locally owned cells or the layer of ghost cells that
  //   surround it).
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide();

    virtual void   vector_value(const Point<dim> &p,
                                Vector<double>   &value) const override;
    virtual double inside_out_factor(const Point<dim> &p) const;

    double viscosity(const Point<dim> &p);

  private:
    double                dynamic_viscosity_ratio;
    unsigned int          n_sinkers;
    std::vector<Point<3>> centers;
    double                delta;
    double                omega;
    double                beta;
  };

  template <int dim>
  RightHandSide<dim>::RightHandSide()
    : n_sinkers(8)
    , delta(200.0)
    , omega(0.1)
    , beta(10)
    , dynamic_viscosity_ratio(1e4)
  {
    centers.resize(75);
    centers[0] = Point<3>(2.4257829890e-01, 1.3469574514e-02, 3.8313885004e-01);
    centers[1] = Point<3>(4.1465269048e-01, 6.7768972864e-02, 9.9312692973e-01);
    centers[2] = Point<3>(4.8430804651e-01, 7.6533776604e-01, 3.1833815403e-02);
    centers[3] = Point<3>(3.0935481671e-02, 9.3264044027e-01, 8.8787953411e-01);
    centers[4] = Point<3>(5.9132973039e-01, 4.7877868473e-01, 8.3335433660e-01);
    centers[5] = Point<3>(1.8633519681e-01, 7.3565270739e-01, 1.1505317181e-01);
    centers[6] = Point<3>(6.9865863058e-01, 3.5560411138e-01, 6.3830000658e-01);
    centers[7] = Point<3>(9.0821050755e-01, 2.9400041480e-01, 2.6497158886e-01);
    centers[8] = Point<3>(3.7749399775e-01, 5.4162011554e-01, 9.2818150340e-03);
    centers[9] = Point<3>(8.5247022139e-01, 4.6701098395e-01, 5.3607231962e-02);
    centers[10] =
      Point<3>(9.7674759057e-01, 1.9675474344e-01, 8.5697294067e-01);
    centers[11] =
      Point<3>(1.4421375987e-01, 8.0066218823e-01, 7.2939761948e-01);
    centers[12] =
      Point<3>(9.8579064709e-01, 1.8340570954e-01, 4.9976021075e-01);
    centers[13] =
      Point<3>(4.6986202126e-01, 9.7099129947e-01, 4.5077026191e-01);
    centers[14] =
      Point<3>(9.5791877292e-02, 9.7408164664e-01, 3.9023506101e-01);
    centers[15] =
      Point<3>(6.8067035576e-01, 2.6669318800e-02, 2.3124107450e-01);
    centers[16] =
      Point<3>(4.6873909443e-01, 9.7960100555e-02, 4.1541002524e-01);
    centers[17] =
      Point<3>(7.9629418710e-01, 3.1640260216e-01, 7.7853444953e-01);
    centers[18] =
      Point<3>(8.2849331472e-01, 4.8714042059e-01, 3.6904878000e-01);
    centers[19] =
      Point<3>(6.0284549678e-01, 2.4264360789e-02, 8.1111178631e-01);
    centers[20] =
      Point<3>(3.5579259291e-01, 8.0610905439e-01, 2.7487712366e-01);
    centers[21] =
      Point<3>(8.5981739865e-01, 9.5101905612e-01, 7.7727618477e-01);
    centers[22] =
      Point<3>(6.8083745971e-01, 8.3518540665e-01, 9.6112961413e-01);
    centers[23] =
      Point<3>(7.0542474869e-01, 7.3751226102e-02, 5.3685709440e-01);
    centers[24] =
      Point<3>(9.5718558131e-01, 4.1806501915e-01, 4.1877679639e-01);
    centers[25] =
      Point<3>(3.8161700050e-01, 8.3692747440e-01, 2.4006224854e-01);
    centers[26] =
      Point<3>(7.2621119848e-01, 4.3161282150e-01, 1.1669089744e-01);
    centers[27] =
      Point<3>(2.2391322592e-01, 3.0958795748e-01, 2.4480139429e-01);
    centers[28] =
      Point<3>(3.7703382754e-01, 8.0753940242e-01, 3.1473643301e-01);
    centers[29] =
      Point<3>(7.7522956709e-01, 2.8333410774e-01, 9.9634871585e-01);
    centers[30] =
      Point<3>(6.3286731189e-01, 6.0091089904e-01, 5.0948022423e-01);
    centers[31] =
      Point<3>(8.3412860373e-01, 1.9944285005e-01, 3.5980841627e-02);
    centers[32] =
      Point<3>(7.3000523063e-01, 1.9791117972e-01, 2.9319749786e-01);
    centers[33] =
      Point<3>(7.7034656693e-01, 2.1475035521e-01, 3.0922000730e-01);
    centers[34] =
      Point<3>(6.0662675677e-02, 5.5759010630e-01, 4.1691651960e-01);
    centers[35] =
      Point<3>(1.1594487686e-01, 6.8554530558e-01, 9.5995079957e-01);
    centers[36] =
      Point<3>(2.7973348288e-02, 1.4806467395e-01, 5.2297503060e-01);
    centers[37] =
      Point<3>(6.4133927209e-01, 9.8914607800e-01, 5.7813295237e-01);
    centers[38] =
      Point<3>(6.8053043246e-01, 6.7497840462e-01, 3.6204645148e-01);
    centers[39] =
      Point<3>(9.1470996426e-01, 5.3036934674e-01, 9.1761070439e-01);
    centers[40] =
      Point<3>(2.8310876353e-01, 2.0898862472e-01, 4.7181570645e-01);
    centers[41] =
      Point<3>(8.0657831198e-01, 1.6168943288e-01, 5.1429839456e-01);
    centers[42] =
      Point<3>(8.1311740159e-01, 6.4168478858e-02, 4.7962416312e-01);
    centers[43] =
      Point<3>(4.3309508843e-02, 9.0291512474e-01, 2.9450144167e-01);
    centers[44] =
      Point<3>(6.8573011443e-01, 6.6033273035e-02, 8.2121989495e-01);
    centers[45] =
      Point<3>(2.4277445452e-01, 3.1025718772e-01, 4.9255406554e-01);
    centers[46] =
      Point<3>(3.5617944848e-01, 3.0799053857e-01, 3.9698166931e-01);
    centers[47] =
      Point<3>(7.0916077621e-02, 8.8651657239e-01, 6.8403214295e-01);
    centers[48] =
      Point<3>(5.2822650202e-01, 9.0281945043e-01, 6.8650344000e-01);
    centers[49] =
      Point<3>(6.3316007640e-02, 1.5214040370e-01, 2.3765034985e-02);
    centers[50] =
      Point<3>(4.1894298765e-01, 1.7479340461e-01, 7.5275125343e-01);
    centers[51] =
      Point<3>(4.9031640053e-01, 7.4774375406e-01, 3.2927456281e-01);
    centers[52] =
      Point<3>(1.1757708859e-01, 1.1812786251e-01, 3.7498524244e-01);
    centers[53] =
      Point<3>(3.7696964032e-01, 7.2874483733e-01, 1.4480990830e-02);
    centers[54] =
      Point<3>(3.8201288152e-01, 4.9049964756e-01, 8.2757658503e-01);
    centers[55] =
      Point<3>(7.9664661586e-02, 9.2396727806e-01, 1.1804237828e-01);
    centers[56] =
      Point<3>(9.3825167927e-01, 1.9597347043e-01, 7.2611756191e-01);
    centers[57] =
      Point<3>(8.5786301170e-01, 1.0363770514e-01, 8.3891028205e-01);
    centers[58] =
      Point<3>(5.6511039453e-01, 8.1040084307e-01, 4.0696941614e-01);
    centers[59] =
      Point<3>(9.3497714490e-01, 1.6087440083e-01, 8.1605472361e-01);
    centers[60] =
      Point<3>(4.3173963829e-01, 2.4810082244e-01, 8.3052277138e-01);
    centers[61] =
      Point<3>(5.9621858625e-01, 6.4577903070e-01, 6.0816894547e-01);
    centers[62] =
      Point<3>(4.9546643556e-01, 3.0438243752e-01, 7.5562733447e-01);
    centers[63] =
      Point<3>(8.2861043319e-01, 4.5555055302e-01, 4.3814466774e-01);
    centers[64] =
      Point<3>(8.9743076959e-01, 1.1894442752e-01, 9.8993320995e-02);
    centers[65] =
      Point<3>(6.9884936497e-01, 5.6127713367e-01, 3.8478565932e-01);
    centers[66] =
      Point<3>(9.2576270966e-02, 9.2938612771e-01, 1.9264837596e-01);
    centers[67] =
      Point<3>(8.4125479722e-01, 9.6937695284e-01, 3.1844636161e-01);
    centers[68] =
      Point<3>(1.2799954700e-01, 2.8838638276e-01, 9.0993508972e-01);
    centers[69] =
      Point<3>(2.7905288352e-01, 4.1813262758e-02, 7.5550716964e-01);
    centers[70] =
      Point<3>(8.0900019305e-01, 8.6624463269e-01, 9.7354159503e-01);
    centers[71] =
      Point<3>(3.1358765965e-01, 4.6779574243e-01, 2.4304298462e-01);
    centers[72] =
      Point<3>(8.2344259034e-01, 5.9961585635e-01, 7.4369772512e-01);
    centers[73] =
      Point<3>(3.2766604253e-01, 8.3176720460e-02, 9.5114077951e-01);
    centers[74] =
      Point<3>(8.2308128282e-01, 5.2712029523e-01, 3.1080186614e-01);
  }
  template <>
  double RightHandSide<3>::inside_out_factor(const Point<3> &p) const
  {
    double chi = 1.0;

    for (unsigned int s = 0; s < n_sinkers; ++s)
      {
        double dist = p.distance(centers[s]);
        double temp =
          1 - std::exp(-delta * std::pow(std::max(0.0, dist - omega / 2.0), 2));
        chi *= temp;
      }
    return chi;
  }


  template <int dim>
  double RightHandSide<dim>::viscosity(const Point<dim> &p)
  {
    double mu_min = pow(dynamic_viscosity_ratio, -1.0 / 2);
    double mu_max = pow(dynamic_viscosity_ratio, 1.0 / 2);
    return inside_out_factor(p) * mu_min +
           (1 - inside_out_factor(p)) * mu_max; // TODO fix me
  }

  template <int dim>
  void RightHandSide<dim>::vector_value(const Point<dim> &p,
                                        Vector<double>   &values) const
  {
    double Chi = 1.0;
    for (unsigned int s = 0; s < n_sinkers; ++s)
      {
        double dist = p.distance(centers[s]);
        double temp =
          1 - std::exp(-delta * std::pow(std::max(0.0, dist - omega / 2.0), 2));
        Chi *= temp;
      }

    values[0] = 0.0;
    values[1] = 0.0;
    values[2] = beta * (Chi - 1);
    // values[3] = 0.0;

    return;
  }
  template <int dim>
  class LaplaceProblem
  {
  public:
    LaplaceProblem();

    void run();

  private:
    void               setup_system();
    void               assemble_system();
    void               solve();
    void               refine_grid();
    void               output_results(const unsigned int cycle);
    RightHandSide<dim> right_hand_side;
    MPI_Comm           mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    const FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints;

    LA::MPI::SparseMatrix system_matrix;
    LA::MPI::Vector       locally_relevant_solution;
    LA::MPI::Vector       system_rhs;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;
  };


  // @sect3{The <code>LaplaceProblem</code> class implementation}

  // @sect4{Constructor}

  // Constructors and destructors are rather trivial. In addition to what we
  // do in step-6, we set the set of processors we want to work on to all
  // machines available (MPI_COMM_WORLD); ask the triangulation to ensure that
  // the mesh remains smooth and free to refined islands, for example; and
  // initialize the <code>pcout</code> variable to only allow processor zero
  // to output anything. The final piece is to initialize a timer that we
  // use to determine how much compute time the different parts of the program
  // take:
  template <int dim>
  LaplaceProblem<dim>::LaplaceProblem()
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening))
    , fe(2)
    , dof_handler(triangulation)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::never,
                      TimerOutput::wall_times)
  {}



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
  void LaplaceProblem<dim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs(fe);

    pcout << "   Number of active cells:       "
          << triangulation.n_global_active_cells() << std::endl
          << "   Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;


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
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

    // Next, let us initialize the solution and right hand side vectors. As
    // mentioned above, the solution vector we seek does not only store
    // elements we own, but also ghost entries; on the other hand, the right
    // hand side vector only needs to have the entries the current processor
    // owns since all we will ever do is write into it, never read from it on
    // locally owned cells (of course the linear solvers will read from it,
    // but they do not care about the geometric location of degrees of
    // freedom).
    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    // The next step is to compute hanging node and boundary value
    // constraints, which we combine into a single object storing all
    // constraints.
    //
    // As with all other things in %parallel, the mantra must be that no
    // processor can store all information about the entire universe. As a
    // consequence, we need to tell the AffineConstraints object for which
    // degrees of freedom it can store constraints and for which it may not
    // expect any information to store. In our case, as explained in the
    // @ref distributed topic, the degrees of freedom we need to care about on
    // each processor are the locally relevant ones, so we pass this to the
    // AffineConstraints::reinit() function as a second argument. A further
    // optimization, AffineConstraint can avoid certain operations if you also
    // provide it with the set of locally owned degrees of freedom -- the
    // first argument to AffineConstraints::reinit().
    //
    // (What would happen if we didn't pass this information to
    // AffineConstraints, for example if we called the argument-less version of
    // AffineConstraints::reinit() typically used in non-parallel codes? In that
    // case, the AffineConstraints class will allocate an array
    // with length equal to the largest DoF index it has seen so far. For
    // processors with large numbers of MPI processes, this may be very large --
    // maybe on the order of billions. The program would then allocate more
    // memory than for likely all other operations combined for this single
    // array. Fortunately, recent versions of deal.II would trigger an assertion
    // that tells you that this is considered a bug.)
    constraints.clear();
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    if (false)
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               Functions::ZeroFunction<dim>(),
                                               constraints);
    else
      std::cout << "not setting boundary values" << std::endl;

    constraints.close();

    // The last part of this function deals with initializing the matrix with
    // accompanying sparsity pattern. As in previous tutorial programs, we use
    // the DynamicSparsityPattern as an intermediate with which we
    // then initialize the system matrix. To do so, we have to tell the sparsity
    // pattern its size, but as above, there is no way the resulting object will
    // be able to store even a single pointer for each global degree of
    // freedom; the best we can hope for is that it stores information about
    // each locally relevant degree of freedom, i.e., all those that we may
    // ever touch in the process of assembling the matrix (the
    // @ref distributed_paper "distributed computing paper" has a long
    // discussion why one really needs the locally relevant, and not the small
    // set of locally active degrees of freedom in this context).
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
    DynamicSparsityPattern dsp(locally_relevant_dofs);

    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               dof_handler.locally_owned_dofs(),
                                               mpi_communicator,
                                               locally_relevant_dofs);

    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);
  }



  // @sect4{LaplaceProblem::assemble_system}

  // The function that then assembles the linear system is comparatively
  // boring, being almost exactly what we've seen before. The points to watch
  // out for are:
  // - Assembly must only loop over locally owned cells. There
  //   are multiple ways to test that; for example, we could compare a cell's
  //   subdomain_id against information from the triangulation as in
  //   <code>cell->subdomain_id() ==
  //   triangulation.locally_owned_subdomain()</code>, or skip all cells for
  //   which the condition <code>cell->is_ghost() ||
  //   cell->is_artificial()</code> is true. The simplest way, however, is to
  //   simply ask the cell whether it is owned by the local processor.
  // - Copying local contributions into the global matrix must include
  //   distributing constraints and boundary values not just from the local
  //   matrix and vector into the global ones, but in the process
  //   also -- possibly -- from one MPI process to other processes if the
  //   entries we want to write to are not stored on the current process.
  //   Interestingly, this requires essentially no additional work: The
  //   AffineConstraints class we already used in step-6 is perfectly
  //   capable to also do this in parallel, and the only difference in this
  //   regard is that at the very end of the function, we have to call a
  //   `compress()` function on the global matrix and right hand side vector
  //   objects (see the description of what this does just before these calls).
  // - The way we compute the right hand side (given the
  //   formula stated in the introduction) may not be the most elegant but will
  //   do for a program whose focus lies somewhere entirely different.
  template <int dim>
  void LaplaceProblem<dim>::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "assembly");

    const QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          cell_matrix = 0.;
          cell_rhs    = 0.;

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              double eta =
                right_hand_side.viscosity(fe_values.quadrature_point(q_point));
              Vector<double> rhs_vector(3);
              right_hand_side.vector_value(fe_values.quadrature_point(q_point),
                                           rhs_vector);
              const double rhs_value = rhs_vector[2];
              // const double rhs_value =
              //   (fe_values.quadrature_point(q_point)[1] >
              //        0.5 +
              //          0.25 * std::sin(4.0 * numbers::PI *
              //                          fe_values.quadrature_point(q_point)[0])
              //                          ?
              //      1. :
              //      -1.);

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    cell_matrix(i, j) +=
                      eta * fe_values.shape_grad(i, q_point) *
                      fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point);

                  cell_rhs(i) += rhs_value *                         //
                                 fe_values.shape_value(i, q_point) * //
                                 fe_values.JxW(q_point);
                }
            }

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);
        }

    // In the operations above, specifically the call to
    // `distribute_local_to_global()` in the last line, every MPI
    // process was only working on its local data. If the operation
    // required adding something to a matrix or vector entry that is
    // not actually stored on the current process, then the matrix or
    // vector object keeps track of this for a later data exchange,
    // but for efficiency reasons, this part of the operation is only
    // queued up, rather than executed right away. But now that we got
    // here, it is time to send these queued-up additions to those
    // processes that actually own these matrix or vector entries. In
    // other words, we want to "finalize" the global data
    // structures. This is done by invoking the function `compress()`
    // on both the matrix and vector objects. See
    // @ref GlossCompress "Compressing distributed objects"
    // for more information on what `compress()` actually does.
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }


  template <class VectorType>
  struct Nullspace
  {
    std::vector<VectorType> basis;
  };


  template <typename Range,
            typename Domain,
            typename Payload,
            class Op,
            class VectorType>
  LinearOperator<Range, Domain, Payload>
  my_operator(Op                                     &op,
              LinearOperator<Range, Domain, Payload> &exemplar,
              Nullspace<VectorType>                  &nullspace)
  {
    LinearOperator<Range, Domain, Payload> return_op;

    return_op.reinit_range_vector  = exemplar.reinit_range_vector;
    return_op.reinit_domain_vector = exemplar.reinit_domain_vector;

    return_op.vmult = [&](Range &dest, const Domain &src) {
      // std::cout << "before vmult" << std::endl;
      op.vmult(dest, src); // dest = Phi(src)

      // std::cout << "projection" << std::endl;
      //  Projection.
      for (unsigned int i = 0; i < nullspace.basis.size(); ++i)
        {
          double inner_product = nullspace.basis[i] * dest;
          dest.add(-1.0 * inner_product, nullspace.basis[i]);
        }
      //  std::cout << "ok" << std::endl;
    };

    // return_op.vmult_add = [&](Range &dest, const Domain &src) {
    //     std::cout << "before vmult_add" << std::endl;
    //     op.vmult_add(dest, src);  // dest += Phi(src)
    //     std::cout << "after vmult_add" << std::endl;
    // };

    // return_op.Tvmult = [&](Domain &dest, const Range &src) {
    //     std::cout << "before Tvmult" << std::endl;
    //     op.Tvmult(dest, src);
    //     std::cout << "after Tvmult" << std::endl;
    // };

    // return_op.Tvmult_add = [&](Domain &dest, const Range &src) {
    //     std::cout << "before Tvmult_add" << std::endl;
    //     op.Tvmult_add(dest, src);
    //     std::cout << "after Tvmult_add" << std::endl;
    // };

    return return_op;
  }

  // @sect4{LaplaceProblem::solve}

  // Even though solving linear systems on potentially tens of thousands of
  // processors is by far not a trivial job, the function that does this is --
  // at least at the outside -- relatively simple. Most of the parts you've
  // seen before. There are really only two things worth mentioning:
  // - Solvers and preconditioners are built on the deal.II wrappers of PETSc
  //   and Trilinos functionality. It is relatively well known that the
  //   primary bottleneck of massively %parallel linear solvers is not
  //   actually the communication between processors, but the fact that it is
  //   difficult to produce preconditioners that scale well to large numbers
  //   of processors. Over the second half of the first decade of the 21st
  //   century, it has become clear that algebraic multigrid (AMG) methods
  //   turn out to be extremely efficient in this context, and we will use one
  //   of them -- either the BoomerAMG implementation of the Hypre package
  //   that can be interfaced to through PETSc, or a preconditioner provided
  //   by ML, which is part of Trilinos -- for the current program. The rest
  //   of the solver itself is boilerplate and has been shown before. Since
  //   the linear system is symmetric and positive definite, we can use the CG
  //   method as the outer solver.
  // - Ultimately, we want a vector that stores not only the elements
  //   of the solution for degrees of freedom the current processor owns, but
  //   also all other locally relevant degrees of freedom. On the other hand,
  //   the solver itself needs a vector that is uniquely split between
  //   processors, without any overlap. We therefore create a vector at the
  //   beginning of this function that has these properties, use it to solve the
  //   linear system, and only assign it to the vector we want at the very
  //   end. This last step ensures that all ghost elements are also copied as
  //   necessary.
  template <int dim>
  void LaplaceProblem<dim>::solve()
  {
    Nullspace<LA::MPI::Vector> nullspace;

    nullspace.basis.emplace_back(locally_owned_dofs, mpi_communicator);
    LA::MPI::Vector &vec = nullspace.basis[0];
    if (false)
      {
        // use the constant vector 1
        vec = 0.0;
        vec.add(1.0);
      }
    else
      {
        // use the vector f(x)=1

        AffineConstraints<double> c;
        c.close();

        const unsigned int gauss_degree = fe.degree + 1;

        QGauss<dim>                          quadrature_formula(gauss_degree);
        std::vector<types::global_dof_index> local_dof_indices(
          fe.n_dofs_per_cell());

        FEValues<dim, dim> fe_values(fe,
                                     quadrature_formula,
                                     update_JxW_values | update_values);
        Vector<double>     local_constraint(fe_values.dofs_per_cell);

        for (const auto &cell : dof_handler.active_cell_iterators())

          if (cell->is_locally_owned())
            {
              local_constraint = 0;
              fe_values.reinit(cell);
              for (unsigned int q_point = 0;
                   q_point < fe_values.n_quadrature_points;
                   ++q_point)
                for (unsigned int i = 0; i < fe_values.dofs_per_cell; ++i)
                  local_constraint(i) +=
                    fe_values.shape_value(i, q_point) * fe_values.JxW(q_point);


              cell->get_dof_indices(local_dof_indices);
              if (true)
                {
                  // no hanging node constraints:
                  c.distribute_local_to_global(local_constraint,
                                               local_dof_indices,
                                               vec);
                }
              else
                {
                  // with constraints:
                  constraints.distribute_local_to_global(local_constraint,
                                                         local_dof_indices,
                                                         vec);
                }
            }
        vec.compress(VectorOperation::add);
      }
    vec /= vec.l2_norm();

    TimerOutput::Scope t(computing_timer, "solve");

    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);

    SolverControl                     solver_control(dof_handler.n_dofs(),
                                 1e-6 * system_rhs.l2_norm());
    dealii::SolverCG<LA::MPI::Vector> solver(solver_control);


    LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef USE_PETSC_LA
    data.symmetric_operator = true;
#else
    /* Trilinos defaults are good */
#endif
    LA::MPI::PreconditionAMG preconditioner;
    preconditioner.initialize(system_matrix, data);

    auto matrix_op  = linear_operator<LA::MPI::Vector>(system_matrix);
    auto pmatrix_op = my_operator(system_matrix, matrix_op, nullspace);
    auto prec_op    = my_operator(preconditioner, matrix_op, nullspace);


    double r = system_rhs * nullspace.basis[0];
    std::cout << "before project RHS: " << r << std::endl;
    system_rhs.add(-r, nullspace.basis[0]);
    r = system_rhs * nullspace.basis[0];
    std::cout << "project RHS after:" << r << std::endl;

    solver.solve(pmatrix_op,
                 completely_distributed_solution,
                 system_rhs,
                 prec_op);

    pcout << "   Solved in " << solver_control.last_step() << " iterations."
          << std::endl;

    constraints.distribute(completely_distributed_solution);

    locally_relevant_solution = completely_distributed_solution;
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
  void LaplaceProblem<dim>::refine_grid()
  {
    TimerOutput::Scope t(computing_timer, "refine");

    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      QGauss<dim - 1>(fe.degree + 1),
      std::map<types::boundary_id, const Function<dim> *>(),
      locally_relevant_solution,
      estimated_error_per_cell);
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
      triangulation, estimated_error_per_cell, 0.3, 0.03);
    triangulation.execute_coarsening_and_refinement();
  }



  // @sect4{LaplaceProblem::output_results}

  // Compared to the corresponding function in step-6, the one here is
  // a tad more complicated. There are two reasons: the first one is
  // that we do not just want to output the solution but also for each
  // cell which processor owns it (i.e. which "subdomain" it is
  // in). Secondly, as discussed at length in step-17 and step-18,
  // generating graphical data can be a bottleneck in
  // parallelizing. In those two programs, we simply generate one
  // output file per process. That worked because the
  // parallel::shared::Triangulation cannot be used with large numbers
  // of MPI processes anyway.  But this doesn't scale: Creating a
  // single file per processor will overwhelm the filesystem with a
  // large number of processors.
  //
  // We here follow a more sophisticated approach that uses
  // high-performance, parallel IO routines using MPI I/O to write to
  // a small, fixed number of visualization files (here 8). We also
  // generate a .pvtu record referencing these .vtu files, which can
  // be opened directly in visualizatin tools like ParaView and VisIt.
  //
  // To start, the top of the function looks like it usually does. In addition
  // to attaching the solution vector (the one that has entries for all locally
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
  void LaplaceProblem<dim>::output_results(const unsigned int cycle)
  {
    TimerOutput::Scope t(computing_timer, "output");

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution, "u");

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    Vector<float> viscosity(triangulation.n_active_cells());
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            viscosity[cell->active_cell_index()] =
              right_hand_side.viscosity(cell->center());
          }
      }
    data_out.add_data_vector(viscosity, "eta");
    data_out.build_patches();

    // The final step is to write this data to disk. We write up to 8 VTU files
    // in parallel with the help of MPI-IO. Additionally a PVTU record is
    // generated, which groups the written VTU files.
    data_out.write_vtu_with_pvtu_record(
      "./", "solution", cycle, mpi_communicator, 2, 8);
  }



  // @sect4{LaplaceProblem::run}

  // The function that controls the overall behavior of the program is again
  // like the one in step-6. The minor difference are the use of
  // <code>pcout</code> instead of <code>std::cout</code> for output to the
  // console (see also step-17).
  //
  // A functional difference to step-6 is the use of a square domain and that
  // we start with a slightly finer mesh (5 global refinement cycles) -- there
  // just isn't much of a point showing a massively %parallel program starting
  // on 4 cells (although admittedly the point is only slightly stronger
  // starting on 1024).
  template <int dim>
  void LaplaceProblem<dim>::run()
  {
    pcout << "Running with "
#ifdef USE_PETSC_LA
          << "PETSc"
#else
          << "Trilinos"
#endif
          << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    const unsigned int n_cycles = 5;
    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
      {
        pcout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
          {
            GridGenerator::hyper_cube(triangulation);
            triangulation.refine_global(3);
          }
        else
          refine_grid();

        setup_system();
        assemble_system();
        solve();
        output_results(cycle);

        computing_timer.print_summary();
        computing_timer.reset();

        pcout << std::endl;
      }
  }
} // namespace Step40



// @sect4{main()}

// The final function, <code>main()</code>, again has the same structure as in
// all other programs, in particular step-6. Like the other programs that use
// MPI, we have to initialize and finalize MPI, which is done using the helper
// object Utilities::MPI::MPI_InitFinalize. The constructor of that class also
// initializes libraries that depend on MPI, such as p4est, PETSc, SLEPc, and
// Zoltan (though the last two are not used in this tutorial). The order here
// is important: we cannot use any of these libraries until they are
// initialized, so it does not make sense to do anything before creating an
// instance of Utilities::MPI::MPI_InitFinalize.
//
// After the solver finishes, the LaplaceProblem destructor will run followed
// by Utilities::MPI::MPI_InitFinalize::~MPI_InitFinalize(). This order is
// also important: Utilities::MPI::MPI_InitFinalize::~MPI_InitFinalize() calls
// <code>PetscFinalize</code> (and finalization functions for other
// libraries), which will delete any in-use PETSc objects. This must be done
// after we destruct the Laplace solver to avoid double deletion
// errors. Fortunately, due to the order of destructor call rules of C++, we
// do not need to worry about any of this: everything happens in the correct
// order (i.e., the reverse of the order of construction). The last function
// called by Utilities::MPI::MPI_InitFinalize::~MPI_InitFinalize() is
// <code>MPI_Finalize</code>: i.e., once this object is destructed the program
// should exit since MPI will no longer be available.
int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace Step40;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      LaplaceProblem<3> laplace_problem_2d;
      laplace_problem_2d.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
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
      std::cerr << std::endl
                << std::endl
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
