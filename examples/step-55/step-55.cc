/* ---------------------------------------------------------------------
 * Author: Ryan Grove, Clemson University, 2015
 *---------------------------------------------------------------------
 */
// DEV

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

#include <deal.II/base/timer.h>


/*
 * These, now, are the include necessary for the multilevel methods. The first one declares how
 * to handle Dirichlet boundary conditions on each of the levels of the multigrid method. For
 * the actual description of the degrees of freedom, we do not need any new include file because
 * DoFHandler already has all necessary methods implemented. We will only need to distribute the
 * DoFs for the levels further down.
 *
 * The rest of the include files deals with the mechanics of multigrid as a linear operator
 * (solver or preconditioner).
 */
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

int count_mm = 0;

namespace Step22
{
using namespace dealii;

// TIMO: New Way

/**
 * Implement the block Schur preconditioner for the Stokes system.
 */
template <class PreconditionerA, class PreconditionerMp>
class NewBlockSchurPreconditioner : public Subscriptor
{
  public:
    /**
     * @brief Constructor
     *
     * @param S The entire Stokes matrix
     * @param Spre The matrix whose blocks are used in the definition of
     *     the preconditioning of the Stokes matrix, i.e. containing approximations
     *     of the A and S blocks.
     * @param Mppreconditioner Preconditioner object for the Schur complement,
     *     typically chosen as the mass matrix.
     * @param Apreconditioner Preconditioner object for the matrix A.
     * @param do_solve_A A flag indicating whether we should actually solve with
     *     the matrix $A$, or only apply one preconditioner step with it.
     **/
    NewBlockSchurPreconditioner (const BlockSparseMatrix<double>                 &S,
    		                  const SparseMatrix<double>                 &P,
                              const PreconditionerMp                     &Mppreconditioner,
                              const PreconditionerA                      &Apreconditioner,
                              const bool                                do_solve_A);

         /**
         * Matrix vector product with this preconditioner object.
         */
        void vmult (BlockVector<double>       &dst,
                    const BlockVector<double> &src) const;

        unsigned int n_iterations_A() const;
        unsigned int n_iterations_S() const;

      private:
        /**
         * References to the various matrix object this preconditioner works on.
         */
        const BlockSparseMatrix<double> &stokes_matrix;
        const SparseMatrix<double> &pressure_mass_matrix;
        const PreconditionerMp                    &mp_preconditioner;
        const PreconditionerA                     &a_preconditioner;

        /**
         * Whether to actually invert the $\tilde A$ part of the preconditioner matrix
         * or to just apply a single preconditioner step with it.
         **/
        const bool do_solve_A;
        mutable unsigned int n_iterations_A_;
        mutable unsigned int n_iterations_S_;
};

template <class PreconditionerA, class PreconditionerMp>
   NewBlockSchurPreconditioner<PreconditionerA, PreconditionerMp>::
   NewBlockSchurPreconditioner (const BlockSparseMatrix<double>  &S,
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
  NewBlockSchurPreconditioner<PreconditionerA, PreconditionerMp>::
  n_iterations_A() const
  {
    return n_iterations_A_;
  }
  template <class PreconditionerA, class PreconditionerMp>
  unsigned int
  NewBlockSchurPreconditioner<PreconditionerA, PreconditionerMp>::
  n_iterations_S() const
  {
    return n_iterations_S_;
  }


  template <class PreconditionerA, class PreconditionerMp>
    void
    NewBlockSchurPreconditioner<PreconditionerA, PreconditionerMp>::
    vmult (BlockVector<double>       &dst,
           const BlockVector<double> &src) const
    {
      Vector<double> utmp(src.block(0));

      // first solve with the bottom left block, which we have built
      // as a mass matrix with the inverse of the viscosity
      {
        SolverControl solver_control(1000, 1e-6 * src.block(1).l2_norm());
        SolverCG<>    cg (solver_control);


                // mass matrix
                dst.block(1) = 0.0;
                cg.solve(pressure_mass_matrix,
                             dst.block(1), src.block(1),
                             mp_preconditioner);
                n_iterations_S_ += solver_control.last_step();


        dst.block(1) *= -1.0;
      }

      // apply the top right block
      {
        stokes_matrix.block(0,1).vmult(utmp, dst.block(1)); //B^T
        utmp*=-1.0;
        utmp.add(src.block(0));
      }

      // now either solve with the top left block (if do_solve_A==true)
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

// As explained in the introduction, we are going to use different
// preconditioners for two and three space dimensions, respectively. We
// distinguish between them by the use of the spatial dimension as a
// template parameter. See step-4 for details on templates. We are not going
// to create any preconditioner object here, all we do is to create class
// that holds a local typedef determining the preconditioner class so we can
// write our program in a dimension-independent way.
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
	void setup_dofs ();
	void setup_dofs_mg ();
	void assemble_system ();
	void assemble_system_mg ();
	void assemble_multigrid ();
	void solve (); //TODO: remove
	void solve_block ();
	void solve_mg (); //TODO: remove
	void solve_block_mg ();
	void output_results (const unsigned int refinement_cycle) const;
	void refine_mesh ();

	const unsigned int   degree;

	Triangulation<dim>   triangulation;
	FESystem<dim>        fe;
	FESystem<dim>        velocity_fe;
	DoFHandler<dim>      dof_handler;
	DoFHandler<dim>      velocity_dof_handler;

	ConstraintMatrix     constraints;
	ConstraintMatrix     velocity_constraints;
	/*
	 * We need an additional object for the hanging nodes constraints. They are handed to the transfer object
	 * in the multigrid. Since we call a compress inside the multigrid these constraints are not allowed to
	 * be inhomogeneous so we store them in different ConstraintMatrix objects.
	 */
	ConstraintMatrix     hanging_node_constraints;
	ConstraintMatrix     velocity_hanging_node_constraints;

	BlockSparsityPattern      sparsity_pattern;
	BlockSparsityPattern      sparsity_pattern_velocity;
	BlockSparseMatrix<double> system_matrix;

	BlockVector<double> solution;
	BlockVector<double> system_rhs;

	std::vector<double> dof_indices;
	std::vector<double> vel_dof_indices;

	// This one is new: We shall use a so-called shared pointer structure to
	// access the preconditioner. Shared pointers are essentially just a
	// convenient form of pointers. Several shared pointers can point to the
	// same object (just like regular pointers), but when the last shared
	// pointer object to point to a preconditioner object is deleted (for
	// example if a shared pointer object goes out of scope, if the class of
	// which it is a member is destroyed, or if the pointer is assigned a
	// different preconditioner object) then the preconditioner object pointed
	// to is also destroyed. This ensures that we don't have to manually track
	// in how many places a preconditioner object is still referenced, it can
	// never create a memory leak, and can never produce a dangling pointer to
	// an already destroyed object:
	std_cxx11::shared_ptr<typename InnerPreconditioner<dim>::type> A_preconditioner;

	/*
	 * The following members are the essential data structures for the multigrid method. The first two
	 * represent the sparsity patterns and the matrices on individual levels of the multilevel hierarchy,
	 * very much like the objects for the global mesh above.
	 *
	 * Then we have two new matrices only needed for multigrid methods with local smoothing on adaptive
	 * meshes. They convey data between the interior part of the refined region and the refinement edge,
	 * as outline in detail in mg_paper.
	 *
	 * The last object stores information about the boundary indices on each level and information about
	 * indices lying on a refinement edge between two different refinement levels. It thus serves a
	 * similar purpose as ConstraintMatrix, but on each level.
	 */
	MGLevelObject<SparsityPattern>        mg_sparsity_patterns;
	MGLevelObject<SparseMatrix<double> > mg_matrices;
	MGLevelObject<SparseMatrix<double> > mg_interface_matrices;
	MGConstrainedDoFs                     mg_constrained_dofs;

	TimerOutput computing_timer;
};

template <int dim>
class BoundaryValues : public Function<dim>
{
public:
	BoundaryValues () : Function<dim>(dim+1) {}

	virtual double value (const Point<dim>   &p,
			const unsigned int  component = 0) const;

	virtual void vector_value (const Point<dim> &p,
			Vector<double>   &value) const;
};

template <int dim>
double
BoundaryValues<dim>::value (const Point<dim>  &p,
		const unsigned int component) const
		{
	Assert (component < this->n_components,
			ExcIndexRange (component, 0, this->n_components));

	if (component == 0)
		return (p[0] < 0 ? -1 : (p[0] > 0 ? 1 : 0));
	return 0;
		}

template <int dim>
void
BoundaryValues<dim>::vector_value (const Point<dim> &p,
		Vector<double>   &values) const
		{
	for (unsigned int c=0; c<this->n_components; ++c)
		values(c) = BoundaryValues<dim>::value (p, c);
		}

template <int dim>
class BoundaryValuesVel : public Function<dim>
{
public:
	BoundaryValuesVel () : Function<dim>(dim) {}

	virtual double value (const Point<dim>   &p,
			const unsigned int  component = 0) const;

	virtual void vector_value (const Point<dim> &p,
			Vector<double>   &value) const;
};


template <int dim>
double
BoundaryValuesVel<dim>::value (const Point<dim>  &p,
		const unsigned int component) const
		{
	Assert (component < this->n_components,
			ExcIndexRange (component, 0, this->n_components));

	if (component == 0)
		return (p[0] < 0 ? -1 : (p[0] > 0 ? 1 : 0));
	return 0;
		}


template <int dim>
void
BoundaryValuesVel<dim>::vector_value (const Point<dim> &p,
		Vector<double>   &values) const
		{
	for (unsigned int c=0; c<this->n_components; ++c)
		values(c) = BoundaryValuesVel<dim>::value (p, c);
		}



template <int dim>
class RightHandSide : public Function<dim>
{
public:
	RightHandSide () : Function<dim>(dim+1) {}

	virtual double value (const Point<dim>   &p,
			const unsigned int  component = 0) const;

	virtual void vector_value (const Point<dim> &p,
			Vector<double>   &value) const;

};


template <int dim>
double
RightHandSide<dim>::value (const Point<dim>  &/*p*/,
		const unsigned int /*component*/) const
		{
	return 0;
		}


template <int dim>
void
RightHandSide<dim>::vector_value (const Point<dim> &p,
		Vector<double>   &values) const
		{
	for (unsigned int c=0; c<this->n_components; ++c)
		values(c) = RightHandSide<dim>::value (p, c);
		}





template <class Matrix, class Preconditioner>
class InverseMatrix : public Subscriptor
{
public:
	InverseMatrix (const Matrix         &m,
			const Preconditioner &preconditioner);

	void vmult (Vector<double>       &dst,
			const Vector<double> &src) const;

private:
	const SmartPointer<const Matrix> matrix;
	const SmartPointer<const Preconditioner> preconditioner;
};


template <class Matrix, class Preconditioner>
InverseMatrix<Matrix,Preconditioner>::InverseMatrix (const Matrix &m,
		const Preconditioner &preconditioner)
		:
		matrix (&m),
		preconditioner (&preconditioner)
		{}



template <class Matrix, class Preconditioner>
void InverseMatrix<Matrix,Preconditioner>::vmult (Vector<double>       &dst,
		const Vector<double> &src) const
		{
	SolverControl solver_control (src.size(), 1e-6*src.l2_norm());
	SolverCG<>    cg (solver_control);

	dst = 0;

	cg.solve (*matrix, dst, src, *preconditioner); // in Schur complement important that A inverse is accurate so need to do it this way
	// preconditioner needs to be A_Multigrid
	// Note: 10e-6 might not be accurate enoughG
		count_mm = count_mm + solver_control.last_step();
		}



template <class Preconditioner> // RG: Template on dim? Why doesn't it know about PreconditionMG?
class SchurComplement : public Subscriptor
{
public:
	SchurComplement (const BlockSparseMatrix<double> &system_matrix,
			const InverseMatrix<SparseMatrix<double>, Preconditioner> &A_inverse);

	void vmult (Vector<double>       &dst,
			const Vector<double> &src) const;

private:
	const SmartPointer<const BlockSparseMatrix<double> > system_matrix;
	const SmartPointer<const InverseMatrix<SparseMatrix<double>, Preconditioner> > A_inverse;

	mutable Vector<double> tmp1, tmp2;
};



template <class Preconditioner>
SchurComplement<Preconditioner>::
SchurComplement (const BlockSparseMatrix<double> &system_matrix,
		const InverseMatrix<SparseMatrix<double>,Preconditioner> &A_inverse)
		:
		system_matrix (&system_matrix),
		A_inverse (&A_inverse),
		tmp1 (system_matrix.block(0,0).m()),
		tmp2 (system_matrix.block(0,0).m())
		{}


template <class Preconditioner>
void SchurComplement<Preconditioner>::vmult (Vector<double>       &dst,
		const Vector<double> &src) const
		{// This is the transfer (understand)
	system_matrix->block(0,1).vmult (tmp1, src);    // multiply with the top right block: B
	A_inverse->vmult (tmp2, tmp1);                  // multiply with A^-1
	system_matrix->block(1,0).vmult (dst, tmp2);    // multiply with the bottom left block: B^T
		}

template <class PreconditionerA, class PreconditionerMp>
class BlockSchurPreconditioner : public Subscriptor
{
public:
	BlockSchurPreconditioner (const BlockSparseMatrix<double> &S,
			const InverseMatrix<SparseMatrix<double>,PreconditionerMp> &Mpinv,
			const PreconditionerA &Apreconditioner);
	void vmult (BlockVector<double> &dst,
			const BlockVector<double> &src) const;
private:
	const SmartPointer<const BlockSparseMatrix<double> > system_matrix;
	const SmartPointer<const InverseMatrix<SparseMatrix<double>,
	PreconditionerMp > > m_inverse;
	const PreconditionerA &a_preconditioner;
	mutable Vector<double> tmp;
};
template <class PreconditionerA, class PreconditionerMp>
BlockSchurPreconditioner<PreconditionerA, PreconditionerMp>::BlockSchurPreconditioner(
		const BlockSparseMatrix<double> &S,
		const InverseMatrix<SparseMatrix<double>,PreconditionerMp> &Mpinv,
		const PreconditionerA &Apreconditioner
)
:
system_matrix (&S),
m_inverse (&Mpinv),
a_preconditioner (Apreconditioner),
tmp (S.block(1,1).m())
{}
// Now the interesting function, the multiplication of
// the preconditioner with a BlockVector.
template <class PreconditionerA, class PreconditionerMp>
void BlockSchurPreconditioner<PreconditionerA, PreconditionerMp>::vmult (
		BlockVector<double> &dst,
		const BlockVector<double> &src) const
		{

//	    bool use_cg = false;
//	    if (use_cg == true)
//	    {
//			SolverControl solver_control (src.size(), 1e-6*src.l2_norm());
//			SolverCG<>    cg (solver_control);
//
//			dst = 0;
//
//			cg.solve (system_matrix, dst, src, *a_preconditioner);
//		}
//	    else
//	    {

	// Form u_new = A^{-1} u
	a_preconditioner.vmult (dst.block(0), src.block(0));
//	    }
	// Form tmp = - B u_new + p
	// (<code>SparseMatrix::residual</code>
	// does precisely this)
	system_matrix->block(1,0).residual(tmp, dst.block(0), src.block(1));
	// Change sign in tmp
	tmp *= -1;
	// Multiply by approximate Schur complement
	// (i.e. a pressure mass matrix)
	m_inverse->vmult (dst.block(1), tmp);
		}

template <int dim>
StokesProblem<dim>::StokesProblem (const unsigned int degree)
:
degree (degree),
triangulation (Triangulation<dim>::maximum_smoothing),
fe (FE_Q<dim>(degree+1), dim, // Finite element for whole system
		FE_Q<dim>(degree), 1),
		velocity_fe (FE_Q<dim>(degree+1), dim), // Velocity only finite element
		dof_handler (triangulation),
		velocity_dof_handler (triangulation),
		computing_timer (std::cout, TimerOutput::summary,
				TimerOutput::wall_times)
				{}

// @sect4{StokesProblem::setup_dofs}

 // Given a mesh, this function associates the degrees of freedom with it and
 // creates the corresponding matrices and vectors. At the beginning it also
 // releases the pointer to the preconditioner object (if the shared pointer
 // pointed at anything at all at this point) since it will definitely not be
 // needed any more after this point and will have to be re-computed after
 // assembling the matrix, and unties the sparse matrix from its sparsity
 // pattern object.
 //
 // We then proceed with distributing degrees of freedom and renumbering
 // them: In order to make the ILU preconditioner (in 3D) work efficiently,
 // it is important to enumerate the degrees of freedom in such a way that it
 // reduces the bandwidth of the matrix, or maybe more importantly: in such a
 // way that the ILU is as close as possible to a real LU decomposition. On
 // the other hand, we need to preserve the block structure of velocity and
 // pressure already seen in in step-20 and step-21. This is done in two
 // steps: First, all dofs are renumbered to improve the ILU and then we
 // renumber once again by components. Since
 // <code>DoFRenumbering::component_wise</code> does not touch the
 // renumbering within the individual blocks, the basic renumbering from the
 // first step remains. As for how the renumber degrees of freedom to improve
 // the ILU: deal.II has a number of algorithms that attempt to find
 // orderings to improve ILUs, or reduce the bandwidth of matrices, or
 // optimize some other aspect. The DoFRenumbering namespace shows a
 // comparison of the results we obtain with several of these algorithms
 // based on the testcase discussed here in this tutorial program. Here, we
 // will use the traditional Cuthill-McKee algorithm already used in some of
 // the previous tutorial programs.  In the <a href="#improved-ilu">section
 // on improved ILU</a> we're going to discuss this issue in more detail.

 // There is one more change compared to previous tutorial programs: There is
 // no reason in sorting the <code>dim</code> velocity components
 // individually. In fact, rather than first enumerating all $x$-velocities,
 // then all $y$-velocities, etc, we would like to keep all velocities at the
 // same location together and only separate between velocities (all
 // components) and pressures. By default, this is not what the
 // DoFRenumbering::component_wise function does: it treats each vector
 // component separately; what we have to do is group several components into
 // "blocks" and pass this block structure to that function. Consequently, we
 // allocate a vector <code>block_component</code> with as many elements as
 // there are components and describe all velocity components to correspond
 // to block 0, while the pressure component will form block 1:
 template <int dim>
 void StokesProblem<dim>::setup_dofs ()
 {
   A_preconditioner.reset ();
   system_matrix.clear ();

   dof_handler.distribute_dofs (fe);
   DoFRenumbering::Cuthill_McKee (dof_handler);

   std::vector<unsigned int> block_component (dim+1,0);
   block_component[dim] = 1;
   DoFRenumbering::component_wise (dof_handler, block_component);

   // Now comes the implementation of Dirichlet boundary conditions, which
   // should be evident after the discussion in the introduction. All that
   // changed is that the function already appears in the setup functions,
   // whereas we were used to see it in some assembly routine. Further down
   // below where we set up the mesh, we will associate the top boundary
   // where we impose Dirichlet boundary conditions with boundary indicator
   // 1.  We will have to pass this boundary indicator as second argument to
   // the function below interpolating boundary values.  There is one more
   // thing, though.  The function describing the Dirichlet conditions was
   // defined for all components, both velocity and pressure. However, the
   // Dirichlet conditions are to be set for the velocity only.  To this end,
   // we use a ComponentMask that only selects the velocity components. The
   // component mask is obtained from the finite element by specifying the
   // particular components we want. Since we use adaptively refined grids
   // the constraint matrix needs to be first filled with hanging node
   // constraints generated from the DoF handler. Note the order of the two
   // functions &mdash; we first compute the hanging node constraints, and
   // then insert the boundary values into the constraint matrix. This makes
   // sure that we respect H<sup>1</sup> conformity on boundaries with
   // hanging nodes (in three space dimensions), where the hanging node needs
   // to dominate the Dirichlet boundary values.
   {
     constraints.clear ();

     FEValuesExtractors::Vector velocities(0);
     DoFTools::make_hanging_node_constraints (dof_handler,
                                              constraints);
     VectorTools::interpolate_boundary_values (dof_handler,
                                               1,
                                               BoundaryValues<dim>(),
                                               constraints,
                                               fe.component_mask(velocities));
   }

   constraints.close ();

   // In analogy to step-20, we count the dofs in the individual components.
   // We could do this in the same way as there, but we want to operate on
   // the block structure we used already for the renumbering: The function
   // <code>DoFTools::count_dofs_per_block</code> does the same as
   // <code>DoFTools::count_dofs_per_component</code>, but now grouped as
   // velocity and pressure block via <code>block_component</code>.
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

   // The next task is to allocate a sparsity pattern for the system matrix
   // we will create. We could do this in the same way as in step-20,
   // i.e. directly build an object of type SparsityPattern through
   // DoFTools::make_sparsity_pattern. However, there is a major reason not
   // to do so: In 3D, the function DoFTools::max_couplings_between_dofs
   // yields a conservative but rather large number for the coupling between
   // the individual dofs, so that the memory initially provided for the
   // creation of the sparsity pattern of the matrix is far too much -- so
   // much actually that the initial sparsity pattern won't even fit into the
   // physical memory of most systems already for moderately-sized 3D
   // problems, see also the discussion in step-18.  Instead, we first build
   // a temporary object that uses a different data structure that doesn't
   // require allocating more memory than necessary but isn't suitable for
   // use as a basis of SparseMatrix or BlockSparseMatrix objects; in a
   // second step we then copy this object into an object of
   // BlockSparsityPattern. This is entirely analogous to what we already did
   // in step-11 and step-18.
   //
   // All this is done inside a new scope, which
   // means that the memory of <code>dsp</code> will be released once the
   // information has been copied to <code>sparsity_pattern</code>.
   {
     BlockDynamicSparsityPattern dsp (2,2);

     dsp.block(0,0).reinit (n_u, n_u);
     dsp.block(1,0).reinit (n_p, n_u);
     dsp.block(0,1).reinit (n_u, n_p);
     dsp.block(1,1).reinit (n_p, n_p);

     dsp.collect_sizes();

     DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints, false);
     sparsity_pattern.copy_from (dsp);
   }

   // Finally, the system matrix, solution and right hand side are created
   // from the block structure as in step-20:
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


template <int dim>
void StokesProblem<dim>::setup_dofs_mg ()
{
	//	computing_timer.enter_subsection ("Setup - Misc");
	// RG: used to rreset the preconditioner here
	system_matrix.clear ();

	dof_handler.distribute_dofs(fe); // Don't need multigrid dofs for whole problem

	velocity_dof_handler.distribute_dofs(velocity_fe); // Distribute only dofs for velocity
	velocity_dof_handler.distribute_mg_dofs (velocity_fe); // Multigrid only needs for velocity

	std::vector<unsigned int> block_component (dim+1,0);
	block_component[dim] = 1; // creates array (0,0,1) -- first do everything with index 0, then 1
	// can change to (0,1,2) and print sparsity pattern to see what you should see
	DoFRenumbering::component_wise (dof_handler, block_component); // Renumbers unknowns
	//DoFRenumbering::component_wise (velocity_dof_handler, block_component); // Needs renumbered as 01 if top is (0,1,2)

	constraints.clear ();
	velocity_constraints.clear ();
	hanging_node_constraints.clear ();
	velocity_hanging_node_constraints.clear ();

	FEValuesExtractors::Vector velocities(0); // This always knows how to use the dim somehow (start at 0 one) -
	// Explained in vector valued dealii step-20
	DoFTools::make_hanging_node_constraints (dof_handler, constraints);
	DoFTools::make_hanging_node_constraints (dof_handler, hanging_node_constraints);

	// RG: for adaptivity, these must be taken care of (correctly)
	DoFTools::make_hanging_node_constraints (velocity_dof_handler, velocity_constraints);
	DoFTools::make_hanging_node_constraints (velocity_dof_handler, velocity_hanging_node_constraints);

	VectorTools::interpolate_boundary_values (dof_handler,
			1,
			BoundaryValues<dim>(),
			constraints,
			fe.component_mask(velocities));
	VectorTools::interpolate_boundary_values (velocity_dof_handler, //has 2 components
			1,
			BoundaryValuesVel<dim>(), //has 2 components
			velocity_constraints);

	constraints.close ();
	velocity_constraints.close ();
	hanging_node_constraints.close ();
	velocity_hanging_node_constraints.close ();

	typename FunctionMap<dim>::type      boundary_condition_function_map;
	//BoundaryValues<dim>                   boundary_condition;
	BoundaryValuesVel<dim>                velocity_boundary_condition;
	//boundary_condition_function_map[1] = &boundary_condition; // Map boundary indicator 1 to boundary values
	boundary_condition_function_map[1] = &velocity_boundary_condition;
	//	computing_timer.leave_subsection();
	//	computing_timer.enter_subsection ("Setup - MG Stuff");
	/*
	 * The multigrid constraints have to be initialized. They need to know about the boundary values as well.
	 */
	mg_constrained_dofs.clear();
	mg_constrained_dofs.initialize(velocity_dof_handler, boundary_condition_function_map);
	const unsigned int n_levels = triangulation.n_levels();

	mg_interface_matrices.resize(0, n_levels-1);
	mg_interface_matrices.clear ();
	mg_matrices.resize(0, n_levels-1);
	mg_matrices.clear ();
	mg_sparsity_patterns.resize(0, n_levels-1);

	/*
	 * Now, we have to provide a matrix on each level. To this end, we first use the MGTools::make_sparsity_pattern
	 * function to first generate a preliminary compressed sparsity pattern on each level (see the Sparsity patterns
	 * module for more information on this topic) and then copy it over to the one we really want. The next step is
	 * to initialize both kinds of level matrices with these sparsity patterns.
	 *
	 * It may be worth pointing out that the interface matrices only have entries for degrees of freedom that sit
	 * at or next to the interface between coarser and finer levels of the mesh. They are therefore even sparser
	 * than the matrices on the individual levels of our multigrid hierarchy. If we were more concerned about memory
	 * usage (and possibly the speed with which we can multiply with these matrices), we should use separate and
	 * different sparsity patterns for these two kinds of matrices.
	 */
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
	//	computing_timer.leave_subsection();
	//	computing_timer.enter_subsection ("Setup - Block Stuff");
	std::vector<types::global_dof_index> dofs_per_block (2); //if you did 012 then you'd have three answers
	DoFTools::count_dofs_per_block (dof_handler, dofs_per_block, block_component);
	const unsigned int n_u = dofs_per_block[0],
			n_p = dofs_per_block[1];

	//	std::cout << "   Number of active cells: "
	//			<< triangulation.n_active_cells()
	//			<< std::endl
	//			<< "   Number of degrees of freedom: "
	//			<< dof_handler.n_dofs()
	//			<< " (" << n_u << '+' << n_p << ')'
	//			<< std::endl
	//			<< "   Number of velocity degrees of freedom: "
	//			<< velocity_dof_handler.n_dofs()
	//			<< std::endl;  //TODO: Output number of unknowns per level

	Assert (n_u == velocity_dof_handler.n_dofs(), ExcMessage ("Numbers of degrees of freedom must match for the velocity part of the main dof handler and the whole velocity dof handler."));

	//	computing_timer.enter_subsection ("Setup - Block Stuff - SP");
	{
		BlockDynamicSparsityPattern csp (2,2);

		//	computing_timer.enter_subsection ("Setup - Block Stuff - SP - Reinit");
		csp.block(0,0).reinit (n_u, n_u); //how big is each block?
		csp.block(1,0).reinit (n_p, n_u);
		csp.block(0,1).reinit (n_u, n_p);
		csp.block(1,1).reinit (n_p, n_p);
		//	computing_timer.leave_subsection();

		//	computing_timer.enter_subsection ("Setup - Block Stuff - SP - Sizes");
		csp.collect_sizes();
		//	computing_timer.leave_subsection();

		//	computing_timer.enter_subsection ("Setup - Block Stuff - SP - SP");
		DoFTools::make_sparsity_pattern (dof_handler, csp, constraints, false);
		sparsity_pattern.copy_from (csp);
		//	computing_timer.leave_subsection();
		//	std::ofstream out ("everything_sp.gnuplot");
		//	sparsity_pattern.print_gnuplot(out);

	}
	//	computing_timer.leave_subsection();
	//	computing_timer.enter_subsection ("Setup - Block Stuff - Reinit");
	system_matrix.reinit (sparsity_pattern); // Note: (system_matrix.block(0,0) should give you first block

	solution.reinit (2);
	solution.block(0).reinit (n_u);
	solution.block(1).reinit (n_p);
	solution.collect_sizes ();

	system_rhs.reinit (2);
	system_rhs.block(0).reinit (n_u);
	system_rhs.block(1).reinit (n_p);
	system_rhs.collect_sizes ();
	//	computing_timer.leave_subsection();
	//	computing_timer.leave_subsection();
}

// @sect4{StokesProblem::assemble_system}

// The assembly process follows the discussion in step-20 and in the
// introduction. We use the well-known abbreviations for the data structures
// that hold the local matrix, right hand side, and global numbering of the
// degrees of freedom for the present cell.
template <int dim>
void StokesProblem<dim>::assemble_system ()
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

  // Next, we need two objects that work as extractors for the FEValues
  // object. Their use is explained in detail in the report on @ref
  // vector_valued :
  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Scalar pressure (dim);

  // As an extension over step-20 and step-21, we include a few
  // optimizations that make assembly much faster for this particular
  // problem.  The improvements are based on the observation that we do a
  // few calculations too many times when we do as in step-20: The symmetric
  // gradient actually has <code>dofs_per_cell</code> different values per
  // quadrature point, but we extract it
  // <code>dofs_per_cell*dofs_per_cell</code> times from the FEValues object
  // - for both the loop over <code>i</code> and the inner loop over
  // <code>j</code>. In 3d, that means evaluating it $89^2=7921$ instead of
  // $89$ times, a not insignificant difference.
  //
  // So what we're going to do here is to avoid such repeated calculations
  // by getting a vector of rank-2 tensors (and similarly for the divergence
  // and the basis function value on pressure) at the quadrature point prior
  // to starting the loop over the dofs on the cell. First, we create the
  // respective objects that will hold these values. Then, we start the loop
  // over all cells and the loop over the quadrature points, where we first
  // extract these values. There is one more optimization we implement here:
  // the local matrix (as well as the global one) is going to be symmetric,
  // since all the operations involved are symmetric with respect to $i$ and
  // $j$. This is implemented by simply running the inner loop not to
  // <code>dofs_per_cell</code>, but only up to <code>i</code>, the index of
  // the outer loop.
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

              // For the right-hand side we use the fact that the shape
              // functions are only non-zero in one component (because our
              // elements are primitive).  Instead of multiplying the tensor
              // representing the dim+1 values of shape function i with the
              // whole right-hand side vector, we only look at the only
              // non-zero component. The Function
              // FiniteElement::system_to_component_index(i) will return
              // which component this shape function lives in (0=x velocity,
              // 1=y velocity, 2=pressure in 2d), which we use to pick out
              // the correct component of the right-hand side vector to
              // multiply with.

              const unsigned int component_i =
                fe.system_to_component_index(i).first;
              local_rhs(i) += fe_values.shape_value(i,q) *
                              rhs_values[q](component_i) *
                              fe_values.JxW(q);
            }
        }

      // Note that in the above computation of the local matrix contribution
      // we added the term <code> phi_p[i] * phi_p[j] </code>, yielding a
      // pressure mass matrix in the $(1,1)$ block of the matrix as
      // discussed in the introduction. That this term only ends up in the
      // $(1,1)$ block stems from the fact that both of the factors in
      // <code>phi_p[i] * phi_p[j]</code> are only non-zero when all the
      // other terms vanish (and the other way around).
      //
      // Note also that operator* is overloaded for symmetric tensors,
      // yielding the scalar product between the two tensors in the first
      // line of the local matrix contribution.

      // Before we can write the local data into the global matrix (and
      // simultaneously use the ConstraintMatrix object to apply Dirichlet
      // boundary conditions and eliminate hanging node constraints, as we
      // discussed in the introduction), we have to be careful about one
      // thing, though. We have only built half of the local matrix
      // because of symmetry, but we're going to save the full system matrix
      // in order to use the standard functions for solution. This is done
      // by flipping the indices in case we are pointing into the empty part
      // of the local matrix.
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=i+1; j<dofs_per_cell; ++j)
          local_matrix(i,j) = local_matrix(j,i);

      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global (local_matrix, local_rhs,
                                              local_dof_indices,
                                              system_matrix, system_rhs);
    }
}

template <int dim>
void StokesProblem<dim>::assemble_system_mg ()
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
	const unsigned int   velocity_dofs_per_cell   = velocity_fe.dofs_per_cell;

	const unsigned int   n_q_points      = quadrature_formula.size();

	FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
	Vector<double>       local_rhs (dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
	std::vector<types::global_dof_index> local_velocity_dof_indices (velocity_dofs_per_cell);

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
	endc = dof_handler.end(),
	cell_vel = velocity_dof_handler.begin_active(),
	endc_vel = velocity_dof_handler.end();
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
					local_matrix(i,j) += (symgrad_phi_u[i] * symgrad_phi_u[j]
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
		cell_vel->get_dof_indices (local_velocity_dof_indices);

		// TODO: THIS COULD BE USED FOR A TEST
		// TODO: system_to_Componenet index (for an index does it belogn to velocity or pressure) -(if belongs to velocity check it)
		// or DO componenet_to_system_index (system is i -- you can ask it waht componenet it belogns to_
		// loop in smaller array from 0 to size
		// look it up in other one (keep incremeneting index if system to componenet index gives you a pressure (twoo running indices)
		// one from 0 to 18 and other incremeented every cheack and gets incrememented (if system componenet index is preussure)
		// careful (returns a pair) - (1st element is component (0 1 or 2 <-- if 2 pressure))
		//		std::pair< unsigned int, unsigned int > temp;
		//		int j = 0;
		//		temp = fe.system_to_component_index(0);
		//		for (int i = 0; i < local_velocity_dof_indices.size(); i++)
		//		{
		//			temp = fe.system_to_component_index(j);
		//			while (temp.first == dim && j<local_dof_indices.size()-1) // GG: Why aren't you working? >.<
		//			{
		//				j++;
		//				temp = fe.system_to_component_index(j);
		//				//					std::cout<<"temp.first: " << temp.first<<std::endl;
		//			}
		//			//				std::cout<<"i: " << i<<std::endl;
		//			//				std::cout<<"j: " << j<<std::endl;
		//			//				std::cout<<"local_dof_indices[j] : " << local_dof_indices[j] <<std::endl;
		//			//				std::cout<<"local_velocity_dof_indices[i]: " << local_velocity_dof_indices[i]<<std::endl;
		//			Assert(local_dof_indices[j] == local_velocity_dof_indices[i], ExcMessage ("Big DoF and Velocity DoF not agreeing."));
		//			j++;
		//		}
		//		++cell_vel;

		constraints.distribute_local_to_global (local_matrix, local_rhs,
				local_dof_indices,
				system_matrix, system_rhs);
	}

	std::cout << "   Computing preconditioner..." << std::endl << std::flush;
}

template <int dim>
void StokesProblem<dim>::assemble_multigrid ()
{

	mg_matrices = 0; //RG: Reset multi-grid matrices

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

	typename DoFHandler<dim>::cell_iterator cell = velocity_dof_handler.begin(), // Goes over all cells (not just active)
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
					cell_matrix(i,j) += (symgrad_phi_u[i] //RG: Just assembling the laplace of u
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

		// The next step is again slightly more
		// obscure (but explained in the @ref
		// mg_paper): We need the remainder of
		// the operator that we just copied
		// into the <code>mg_matrices</code>
		// object, namely the part on the
		// interface between cells at the
		// current level and cells one level
		// coarser. This matrix exists in two
		// directions: for interior DoFs (index
		// $i$) of the current level to those
		// sitting on the interface (index
		// $j$), and the other way around. Of
		// course, since we have a symmetric
		// operator, one of these matrices is
		// the transpose of the other.
		//
		// The way we assemble these matrices
		// is as follows: since the are formed
		// from parts of the local
		// contributions, we first delete all
		// those parts of the local
		// contributions that we are not
		// interested in, namely all those
		// elements of the local matrix for
		// which not $i$ is an interface DoF
		// and $j$ is not. The result is one of
		// the two matrices that we are
		// interested in, and we then copy it
		// into the
		// <code>mg_interface_matrices</code>
		// object. The
		// <code>boundary_interface_constraints</code>
		// object at the same time makes sure
		// that we delete contributions from
		// all degrees of freedom that are not
		// only on the interface but also on
		// the external boundary of the domain.
		//
		// The last part to remember is how to
		// get the other matrix. Since it is
		// only the transpose, we will later
		// (in the <code>solve()</code>
		// function) be able to just pass the
		// transpose matrix where necessary.
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

// After the discussion in the introduction and the definition of the
// respective classes above, the implementation of the <code>solve</code>
// function is rather straight-forward and done in a similar way as in
// step-20. To start with, we need an object of the
// <code>InverseMatrix</code> class that represents the inverse of the
// matrix A. As described in the introduction, the inverse is generated with
// the help of an inner preconditioner of type
// <code>InnerPreconditioner::type</code>.
template <int dim>
void StokesProblem<dim>::solve ()
{

	  // Before we're going to solve this linear system, we generate a
	    // preconditioner for the velocity-velocity matrix, i.e.,
	    // <code>block(0,0)</code> in the system matrix. As mentioned above, this
	    // depends on the spatial dimension. Since the two classes described by
	    // the <code>InnerPreconditioner::type</code> typedef have the same
	    // interface, we do not have to do anything different whether we want to
	    // use a sparse direct solver or an ILU:
	    std::cout << "   Computing preconditioner..." << std::endl << std::flush;

	    // RG: extra?
	    A_preconditioner
	      = std_cxx11::shared_ptr<typename InnerPreconditioner<dim>::type>(new typename InnerPreconditioner<dim>::type());
        A_preconditioner->initialize (system_matrix.block(0,0),
                                  typename InnerPreconditioner<dim>::type::AdditionalData());

	const InverseMatrix<SparseMatrix<double>,
	typename InnerPreconditioner<dim>::type>
	A_inverse (system_matrix.block(0,0), *A_preconditioner);
	Vector<double> tmp (solution.block(0).size());

	// This is as in step-20. We generate the right hand side $B A^{-1} F - G$
	// for the Schur complement and an object that represents the respective
	// linear operation $B A^{-1} B^T$, now with a template parameter
	// indicating the preconditioner - in accordance with the definition of
	// the class.
	{
		Vector<double> schur_rhs (solution.block(1).size());
		A_inverse.vmult (tmp, system_rhs.block(0));
		system_matrix.block(1,0).vmult (schur_rhs, tmp);
		schur_rhs -= system_rhs.block(1);

		SchurComplement<typename InnerPreconditioner<dim>::type>
		schur_complement (system_matrix, A_inverse);

		// The usual control structures for the solver call are created...
		SolverControl solver_control (solution.block(1).size(),
				1e-6*schur_rhs.l2_norm());
		SolverCG<>    cg (solver_control);

		// Now to the preconditioner to the Schur complement. As explained in
		// the introduction, the preconditioning is done by a mass matrix in the
		// pressure variable.  It is stored in the $(1,1)$ block of the system
		// matrix (that is not used anywhere else but in preconditioning).
		//
		// Actually, the solver needs to have the preconditioner in the form
		// $P^{-1}$, so we need to create an inverse operation. Once again, we
		// use an object of the class <code>InverseMatrix</code>, which
		// implements the <code>vmult</code> operation that is needed by the
		// solver.  In this case, we have to invert the pressure mass matrix. As
		// it already turned out in earlier tutorial programs, the inversion of
		// a mass matrix is a rather cheap and straight-forward operation
		// (compared to, e.g., a Laplace matrix). The CG method with ILU
		// preconditioning converges in 5-10 steps, independently on the mesh
		// size.  This is precisely what we do here: We choose another ILU
		// preconditioner and take it along to the InverseMatrix object via the
		// corresponding template parameter.  A CG solver is then called within
		// the vmult operation of the inverse matrix.
		//
		// An alternative that is cheaper to build, but needs more iterations
		// afterwards, would be to choose a SSOR preconditioner with factor
		// 1.2. It needs about twice the number of iterations, but the costs for
		// its generation are almost negligible.
		SparseILU<double> preconditioner;
		preconditioner.initialize (system_matrix.block(1,1),
				SparseILU<double>::AdditionalData());

		InverseMatrix<SparseMatrix<double>,SparseILU<double> >
		m_inverse (system_matrix.block(1,1), preconditioner);

		// With the Schur complement and an efficient preconditioner at hand, we
		// can solve the respective equation for the pressure (i.e. block 0 in
		// the solution vector) in the usual way:
		cg.solve (schur_complement, solution.block(1), schur_rhs,
				m_inverse);

		// After this first solution step, the hanging node constraints have to
		// be distributed to the solution in order to achieve a consistent
		// pressure field.
		constraints.distribute (solution);

		std::cout << "  "
				<< solver_control.last_step()
				<< " outer CG Schur complement iterations for pressure"
				<< std::endl;
	}

	// As in step-20, we finally need to solve for the velocity equation where
	// we plug in the solution to the pressure equation. This involves only
	// objects we already know - so we simply multiply $p$ by $B^T$, subtract
	// the right hand side and multiply by the inverse of $A$. At the end, we
	// need to distribute the constraints from hanging nodes in order to
	// obtain a consistent flow field:
	{
		system_matrix.block(0,1).vmult (tmp, solution.block(1));
		tmp *= -1;
		tmp += system_rhs.block(0);

		A_inverse.vmult (solution.block(0), tmp);

		constraints.distribute (solution);
	}

}

// @sect4{StokesProblem::solve}

// After the discussion in the introduction and the definition of the
// respective classes above, the implementation of the <code>solve</code>
// function is rather straight-forward and done in a similar way as in
// step-20. To start with, we need an object of the
// <code>InverseMatrix</code> class that represents the inverse of the
// matrix A. As described in the introduction, the inverse is generated with
// the help of an inner preconditioner of type
// <code>InnerPreconditioner::type</code>.
template <int dim>
void StokesProblem<dim>::solve_block ()
{
	  // Before we're going to solve this linear system, we generate a
	    // preconditioner for the velocity-velocity matrix, i.e.,
	    // <code>block(0,0)</code> in the system matrix. As mentioned above, this
	    // depends on the spatial dimension. Since the two classes described by
	    // the <code>InnerPreconditioner::type</code> typedef have the same
	    // interface, we do not have to do anything different whether we want to
	    // use a sparse direct solver or an ILU:
	    std::cout << "   Computing preconditioner..." << std::endl << std::flush;

	    // RG: extra?
	    A_preconditioner
	      = std_cxx11::shared_ptr<typename InnerPreconditioner<dim>::type>(new typename InnerPreconditioner<dim>::type());
      A_preconditioner->initialize (system_matrix.block(0,0),
                                typename InnerPreconditioner<dim>::type::AdditionalData());

	SparseMatrix<double> pressure_mass_matrix;
	pressure_mass_matrix.reinit(sparsity_pattern.block(1,1));
	pressure_mass_matrix.copy_from(system_matrix.block(1,1));
	system_matrix.block(1,1) = 0;

	SparseILU<double> pmass_preconditioner;
	pmass_preconditioner.initialize (pressure_mass_matrix,
			SparseILU<double>::AdditionalData());

	InverseMatrix<SparseMatrix<double>,SparseILU<double> >
	m_inverse (pressure_mass_matrix, pmass_preconditioner);

	BlockSchurPreconditioner<typename InnerPreconditioner<dim>::type,
							SparseILU<double> >
	  preconditioner (system_matrix, m_inverse, *A_preconditioner);

	SolverControl solver_control (system_matrix.m(),
					1e-6*system_rhs.l2_norm());

	GrowingVectorMemory<BlockVector<double> > vector_memory;
	SolverGMRES<BlockVector<double> >::AdditionalData gmres_data;
	gmres_data.max_n_tmp_vectors = 100;

	SolverGMRES<BlockVector<double> > gmres(solver_control, vector_memory,
											gmres_data);

	computing_timer.enter_subsection ("Solve - GMRES");
	gmres.solve(system_matrix, solution, system_rhs,
				preconditioner);
	computing_timer.leave_subsection ();

	constraints.distribute (solution);

	std::cout << " "
	<< solver_control.last_step()
	<< " block GMRES iterations";
}

// This is the other function that is
// significantly different in support of the
// multigrid solver (or, in fact, the
// preconditioner for which we use the
// multigrid method).
//
// Let us start out by setting up two of the
// components of multilevel methods: transfer
// operators between levels, and a solver on
// the coarsest level. In finite element
// methods, the transfer operators are
// derived from the finite element function
// spaces involved and can often be computed
// in a generic way independent of the
// problem under consideration. In that case,
// we can use the MGTransferPrebuilt class
// that, given the constraints on the global
// level and an DoFHandler object computes
// the matrices corresponding to these
// transfer operators.
//
// The second part of the following lines
// deals with the coarse grid solver. Since
// our coarse grid is very coarse indeed, we
// decide for a direct solver
template <int dim>
void StokesProblem<dim>::solve_mg ()
{
	//	computing_timer.enter_subsection ("Solve - MG Stuff");

	// Transfer operators between levels
	MGTransferPrebuilt<Vector<double> > mg_transfer(hanging_node_constraints, mg_constrained_dofs);
	mg_transfer.build_matrices(velocity_dof_handler);

	// Coarse grid solver
	FullMatrix<double> coarse_matrix;
	coarse_matrix.copy_from (mg_matrices[0]);
	MGCoarseGridHouseholder<> coarse_grid_solver;
	coarse_grid_solver.initialize (coarse_matrix);

	// The next component of a multilevel solver or preconditioner is that we need a smoother on each level.
	// A common choice for this is to use the application of a relaxation method (such as the SOR, Jacobi or
	// Richardson method) or a small number of iterations of a solver method (such as CG or GMRES).
	// The mg::SmootherRelaxation and MGSmootherPrecondition classes provide support for these two kinds
	// of smoothers. Here, we opt for the application of a single SOR iteration. To this end, we define
	// an appropriate typedef and then setup a smoother object.

	// The last step is to initialize the smoother object with our level matrices and to set some smoothing
	// parameters. The initialize() function can optionally take additional arguments that will be passed to
	// the smoother object on each level. In the current case for the SOR smoother, this could, for example,
	// include a relaxation parameter. However, we here leave these at their default values. The call to set_steps()
	// indicates that we will use two pre- and two post-smoothing steps on each level; to use a variable number of
	// smoother steps on different levels, more options can be set in the constructor call to the mg_smoother object.

	// The last step results from the fact that we use the SOR method as a smoother - which is not symmetric -
	// but we use the conjugate gradient iteration (which requires a symmetric preconditioner) below, we need to
	// let the multilevel preconditioner make sure that we get a symmetric operator even for nonsymmetric smoothers:
	typedef PreconditionSOR<SparseMatrix<double> > Smoother;
	mg::SmootherRelaxation<Smoother, Vector<double> > mg_smoother;
	mg_smoother.initialize(mg_matrices);
	mg_smoother.set_steps(2);
	mg_smoother.set_symmetric(true); // MG as precond for linear (CG) and expects symmetric then need symmetric

	// The next preparatory step is that we must wrap our level and interface matrices in an object having the
	// required multiplication functions. We will create two objects for the interface objects going from coarse
	// to fine and the other way around; the multigrid algorithm will later use the transpose operator for the latter
	// operation, allowing us to initialize both up and down versions of the operator with the matrices we already built:
	mg::Matrix<Vector<double> > mg_matrix(mg_matrices);
	mg::Matrix<Vector<double> > mg_interface_up(mg_interface_matrices);   // This SHOULD be fine!
	mg::Matrix<Vector<double> > mg_interface_down(mg_interface_matrices);

	// Now, we are ready to set up the V-cycle operator and the multilevel preconditioner.
	Multigrid<Vector<double> > mg(velocity_dof_handler,
			mg_matrix,
			coarse_grid_solver,
			mg_transfer,
			mg_smoother,
			mg_smoother);
	mg.set_edge_matrices(mg_interface_down, mg_interface_up);
	//mg.set_debug(5);
	PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double> > >
	A_Multigrid(velocity_dof_handler, mg, mg_transfer);  // RG: Make A_Multigrid then use like the A_inverse below?
	// RG: velocity_dof_handler takes care of fact want block(0,0) ??

	//***
	// To start with, we need an object of the InverseMatrix class that represents the inverse of the matrix A. As described in the introduction,
	// the inverse is generated with the help of an inner preconditioner of type InnerPreconditioner::type.

	//	computing_timer.leave_subsection();
	//	computing_timer.enter_subsection ("Solve - For real");

	const InverseMatrix<SparseMatrix<double>,
	PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double> > > >
	A_inverse (system_matrix.block(0,0), A_Multigrid);

	Vector<double> tmp (solution.block(0).size());

	{
		Vector<double> schur_rhs (solution.block(1).size()); // We generate the right hand side BA−1F−G for the Schur complement and an object
		// that represents the respective linear operation BA−1BT, now with a template parameter
		// indicating the preconditioner - in accordance with the definition of the class.
		A_inverse.vmult (tmp, system_rhs.block(0));
		system_matrix.block(1,0).vmult (schur_rhs, tmp);
		schur_rhs -= system_rhs.block(1);

		SchurComplement<PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double> > >>
		schur_complement (system_matrix, A_inverse);

		SolverControl solver_control (solution.block(1).size(),
				1e-6*schur_rhs.l2_norm());
		SolverCG<>    cg (solver_control);

		SparseILU<double> preconditioner; // RG: Also should be MG?
		preconditioner.initialize (system_matrix.block(1,1),
				SparseILU<double>::AdditionalData());

		InverseMatrix<SparseMatrix<double>,SparseILU<double> >
		m_inverse (system_matrix.block(1,1), preconditioner);

		//With the Schur complement and an efficient preconditioner at hand, we can solve the respective equation for the pressure (i.e. block 0 in the solution vector) in the usual way:
		cg.solve (schur_complement, solution.block(1), schur_rhs, // schru_complement object that looks like matrix (how to multiply vector with it)
				m_inverse); //storign result in solution.block(1)
		// Note: don't need to do any transfer yourself in Schur Complement appraoch (look in vmult)

		constraints.distribute (solution);

		//		std::cout << "  "
		//				<< solver_control.last_step()
		//				<< " outer CG Schur complement iterations for pressure"
		//				<< std::endl;
	}

	{
		// We finally need to solve for the velocity equation where we plug in the solution to the pressure equation. This involves
		// only objects we already know - so we simply multiply p by BT, subtract the right hand side and multiply by the inverse of A.
		// At the end, we need to distribute the constraints from hanging nodes in order to obtain a consistent flow field

		system_matrix.block(0,1).vmult (tmp, solution.block(1));
		tmp *= -1;
		tmp += system_rhs.block(0);

		A_inverse.vmult (solution.block(0), tmp);

		constraints.distribute (solution);
	}
	//	computing_timer.leave_subsection();
}

template <int dim>
void StokesProblem<dim>::solve_block_mg()
{
	//	computing_timer.enter_subsection ("Solve - MG Stuff");
	// Transfer operators between levels
	MGTransferPrebuilt<Vector<double> > mg_transfer(hanging_node_constraints, mg_constrained_dofs);
	mg_transfer.build_matrices(velocity_dof_handler);

	// Coarse grid solver
	FullMatrix<double> coarse_matrix;
	coarse_matrix.copy_from (mg_matrices[0]);
	MGCoarseGridHouseholder<> coarse_grid_solver;
	coarse_grid_solver.initialize (coarse_matrix);

	// The next component of a multilevel solver or preconditioner is that we need a smoother on each level.
	// A common choice for this is to use the application of a relaxation method (such as the SOR, Jacobi or
	// Richardson method) or a small number of iterations of a solver method (such as CG or GMRES).
	// The mg::SmootherRelaxation and MGSmootherPrecondition classes provide support for these two kinds
	// of smoothers. Here, we opt for the application of a single SOR iteration. To this end, we define
	// an appropriate typedef and then setup a smoother object.

	// The last step is to initialize the smoother object with our level matrices and to set some smoothing
	// parameters. The initialize() function can optionally take additional arguments that will be passed to
	// the smoother object on each level. In the current case for the SOR smoother, this could, for example,
	// include a relaxation parameter. However, we here leave these at their default values. The call to set_steps()
	// indicates that we will use two pre- and two post-smoothing steps on each level; to use a variable number of
	// smoother steps on different levels, more options can be set in the constructor call to the mg_smoother object.

	// The last step results from the fact that we use the SOR method as a smoother - which is not symmetric -
	// but we use the conjugate gradient iteration (which requires a symmetric preconditioner) below, we need to
	// let the multilevel preconditioner make sure that we get a symmetric operator even for nonsymmetric smoothers:
	typedef PreconditionSOR<SparseMatrix<double> > Smoother;
	mg::SmootherRelaxation<Smoother, Vector<double> > mg_smoother;
	mg_smoother.initialize(mg_matrices);
	mg_smoother.set_steps(2);
	mg_smoother.set_symmetric(true); // MG as precond for linear (CG) and expects symmetric then need symmetric

	// The next preparatory step is that we must wrap our level and interface matrices in an object having the
	// required multiplication functions. We will create two objects for the interface objects going from coarse
	// to fine and the other way around; the multigrid algorithm will later use the transpose operator for the latter
	// operation, allowing us to initialize both up and down versions of the operator with the matrices we already built:
	mg::Matrix<Vector<double> > mg_matrix(mg_matrices);
	mg::Matrix<Vector<double> > mg_interface_up(mg_interface_matrices);   // This SHOULD be fine!
	mg::Matrix<Vector<double> > mg_interface_down(mg_interface_matrices);

	// Now, we are ready to set up the V-cycle operator and the multilevel preconditioner.
	Multigrid<Vector<double> > mg(velocity_dof_handler,
			mg_matrix,
			coarse_grid_solver,
			mg_transfer,
			mg_smoother,
			mg_smoother);
	mg.set_edge_matrices(mg_interface_down, mg_interface_up);
	PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double> > >
	A_Multigrid(velocity_dof_handler, mg, mg_transfer);  // RG: Make A_Multigrid then use like the A_inverse below?
	// RG: velocity_dof_handler takes care of fact want block(0,0) ??

	//	computing_timer.leave_subsection();
	//	computing_timer.enter_subsection ("Solve - For real");

	SparseMatrix<double> pressure_mass_matrix;
	pressure_mass_matrix.reinit(sparsity_pattern.block(1,1));
	pressure_mass_matrix.copy_from(system_matrix.block(1,1));
	system_matrix.block(1,1) = 0;
	SparseILU<double> pmass_preconditioner;
	pmass_preconditioner.initialize (pressure_mass_matrix,
			SparseILU<double>::AdditionalData());
//	InverseMatrix<SparseMatrix<double>,SparseILU<double> >    //goes away
//	m_inverse (pressure_mass_matrix, pmass_preconditioner);

//	BlockSchurPreconditioner<PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double> > >,
//	SparseILU<double> >
//	preconditioner (system_matrix, m_inverse, A_Multigrid);

//	SolverControl solver_control (system_matrix.m(),
//			1e-6*system_rhs.l2_norm());
//	GrowingVectorMemory<BlockVector<double> > vector_memory;
//	SolverGMRES<BlockVector<double> >::AdditionalData gmres_data;
//	gmres_data.max_n_tmp_vectors = 100;
//	SolverGMRES<BlockVector<double> > gmres(solver_control, vector_memory,
//			gmres_data);
//	computing_timer.enter_subsection ("Solve - GMRES");
//	gmres.solve(system_matrix, solution, system_rhs,
//			preconditioner);
//	computing_timer.leave_subsection ();
//	constraints.distribute (solution);
//	std::cout << " "
//			<< solver_control.last_step()
//			<< " block GMRES iterations";

	// TIMO: New Way
    SolverControl solver_control (system_matrix.m(),
           			1e-6*system_rhs.l2_norm());
    GrowingVectorMemory<BlockVector<double> > vector_memory;
    SolverGMRES<BlockVector<double> >::AdditionalData gmres_data;
    gmres_data.max_n_tmp_vectors = 100;
    SolverGMRES<BlockVector<double> > gmres(solver_control, vector_memory,
    			gmres_data);

	bool use_cheap = true;
    unsigned int its_A = 0, its_S = 0;
    // if this cheaper solver is not desired, then simply short-cut
    // the attempt at solving with the cheaper preconditioner
    if (use_cheap == true)
    {
        // give it a try with a preconditioner that consists
        // of only a single V-cycle
        const NewBlockSchurPreconditioner<PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double> > >,
              SparseILU<double>>
              preconditioner (system_matrix, pressure_mass_matrix,  //TIMO: Do I have this in mine? I believe we just use system_matrix in this code
            		          pmass_preconditioner, A_Multigrid, //*Mp_preconditioner, *Amg_preconditioner, //NO--^
                              false);

        gmres.solve (system_matrix, //TIMO: oh no.
                      solution, //TIMO: distributed_stokes_solution
                      system_rhs, //TIMO: distributed_stokes_rhs
                      preconditioner);

        its_A += preconditioner.n_iterations_A();
        its_S += preconditioner.n_iterations_S();
    }

    else // TIMO: Do I even need this? .. I have a bool that can go in preconditioner.. everything else is the same  NO! (pass flag)
         {
           // this additional entry serves as a marker between cheap and expensive Stokes solver

           const NewBlockSchurPreconditioner<PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double> > >,
                 SparseILU<double>>
                 preconditioner (system_matrix, pressure_mass_matrix,
                		         pmass_preconditioner, A_Multigrid,
                                 true);

           gmres.solve (system_matrix,
                         solution,
                         system_rhs,
                         preconditioner);

           its_A += preconditioner.n_iterations_A();
           its_S += preconditioner.n_iterations_S();
         }
	//	computing_timer.leave_subsection();
}


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

	std::cout << " "
					<< count_mm
					<< " mass matrix CG iterations"
					<< std::endl;

}



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
//
// We start off with a rectangle of size $4 \times 1$ (in 2d) or $4 \times 1
// \times 1$ (in 3d), placed in $R^2/R^3$ as $(-2,2)\times(-1,0)$ or
// $(-2,2)\times(0,1)\times(-1,0)$, respectively. It is natural to start
// with equal mesh size in each direction, so we subdivide the initial
// rectangle four times in the first coordinate direction. To limit the
// scope of the variables involved in the creation of the mesh to the range
// where we actually need them, we put the entire block between a pair of
// braces:
template <int dim>
void StokesProblem<dim>::run ()
{
	{
		std::vector<unsigned int> subdivisions (dim, 1);
		subdivisions[0] = 4;

		const Point<dim> bottom_left = (dim == 2 ?
				Point<dim>(-2,-1) :
				Point<dim>(-2,0,-1));
		const Point<dim> top_right   = (dim == 2 ?
				Point<dim>(2,0) :
				Point<dim>(2,1,0));

		GridGenerator::subdivided_hyper_rectangle (triangulation,
				subdivisions,
				bottom_left,
				top_right);
	}

	// A boundary indicator of 1 is set to all boundaries that are subject to
	// Dirichlet boundary conditions, i.e.  to faces that are located at 0 in
	// the last coordinate direction. See the example description above for
	// details.
	for (typename Triangulation<dim>::active_cell_iterator
			cell = triangulation.begin_active();
			cell != triangulation.end(); ++cell)
		for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
			if (cell->face(f)->center()[dim-1] == 0)
				cell->face(f)->set_all_boundary_ids(1);


	// We then apply an initial refinement before solving for the first
	// time. In 3D, there are going to be more degrees of freedom, so we
	// refine less there:
	triangulation.refine_global (6-dim);
	// As first seen in step-6, we cycle over the different refinement levels
	// and refine (except for the first cycle), setup the degrees of freedom
	// and matrices, assemble, solve and create output:
	for (unsigned int refinement_cycle = 0; refinement_cycle<3;
			++refinement_cycle)
	{
		std::cout << "Refinement cycle " << refinement_cycle << std::endl;

		if (refinement_cycle > 0)
			refine_mesh ();

		computing_timer.enter_subsection ("ILU - Setup");
		setup_dofs();
		computing_timer.leave_subsection();
		std::cout << "   Assembling..." << std::endl << std::flush;
		computing_timer.enter_subsection ("ILU - Assemble");
		assemble_system ();
		computing_timer.leave_subsection();
		std::cout << "   Solving..." << std::flush;
		computing_timer.enter_subsection ("ILU - Solve");
		solve_block ();
		computing_timer.leave_subsection();

		computing_timer.print_summary ();
		computing_timer.reset ();
		output_results (refinement_cycle);

		std::cout << std::endl;

		computing_timer.enter_subsection ("MG - Setup");
		setup_dofs_mg();
		computing_timer.leave_subsection();
		std::cout << "   Assembling..." << std::endl << std::flush;
		computing_timer.enter_subsection ("MG - Assemble");
		assemble_system_mg ();
		computing_timer.leave_subsection();
		std::cout << "   Assembling Multigrid..." << std::endl << std::flush;
		computing_timer.enter_subsection ("MG - Assemble Multigrid");
		assemble_multigrid ();
		computing_timer.leave_subsection();
		std::cout << "   Solving..." << std::flush;
		computing_timer.enter_subsection ("MG - Solve");
		solve_block_mg ();
		computing_timer.leave_subsection();

		computing_timer.print_summary ();
		computing_timer.reset ();
		output_results (refinement_cycle);
	}
}
}


int main ()
{
	try
	{
		using namespace dealii;
		using namespace Step22;

		deallog.depth_console (0); //0

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
