# Matrix-free elasticity example

Turn `tests/matrix_free_kokkos/elasticity_02.cc` into a user-facing example, using the structure and timing style of `step-104.cc`.

## Implemented

1. Removed the test harness, `initlog()`, `deallog`, and fixed test cases. The example runs a 3D problem, reports mesh and DoF counts, solves, and computes the L2 error.
2. Uses the test's exact displacement, `u_i = product_d x_d(1-x_d)`, with homogeneous boundary conditions.
3. Supports conventional `MatrixFree`/`FEEvaluation` and `Portable::MatrixFree`/`Portable::FEEvaluation`. Select the backend with the first command-line argument: `matrixfree` or `portable` (default).
4. Reports setup/preconditioner, solve, postprocessing, and total wall times, plus CG iterations.
5. Adds reusable `Coefficient<dim>` as a scalar backend-storage adapter. The example owns the DGQ0 source fields and passes one field to each instance. Each instance stores exactly one active backend representation: a Kokkos view for Portable::MatrixFree or cell-batch data initialized with `MatrixFree::initialize_cell_data_vector()` and `FEEvaluation::set_cell_data()` for MatrixFree. Packing reads directly from the source field without an intermediate cell-value cache.
6. Uses separate lambda and mu `Coefficient<dim>` instances. The example evaluates `lambda(x)=1+x` and `mu(x)=1+y+z` into separate DGQ0 fields. The second argument selects the GMG policy: `evaluate` (default, evaluate the expressions on each level) or `transfer` (transfer each finest DGQ0 field with `interpolate_to_mg()`). Both policies are supported by both backends.
7. Uses a discrete manufactured load, `b = A I_h(u_exact)`, so the variable coefficient strong-form divergence does not need to be derived by hand. The reported L2 error is against the continuous exact solution.
8. Writes parallel VTU/PVTU output for the displacement vector and the cellwise `lambda` and `mu` fields. The file prefix records the backend and coefficient policy.

The `Coefficient<dim>` class contains no elasticity-specific logic and owns no source FE field. The example owns the distributed fields, so MPI ghost exchange and multigrid transfer use standard deal.II vectors. The adapter packs scalar one-DoF-per-cell fields directly into the selected backend representation. Future quadrature-point or general FE field data will need an extension of this adapter's input/layout interface.

## Verification

Configured with:

```text
cmake -S examples/step-elasticity -B build/examples/step-elasticity -DDEAL_II_DIR=/ssd/deal-git8/build
```

With the Kokkos-only `build/`, built with `ninja -C build/examples/step-elasticity -j 8` and ran all four backend/policy combinations. Each converged in 6 CG iterations and reported the same L2 error, `1.05947e-4`.

The example selects a distributed triangulation when both MPI and p4est are enabled, and retains a serial triangulation otherwise. With the MPI+p4est build, all four backend/policy combinations ran successfully on two OpenMPI ranks, converged in 6 iterations, and reported the same L2 error (`1.05947e-4`). Use the OpenMPI launcher matching the build (`/usr/bin/mpiexec` here); the shell's default `mpiexec` is from a different MPI implementation.

## Follow-up

Add quadrature-point coefficient values and a path that evaluates an existing FE coefficient field at quadrature points. Keep device kernels free of host callbacks and measure GPU performance as those coefficient sources are added.
