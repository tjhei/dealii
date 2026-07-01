# AGENTS.md

## Cursor Cloud specific instructions

deal.II is a large C++17 finite-element library (not an application). You develop
it by configuring with CMake and compiling `libdeal_II`. General build/run docs
live in `README.md`, `doc/`, and the CI recipes in `.github/workflows/linux.yml`.
The notes below only capture non-obvious, durable caveats for this environment.

### Compiler gotcha (important)
The default `c++`/`cc` alternative on this VM points to **Clang 18, which fails to
link (`cannot find -lstdc++`)**. Always configure with GCC explicitly:

```
-D CMAKE_C_COMPILER=gcc -D CMAKE_CXX_COMPILER=g++
```

### Configure & build (development = Debug)
The dependencies are installed by the startup update script (compiler toolchain +
OpenMPI). Boost, Kokkos, muParser, and Taskflow are used from the bundled copies,
so no extra libraries are needed for this configuration.

```
cd /workspace
mkdir -p build && cd build
cmake -D CMAKE_BUILD_TYPE=Debug -D DEAL_II_WITH_MPI=ON \
      -D CMAKE_C_COMPILER=gcc -D CMAKE_CXX_COMPILER=g++ ..
make -j$(nproc)          # ~15 min on 4 cores; produces lib/libdeal_II.g.so
```

- `CMAKE_BUILD_TYPE=Debug` builds only the debug library (`libdeal_II.g.so`).
  Use `DebugRelease` to build both debug + release (roughly doubles build time).
- Reconfiguring existing `build/` after changing feature flags (e.g. toggling
  `DEAL_II_WITH_MPI`) triggers a full rebuild; a clean `build/` is safest.
- `build/` and `contrib/utilities/programs/` are gitignored.

### Tests
The full testsuite is enormous. For a fast sanity check use the quick tests
(includes an MPI `mpirun=2` test):

```
cd build
make setup_tests_quick_tests
ctest -R quick
```

MPI tests run as root here, so export these before `ctest`/`mpirun` to avoid a
hard failure and silence the fabric warning:

```
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_btl_base_warn_component_unused=0
```

### Lint (indentation / formatting)
Requires **exactly clang-format 16.0.6**, which is not the system Clang. Download
the vendored binary once (persists in the snapshot, installs into the gitignored
`contrib/utilities/programs/`):

```
./contrib/utilities/download_clang_format
make -C build indent            # or: REPORT_ONLY=true make -C build indent
```

### Running a tutorial step against the build tree
Point the example at the (uninstalled) build directory via `DEAL_II_DIR`:

```
cmake -D DEAL_II_DIR=/workspace/build -D CMAKE_C_COMPILER=gcc \
      -D CMAKE_CXX_COMPILER=g++ /workspace/examples/step-3
make && ./step-3                # writes solution.vtk
```
