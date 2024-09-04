## ------------------------------------------------------------------------
##
## SPDX-License-Identifier: LGPL-2.1-or-later
## Copyright (C) 2021 - 2024 by the deal.II authors
##
## This file is part of the deal.II library.
##
## Part of the source code is dual licensed under Apache-2.0 WITH
## LLVM-exception OR LGPL-2.1-or-later. Detailed license information
## governing the source code and code contributions can be found in
## LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
##
## ------------------------------------------------------------------------

#
# Try to find the Kokkos library
#
# This module exports
#
#   KOKKOS_INCLUDE_DIRS
#   KOKKOS_INTERFACE_LINK_FLAGS
#

set(KOKKOS_DIR "" CACHE PATH "An optional hint to a Kokkos installation")
set_if_empty(KOKKOS_DIR "$ENV{KOKKOS_DIR}")

# silence a warning when including FindKOKKOS.cmake
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Kokkos 3.7.0 QUIET
  HINTS ${KOKKOS_DIR} ${Kokkos_DIR} $ENV{Kokkos_DIR} ${TRILINOS_DIR} ${PETSC_DIR}
  )

set(KOKKOS_FOUND ${Kokkos_FOUND})

set(_target Kokkos::kokkos)
process_feature(KOKKOS
  TARGETS REQUIRED _target
  )

if(DEAL_II_TRILINOS_WITH_KOKKOS AND NOT Kokkos_FOUND)
  message(FATAL_ERROR "\n"
    "Trilinos is configured with Kokkos, but we could not find the "
    "Kokkos installation. Please provide the location via KOKKOS_DIR.")
endif()

if(DEAL_II_PETSC_WITH_KOKKOS AND NOT Kokkos_FOUND)
  message(FATAL_ERROR "\n"
    "PETSc is configured with Kokkos, but we could not find the "
    "Kokkos installation. Please provide the location via KOKKOS_DIR.")
endif()

# TODO: We should verify that the installation used by PETSc/Trilinos is
# the same one we found.

# GPU support
if(Kokkos_FOUND)
  if(Kokkos_ENABLE_CUDA)
    # We need to disable SIMD vectorization for CUDA device code.
    # Otherwise, nvcc compilers from version 9 on will emit an error message like:
    # "[...] contains a vector, which is not supported in device code". We
    # would like to set the variable in check_01_cpu_feature but at that point
    # we don't know if CUDA support is enabled in Kokkos
    set(DEAL_II_VECTORIZATION_WIDTH_IN_BITS 0)

    # Require lambda support and expt-relaxed-constexpr for Cuda
    # so that we can use std::array and other interfaces with
    # __host__ constexpr functions in device code
    KOKKOS_CHECK(OPTIONS CUDA_LAMBDA CUDA_CONSTEXPR)

    # Disable a bunch of annoying warnings generated by boost, template code,
    # and in other random places:
    #
    # integer conversion resulted in a change of sign:
    enable_if_supported(DEAL_II_CXX_FLAGS "-Xcudafe --diag_suppress=68")
    # loop is not reachable:
    enable_if_supported(DEAL_II_CXX_FLAGS "-Xcudafe --diag_suppress=128")
    # warning #177-D: variable "n_max_face_orientations" was declared but never referenced
    enable_if_supported(DEAL_II_CXX_FLAGS "-Xcudafe --diag_suppress=177")
    # warning #186-D: pointless comparison of unsigned integer with zero
    enable_if_supported(DEAL_II_CXX_FLAGS "-Xcudafe --diag_suppress=186")
    # warning #191-D: type qualifier is meaningless on cast type
    enable_if_supported(DEAL_II_CXX_FLAGS "-Xcudafe --diag_suppress=191")
    # warning #284-D: NULL reference is not allowed
    enable_if_supported(DEAL_II_CXX_FLAGS "-Xcudafe --diag_suppress=284")
    # variable "i" was set but never used:
    enable_if_supported(DEAL_II_CXX_FLAGS "-Xcudafe --diag_suppress=550")
    # warning #940-D: missing return statement at end of non-void function
    enable_if_supported(DEAL_II_CXX_FLAGS "-Xcudafe --diag_suppress=940")
  endif()
endif()
