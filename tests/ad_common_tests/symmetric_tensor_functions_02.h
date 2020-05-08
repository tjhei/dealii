// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------


// Header file:
// Test to check that tensor functions both compile and produce the right
// result when differentiated using the various auto-differentiable number
// types: Tensor inverse

#include <deal.II/base/symmetric_tensor.h>

#include <deal.II/differentiation/ad.h>

#include <iostream>

#include "../tests.h"

namespace AD = dealii::Differentiation::AD;

template <int dim, typename NumberType>
struct FunctionsTestSymmetricTensor
{
  static SymmetricTensor<4, dim, NumberType>
  dt_inv_dt(const SymmetricTensor<2, dim, NumberType> &t_inv)
  {
    // https://en.wikiversity.org/wiki/Introduction_to_Elasticity/Tensors#Derivative_of_the_inverse_of_a_tensor
    SymmetricTensor<4, dim, NumberType> dt_inv_dt;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = i; j < dim; ++j)
        for (unsigned int k = 0; k < dim; ++k)
          for (unsigned int l = k; l < dim; ++l)
            dt_inv_dt[i][j][k][l] =
              -0.5 * (t_inv[i][k] * t_inv[j][l] + t_inv[i][l] * t_inv[j][k]);
    return dt_inv_dt;
  }

  static NumberType
  psi(const SymmetricTensor<2, dim, NumberType> &t)
  {
    // Previously, the invert function would hang for nested Sacado::Fad::DFad
    const SymmetricTensor<2, dim, NumberType> t_inv = invert(t);
    const SymmetricTensor<2, dim, NumberType> I =
      unit_symmetric_tensor<dim, NumberType>();
    return 3.0 * scalar_product(t_inv, I);
  }

  static SymmetricTensor<2, dim, NumberType>
  dpsi_dt(const SymmetricTensor<2, dim, NumberType> &t)
  {
    const SymmetricTensor<2, dim, NumberType> t_inv = invert(t);
    const SymmetricTensor<4, dim, NumberType> dt_inv_dt =
      FunctionsTestSymmetricTensor::dt_inv_dt(t_inv);
    const SymmetricTensor<2, dim, NumberType> I =
      unit_symmetric_tensor<dim, NumberType>();
    return 3.0 * (I * dt_inv_dt);
  }

  static Tensor<4, dim, NumberType>
  d2psi_dt_dt(const SymmetricTensor<2, dim, NumberType> &t)
  {
    const SymmetricTensor<2, dim, NumberType> t_inv = invert(t);
    const SymmetricTensor<4, dim, NumberType> dt_inv_dt =
      FunctionsTestSymmetricTensor::dt_inv_dt(t_inv);
    const SymmetricTensor<2, dim, NumberType> I =
      unit_symmetric_tensor<dim, NumberType>();

    SymmetricTensor<4, dim, NumberType> d2psi_dt_dt;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = i; j < dim; ++j)
        for (unsigned int m = 0; m < dim; ++m)
          for (unsigned int n = m; n < dim; ++n)
            for (unsigned int k = 0; k < dim; ++k)
              for (unsigned int l = k; l < dim; ++l)
                d2psi_dt_dt[i][j][m][n] +=
                  -3.0 * (0.5 * I[k][l] *
                          (dt_inv_dt[i][k][m][n] * t_inv[j][l] +
                           dt_inv_dt[i][l][m][n] * t_inv[j][k] +
                           t_inv[i][k] * dt_inv_dt[j][l][m][n] +
                           t_inv[i][l] * dt_inv_dt[j][k][m][n]));

    return d2psi_dt_dt;
  }
};

template <int dim, typename number_t, enum AD::NumberTypes ad_type_code>
void
test_symmetric_tensor()
{
  typedef AD::ScalarFunction<dim, ad_type_code, number_t> ADHelper;
  typedef typename ADHelper::ad_type                      ADNumberType;
  typedef typename ADHelper::scalar_type                  ScalarNumberType;

  std::cout << "*** dim = " << Utilities::to_string(dim) << ", "
            << "Type code: " << static_cast<int>(ad_type_code) << std::endl;

  // Tensor<2,dim,ADNumberType> grad_u;
  // const Tensor<2,dim,ADNumberType> F =
  // Physics::Elasticity::Kinematics::F(grad_u);

  // Values computed from the AD energy function
  ScalarNumberType             psi;
  Vector<ScalarNumberType>     Dpsi;
  FullMatrix<ScalarNumberType> D2psi;

  // Function and its derivatives
  typedef FunctionsTestSymmetricTensor<dim, ADNumberType> func_ad;

  // Setup the variable components and choose a value at which to
  // evaluate the tape
  const FEValuesExtractors::SymmetricTensor<2> t_dof(0);
  const unsigned int                           n_AD_components =
    SymmetricTensor<2, dim>::n_independent_components;
  ADHelper ad_helper(n_AD_components);
  ad_helper.set_tape_buffer_sizes(); // Increase the buffer size from the
                                     // default values

  SymmetricTensor<2, dim, ScalarNumberType> t =
    unit_symmetric_tensor<dim, ScalarNumberType>();
  for (unsigned int i = 0; i < t.n_independent_components; ++i)
    t[t.unrolled_to_component_indices(i)] += 0.12 * (i + 0.02);

  const int  tape_no = 1;
  const bool is_recording =
    ad_helper.start_recording_operations(tape_no /*material_id*/,
                                         true /*overwrite_tape*/,
                                         true /*keep*/);
  if (is_recording == true)
    {
      ad_helper.register_independent_variable(t, t_dof);

      const SymmetricTensor<2, dim, ADNumberType> t_ad =
        ad_helper.get_sensitive_variables(t_dof);

      const ADNumberType psi(func_ad::psi(t_ad));

      ad_helper.register_dependent_variable(psi);
      ad_helper.stop_recording_operations(false /*write_tapes_to_file*/);

      std::cout << "Recorded data..." << std::endl;
      std::cout << "independent variable values: " << std::flush;
      ad_helper.print_values(std::cout);
      std::cout << "t_ad: " << t_ad << std::endl;
      std::cout << "psi: " << psi << std::endl;
      std::cout << std::endl;
    }
  else
    {
      Assert(is_recording == true, ExcInternalError());
    }

  // Do some work :-)
  // Set a new evaluation point
  if (AD::ADNumberTraits<ADNumberType>::is_taped == true)
    {
      std::cout
        << "Using tape with different values for independent variables..."
        << std::endl;
      ad_helper.activate_recorded_tape(tape_no);
      t *= 1.15;
      ad_helper.set_independent_variable(t, t_dof);

      std::cout << "independent variable values: " << std::flush;
      ad_helper.print_values(std::cout);
    }

  // Compute the function value, gradient and hessian for the new evaluation
  // point
  psi = ad_helper.compute_value();
  ad_helper.compute_gradient(Dpsi);
  if (AD::ADNumberTraits<ADNumberType>::n_supported_derivative_levels >= 2)
    {
      ad_helper.compute_hessian(D2psi);
    }

  // Output the full stored function, gradient vector and hessian matrix
  std::cout << "psi: " << psi << std::endl;
  std::cout << "Dpsi: \n";
  Dpsi.print(std::cout);
  if (AD::ADNumberTraits<ADNumberType>::n_supported_derivative_levels >= 2)
    {
      std::cout << "D2psi: \n";
      D2psi.print_formatted(std::cout, 3, true, 0, "0.0");
    }

  // Extract components of the solution
  const SymmetricTensor<2, dim, ScalarNumberType> dpsi_dt =
    ad_helper.extract_gradient_component(Dpsi, t_dof);

  // Verify the result
  typedef FunctionsTestSymmetricTensor<dim, ScalarNumberType> func;
  static const ScalarNumberType                               tol =
    1e5 * std::numeric_limits<ScalarNumberType>::epsilon();
  std::cout << "psi:              " << psi << std::endl;
  std::cout << "func::psi(t):     " << func::psi(t) << std::endl;
  Assert(std::abs(psi - func::psi(t)) < tol,
         ExcMessage("No match for function value."));
  std::cout << "dpsi_dt:              " << dpsi_dt << std::endl;
  std::cout << "func::dpsi_dt(t):     " << func::dpsi_dt(t) << std::endl;
  Assert(std::abs((dpsi_dt - func::dpsi_dt(t)).norm()) < tol,
         ExcMessage("No match for first derivative."));
  if (AD::ADNumberTraits<ADNumberType>::n_supported_derivative_levels >= 2)
    {
      const Tensor<4, dim, ScalarNumberType> d2psi_dt_dt =
        ad_helper.extract_hessian_component(D2psi, t_dof, t_dof);
      std::cout << "d2psi_dt_dt:          " << d2psi_dt_dt << std::endl;
      std::cout << "func::d2psi_dt_dt(t): " << func::d2psi_dt_dt(t)
                << std::endl;
      Assert(std::abs((d2psi_dt_dt - func::d2psi_dt_dt(t)).norm()) < tol,
             ExcMessage("No match for second derivative."));
    }

  std::cout << std::endl << std::endl;
}
