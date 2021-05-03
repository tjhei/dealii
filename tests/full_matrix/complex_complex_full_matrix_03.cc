// ---------------------------------------------------------------------
//
// Copyright (C) 2007 - 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------



// check running over const iterators starting at the second line. like the
// full_matrix_* tests, but use complex-valued matrices and vectors; this time
// we actually store complex values in them


#include "../tests.h"

#include "full_matrix_common.h"



template <typename number>
void
check()
{
  FullMatrix<std::complex<number>> m;
  make_complex_matrix(m);


  for (typename FullMatrix<std::complex<number>>::const_iterator p = m.begin(1);
       p != m.end(1);
       ++p)
    deallog << p->row() << ' ' << p->column() << ' ' << p->value() << std::endl;
}
