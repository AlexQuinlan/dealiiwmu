// ---------------------------------------------------------------------
//
// Copyright (C) 2010 - 2018 by the deal.II authors
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


// check method TmTmult of FullMatrix

#include <deal.II/lac/eigen.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include "../tests.h"

const double entries_A[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
const double entries_B[9] = {2, 1, 1, 1, 2, 3, 2, 1, 2};
const double entries_Z[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

// Create a positive definite random matrix

int
main()
{
  initlog();
  deallog << std::fixed;
  deallog << std::setprecision(3);
  Testing::srand(3391466);

  FullMatrix<double> A(3, 3, entries_A);
  FullMatrix<double> B(3, 3, entries_B);
  FullMatrix<double> Za(3, 3, entries_Z);
  FullMatrix<double> Zb(3, 3, entries_Z);
  FullMatrix<double> C(3, 3);
  FullMatrix<double> D(3, 3);

  // compute C= A^T*B^T in two different ways and compare for equality
  Za.Tadd(1., A);
  Zb.Tadd(1., B);
  Za.mmult(D, Zb);
  A.TmTmult(C, B);

  D.add(-1, C);
  AssertThrow(D.frobenius_norm() < 1e-15, ExcInternalError());
  deallog << "OK" << std::endl;
}
