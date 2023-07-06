// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2019 by the deal.II authors
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

// Test output for GridGenerator::general_cell() for dim != spacedim

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include "../tests.h"

void
dim_2_3(std::ostream &os)
{
  std::vector<Point<3>> vertices(4);
  vertices[0](0) = -1.;
  vertices[0](1) = -1.;
  vertices[0](0) = 0.;

  vertices[1](0) = 1.;
  vertices[1](1) = -1.5;
  vertices[1](2) = 0.;

  vertices[2](0) = 1.5;
  vertices[2](1) = 1.5;
  vertices[2](2) = 0.;

  vertices[3](0) = 2.;
  vertices[3](1) = 0.5;
  vertices[3](2) = 0.;

  Triangulation<2, 3> tria;
  GridGenerator::general_cell<2, 3>(tria, vertices);

  GridOut gout;
  gout.write_vtk(tria, os);
}

void
dim_1_3(std::ostream &os)
{
  std::vector<Point<3>> vertices(2);
  vertices[0](0) = -1.;
  vertices[0](1) = -1.;
  vertices[0](2) = -1.;
  vertices[1](0) = 1.;
  vertices[1](1) = -1.5;
  vertices[1](2) = -1.5;

  Triangulation<1, 3> tria;
  GridGenerator::general_cell<1, 3>(tria, vertices);

  GridOut gout;
  gout.write_vtk(tria, os);
}

void
dim_1_2(std::ostream &os)
{
  std::vector<Point<2>> vertices(2);
  vertices[0](0) = -1.;
  vertices[0](1) = -1.;

  vertices[1](0) = 1.;
  vertices[1](1) = -1.5;

  Triangulation<1, 2> tria;
  GridGenerator::general_cell<1, 2>(tria, vertices);

  GridOut gout;
  gout.write_vtk(tria, os);
}


int
main()
{
  initlog(true);
  std::ostream &logfile = deallog.get_file_stream();
  dim_2_3(logfile);
  dim_1_3(logfile);
  dim_1_2(logfile);
}
