// ---------------------------------------------------------------------
//
// Copyright (C) 2000 - 2020 by the deal.II authors
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


// like _08, but checks the copy_to_mg and copy_from_mg of MGTransferSelect

#include <deal.II/base/mg_level_object.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_transfer_component.h>

#include <algorithm>

#include "../tests.h"



template <int dim>
void
refine_mesh(Triangulation<dim> &triangulation)
{
  bool cell_refined = false;
  for (typename Triangulation<dim>::active_cell_iterator cell =
         triangulation.begin_active();
       cell != triangulation.end();
       ++cell)
    {
      const Point<dim> p        = cell->center();
      bool             positive = p(0) > 0;
      if (positive)
        {
          cell->set_refine_flag();
          cell_refined = true;
        }
    }
  if (!cell_refined) // if no cell was selected for refinement, refine global
    for (typename Triangulation<dim>::active_cell_iterator cell =
           triangulation.begin_active();
         cell != triangulation.end();
         ++cell)
      cell->set_refine_flag();
  triangulation.execute_coarsening_and_refinement();
}

template <int dim>
void
check(const FiniteElement<dim> &fe, const unsigned int selected_block)
{
  deallog << fe.get_name() << std::endl;

  Triangulation<dim> tr(Triangulation<dim>::limit_level_difference_at_vertices);
  std::vector<unsigned int> subdivisions(dim, 1);
  subdivisions[0] = 2;

  const Point<dim> bottom_left =
    (dim == 2 ? Point<dim>(-1, -1) : Point<dim>(-1, -1, -1));
  const Point<dim> top_right =
    (dim == 2 ? Point<dim>(1, 1) : Point<dim>(1, 1, 1));
  GridGenerator::subdivided_hyper_rectangle(
    tr, subdivisions, bottom_left, top_right, true);
  refine_mesh(tr);

  std::ostringstream out_filename;
  out_filename << "gitter.eps";

  //   std::ofstream grid_output (out_filename.str().c_str());
  //   GridOut grid_out;
  //   grid_out.write_eps (tr, grid_output);


  DoFHandler<dim> mg_dof_handler(tr);
  mg_dof_handler.distribute_dofs(fe);
  mg_dof_handler.distribute_mg_dofs();

  std::vector<unsigned int> block_component(5, 0);
  block_component[2] = 1;
  block_component[3] = 1;
  block_component[4] = 2;

  DoFRenumbering::component_wise(mg_dof_handler, block_component);
  for (unsigned int level = 0; level < tr.n_levels(); ++level)
    DoFRenumbering::component_wise(mg_dof_handler, level, block_component);


  MGTransferSelect<double> transfer;

  transfer.build(mg_dof_handler,
                 selected_block,
                 selected_block,
                 block_component,
                 block_component);

  const std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(mg_dof_handler, block_component);
  std::vector<std::vector<types::global_dof_index>> mg_dofs_per_block(
    tr.n_levels(), std::vector<types::global_dof_index>(3));
  MGTools::count_dofs_per_block(mg_dof_handler,
                                mg_dofs_per_block,
                                block_component);

  deallog << "Global  dofs:";
  for (unsigned int i = 0; i < dofs_per_block.size(); ++i)
    deallog << ' ' << dofs_per_block[i];
  deallog << std::endl;
  for (unsigned int l = 0; l < mg_dofs_per_block.size(); ++l)
    {
      deallog << "Level " << l << " dofs:";
      for (unsigned int i = 0; i < mg_dofs_per_block[l].size(); ++i)
        deallog << ' ' << mg_dofs_per_block[l][i];
      deallog << std::endl;
    }

  BlockVector<double> u;
  u.reinit(dofs_per_block);
  for (unsigned int i = 0; i < u.size(); ++i)
    u(i) = i;

  MGLevelObject<Vector<double>> v(0, tr.n_levels() - 1);
  for (unsigned int l = 0; l < tr.n_levels() - 1; ++l)
    v[l].reinit(mg_dofs_per_block[l][2]);


  transfer.copy_to_mg(mg_dof_handler, v, u);
  for (unsigned int l = 0; l < tr.n_levels(); ++l)
    {
      deallog << "Level " << l << std::endl;
      for (unsigned int i = 0; i < v[l].size(); ++i)
        deallog << ' ' << (int)v[l](i);
      deallog << std::endl;
    }

  for (unsigned int l = 0; l < tr.n_levels(); ++l)
    for (unsigned int i = 0; i < v[l].size(); ++i)
      v[l](i) = i;

  u = 0;
  transfer.copy_from_mg(mg_dof_handler, u, v);
  deallog << "Global" << std::endl;
  for (unsigned int i = 0; i < u.block(selected_block).size(); ++i)
    deallog << ' ' << (int)u.block(selected_block)(i);
  deallog << std::endl;
  for (unsigned int i = 0; i < u.size(); ++i)
    deallog << ' ' << (int)u(i);
  deallog << std::endl;
}


int
main()
{
  initlog();
  deallog << std::setprecision(4);

  // TODO: do in 1d
  check(FESystem<2>(FE_Q<2>(1), 5), 0);
  check(FESystem<2>(FE_Q<2>(1), 5), 1);
  check(FESystem<2>(FE_Q<2>(1), 5), 2);
}
