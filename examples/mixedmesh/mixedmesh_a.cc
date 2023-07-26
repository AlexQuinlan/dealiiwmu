/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2021 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 * xxx Alex modified the original simplex into 3d, then modified to be mixed mesh
 */
#include <deal.II/base/function.h>
 
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
 
#include <deal.II/grid/tria.h>
 
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
 
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
 
#include <deal.II/base/quadrature_lib.h>
 
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
 
#include <deal.II/grid/grid_in.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>


using namespace dealii;

template <int dim>
class Step3
{
public:
  Step3();
 
  void run();
 
private:

  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;
 
  Triangulation<dim> triangulation;
 
  const hp::MappingCollection<dim>     mapping;
  const hp::FECollection<dim>   fe;
  const hp::QCollection<dim> quadrature_formula;
 

  DoFHandler<dim> dof_handler;
 
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
 
  Vector<double> solution;
  Vector<double> system_rhs;
};
 
template <int dim>     
Step3<dim>::Step3()
  : mapping(MappingFE<dim>(FE_SimplexP<dim>(1)), MappingFE<dim>(FE_Q<dim>(1)))
  , fe(FE_SimplexP<dim>(2), FE_Q<dim>(2))
  , quadrature_formula(QGaussSimplex<dim>(3), QGauss<dim>(3))
  , dof_handler(triangulation)
{}
 
template <int dim>
void Step3<dim>::make_grid()
{
  GridIn<dim>(triangulation).read("input/mixed.msh");
 
  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;
}
 
template <int dim>
void Step3<dim>::setup_system()
{

  // The following section added for Mixed Dofs
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->reference_cell() == ReferenceCells::Tetrahedron)
        cell->set_active_fe_index(0);
      else if (cell->reference_cell() == ReferenceCells::Hexahedron)
        cell->set_active_fe_index(1);
      else
        Assert(false, ExcNotImplemented());
    } 

  std::cout << "About to do the dof_handler " << std::endl;
  dof_handler.distribute_dofs(fe);
  std::cout << "Finished the dof_handler " << std::endl;
  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
 
  system_matrix.reinit(sparsity_pattern);
 
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}
 
template <int dim>     
void Step3<dim>::assemble_system()
{
  hp::FEValues<dim> hp_fe_values(mapping,
                        fe,
                        quadrature_formula,
                        update_values | update_gradients | update_JxW_values);
 
  // Note that dofs_per_cell has been removed from these declarations
  FullMatrix<double> cell_matrix;
  Vector<double>     cell_rhs;
  std::vector<types::global_dof_index> local_dof_indices;
 
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      hp_fe_values.reinit(cell);

      const auto &fe_values = hp_fe_values.get_present_fe_values();

      const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
      cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_rhs.reinit(dofs_per_cell);
      local_dof_indices.resize(dofs_per_cell);
      
      cell_matrix = 0;
      cell_rhs    = 0;
 
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx

          for (const unsigned int i : fe_values.dof_indices())
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            1. *                                // f(x_q)
                            fe_values.JxW(q_index));            // dx
        }
      cell->get_dof_indices(local_dof_indices);
 
      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          system_matrix.add(local_dof_indices[i],
                            local_dof_indices[j],
                            cell_matrix(i, j));
 
      for (const unsigned int i : fe_values.dof_indices())
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }
 
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ZeroFunction<dim>(), boundary_values);

  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);
}

template <int dim>     
void Step3<dim>::solve()
{
  SolverControl            solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
}
 
template <int dim>     
void Step3<dim>::output_results() const
{
  DataOut<dim> data_out;
 
//  DataOutBase::VtkFlags flags;
//  flags.write_higher_order_cells = true;
//  data_out.set_flags(flags);
 
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches(mapping, 2);
  std::ofstream output("solution.vtu");
  data_out.write_vtu(output);

  // Changed vtk to vtu
}
 
template <int dim>     
void Step3<dim>::run()
{
  make_grid();
  setup_system();
  assemble_system();
  solve();
  output_results();
  std::cout << "Done" << std::endl;
}
 
int main()
{
  deallog.depth_console(2);
 
  Step3<3> laplace_problem;
  laplace_problem.run();
 
  return 0;
}
 

