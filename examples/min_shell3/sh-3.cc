#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

using namespace dealii;

using namespace GridTools;

template <int dim, int spacedim>
void
apply_grid_fixup_functions(std::vector<Point<spacedim>> &vertices,
                           std::vector<CellData<dim>> &  cells,
                           SubCellData &                 subcelldata)
{
  // check that no forbidden arrays are used
  Assert(subcelldata.check_consistency(dim), ExcInternalError());
  const auto n_hypercube_vertices =
    ReferenceCells::get_hypercube<dim>().n_vertices();
  bool is_only_hypercube = true;
  for (const CellData<dim> &cell : cells)
    if (cell.vertices.size() != n_hypercube_vertices)
      {
        is_only_hypercube = false;
        break;
      }

//  GridTools::delete_unused_vertices(vertices, cells, subcelldata);
//  if (dim == spacedim)
//    GridTools::invert_cells_with_negative_measure(vertices, cells);
//
//  if (is_only_hypercube)
//    GridTools::consistently_order_cells(cells);
}
		//}// end namespace


template <int dim, int spacedim>
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
  Triangulation<dim,spacedim> triangulation;
  FE_Q<dim,spacedim>          fe;
  DoFHandler<dim, spacedim>    dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
};

template <int dim, int spacedim>
Step3<dim,spacedim>::Step3()
  : fe(1)
  , dof_handler(triangulation)
{}

template <int dim, int spacedim>
void Step3<dim,spacedim>::make_grid()
{
  //  GridGenerator::hyper_cube(triangulation, -1, 1);
  //  triangulation.refine_global(5);


	std::map<int, int> vertex_indices;

	vertex_indices[vertex_number] = vertex


  Point<spacedim,double> pt1 (0.0 , 0.0, 0.0);
  Point<spacedim,double> pt2 (1.0 , 0.0, 0.5);
  Point<spacedim,double> pt3 (0.0 , 1.0, 0.0);
  Point<spacedim,double> pt4 (1.0 , 1.0, 0.5);


  
  std::vector< dealii::Point<spacedim,double>> vertices;
  std::vector< dealii::CellData<dim> > cells;
  dealii::CellData<dim> thecell;
  
  dealii::SubCellData  subcelldata;  // boundary_lines, boundary_lines.vertices[]

  ///   Do the vertices
//  vertices.emplace_back();
//  for (unsigned int d = 0; d < spacedim; ++d)
//     vertices.back()(d) = pt1(d);
//  vertices.emplace_back();
//  for (unsigned int d = 0; d < spacedim; ++d)
//     vertices.back()(d) = pt2(d);
//  vertices.emplace_back();
//  for (unsigned int d = 0; d < spacedim; ++d)
//     vertices.back()(d) = pt3(d);
//  vertices.emplace_back();
//  for (unsigned int d = 0; d < spacedim; ++d)
//     vertices.back()(d) = pt4(d);

  vertices.emplace_back();
  vertices.back() = pt1;
  vertices.emplace_back();
  vertices.back() = pt2;
  vertices.emplace_back();
  vertices.back() = pt3;
  vertices.emplace_back();
  vertices.back() = pt4;

  ///   Do a single cell
  thecell.vertices[0] = 0;
  thecell.vertices[1] = 1;
  thecell.vertices[2] = 3;
  thecell.vertices[3] = 2;

  thecell.material_id = 1;
  thecell.boundary_id = 1;
  thecell.manifold_id = static_cast<types::manifold_id>(1);

  ///   Add cell to cell list
  cells.emplace_back();
  cells.back()= thecell;

//  vertex_indices[vertex_number] = vertex;
  vertex_indices[0] = 0;
  vertex_indices[1] = 1;
  vertex_indices[2] = 2;
  vertex_indices[3] = 3;



//  cells.emplace_back();
//  cells.back()= thecell;
  //  Neglect subcell data for now
  //  subcelldata.boundary_quads.emplace_back(n_vertices);
//  apply_grid_fixup_functions(vertices, cells, subcelldata);
  triangulation.create_triangulation(vertices, cells, subcelldata);
  
  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;

  // Also export the grid
  std::ofstream logfile("grid_out.vtu");
  GridOut       grid_out;
  grid_out.write_vtu(triangulation, logfile);


}



template <int dim, int spacedim>
 void Step3<dim,spacedim>::setup_system()
 {
   dof_handler.distribute_dofs(fe);
//   std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
//             << std::endl;
   DynamicSparsityPattern dsp(dof_handler.n_dofs());
   DoFTools::make_sparsity_pattern(dof_handler, dsp);
   sparsity_pattern.copy_from(dsp);
   system_matrix.reinit(sparsity_pattern);
   solution.reinit(dof_handler.n_dofs());
   system_rhs.reinit(dof_handler.n_dofs());
 }

template <int dim, int spacedim>
 void Step3<dim,spacedim>::assemble_system()
 {
   QGauss<dim> quadrature_formula(fe.degree + 1);
   FEValues<dim,spacedim> fe_values(fe,
                         quadrature_formula,
                         update_values | update_gradients | update_JxW_values);
   const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
   FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
   Vector<double>     cell_rhs(dofs_per_cell);
   std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
   for (const auto &cell : dof_handler.active_cell_iterators())
     {
       fe_values.reinit(cell);
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

       // And again, we do the same thing for the right hand side vector.
       for (const unsigned int i : fe_values.dof_indices())
         system_rhs(local_dof_indices[i]) += cell_rhs(i);
     }


   std::map<types::global_dof_index, double> boundary_values;
   VectorTools::interpolate_boundary_values(dof_handler,
                                            0,
                                            Functions::ZeroFunction<spacedim>(),
                                            boundary_values);
   MatrixTools::apply_boundary_values(boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);
 }

template <int dim, int spacedim>

 void Step3<dim,spacedim>::solve()
 {
   SolverControl            solver_control(1000, 1e-6 * system_rhs.l2_norm());
   SolverCG<Vector<double>> solver(solver_control);
   solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
 }


template <int dim, int spacedim>
 void Step3<dim,spacedim>::output_results() const
 {
   DataOut<dim,spacedim> data_out;
   data_out.attach_dof_handler(dof_handler);
   data_out.add_data_vector(solution, "solution");
   data_out.build_patches();
   std::ofstream output("solution.vtu");
   data_out.write_vtu(output);
//   std::ofstream output2("solution.vtk");
//   data_out.write_vtk(output2);
 }
template <int dim, int spacedim>
void Step3<dim,spacedim>::run()
{
  make_grid();
  setup_system();
   assemble_system();
   solve();
   output_results();
}


int main()
{
  deallog.depth_console(2);

  Step3<2,3> laplace_problem;
  laplace_problem.run();

  return 0;
}
