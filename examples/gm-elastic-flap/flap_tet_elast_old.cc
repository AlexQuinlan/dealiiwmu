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
 *
 * <br>
 *
 * <i>
 * This program was contributed by Peter Munch. This work and the required
 * generalizations of the internal data structures of deal.II form part of the
 * project "Virtual Materials Design" funded by the Helmholtz Association of
 * German Research Centres.
 * </i>
 *
 *
 * <a name="Intro"></a>
 * <h1>Introduction</h1>
 *
 * <h3>Motivation</h3>
 *
 * Many freely available mesh-generation tools produce meshes that consist of
 * simplices (triangles in 2D; tetrahedra in 3D). The reason for this is that
 * generating such kind of meshes for complex geometries is simpler than the
 * generation of hex-only meshes. This tutorial shows how to work on such kind
 * of meshes with the experimental simplex features in deal.II. For this
 * purpose, we solve the Poisson problem from @ref step_3 "step-3" in 2D with a mesh only
 * consisting of triangles.
 *
 *
 * <h3>Working on simplex meshes</h3>
 *
 * To be able to work on simplex meshes, one has to select appropriate finite
 * elements, quadrature rules, and mapping objects. In @ref step_3 "step-3", we used FE_Q,
 * QGauss, and (implicitly by not specifying a mapping) MappingQ1. The
 * equivalent classes for the first two classes in the context of simplices are
 * FE_SimplexP and QGaussSimplex, which we will utilize here. For mapping
 * purposes, we use the class MappingFE, which implements an isoparametric
 * mapping. We initialize it with an FE_SimplexP object so that it can be
 * applied on simplex meshes.
 *
 *
 * <h3>Mesh generation</h3>
 *
 * In contrast to @ref step_3 "step-3", we do not use a function from the GridGenerator
 * namespace, but rather read an externally generated mesh. For this tutorial,
 * we have created the mesh (square with width and height of one) with Gmsh with
 * the following journal file "box_2D_tri.geo":
 *
 * @code
 * Rectangle(1) = {0, 0, 0, 1, 1, 0};
 * Mesh 2;
 * Save "box_2D_tri.msh";
 * @endcode
 *
 * The journal file can be processed by Gmsh generating the actual mesh with the
 * ending ".geo":
 *
 * @code
 * gmsh box_2D_tri.geo
 * @endcode
 *
 * We have included in the tutorial folder both the journal file and the mesh
 * file in the event that one does not have access to Gmsh.
 *
 * The mesh can be simply read by deal.II with methods provided by the GridIn
 * class, as shown below.
 *
 */
 
 
// @sect3{Include files}
 
// Include files, as used in @ref step_3 "step-3":
#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
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
#include <deal.II/base/timer.h>
 
// Include files that contain appropriate quadrature rules, finite elements,
// and mapping objects for simplex meshes.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
 
// The following file contains the class GridIn, which allows us to read
// external meshes.
#include <deal.II/grid/grid_in.h>
#include <deal.II/base/function.h>                                                        
#include <deal.II/base/logstream.h> 
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>                
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/opencascade/manifold_lib.h>                                         
#include <deal.II/opencascade/utilities.h> 
#include <deal.II/fe/component_mask.h>

#include <fstream>                                                                        
#include <iostream>

#include <cmath>                                                                      
#include <string>    

using namespace dealii;
 
// @sect3{The <code>Step3</code> class}
//
// This is the main class of the tutorial. Since it is very similar to the
// version from @ref step_3 "step-3", we will only point out and explain the relevant
// differences that allow to perform simulations on simplex meshes.

template <int dim>
class tetElast
{
public:
  tetElast<dim>(
      const std::string &  initial_mesh_filename,                                     
      const std::string &  output_filename);
  void run_mesh();
  void run();
 
private:

  // read_domain();
  void make_grid();
  void nodal_bcs();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;
  void read_domain();
  void output_mesh();
 
 
  // Here, we select a mapping object, a finite element, and a quadrature rule
  // that are compatible with simplex meshes.
  const MappingFE<dim>     mapping;
  //  const FE_SimplexP<dim>   fe;
  const FESystem<dim>   fe;
  const QGaussSimplex<dim> quadrature_formula;
  AffineConstraints<double> constraints;
  Triangulation<dim> triangulation;

  DoFHandler<dim> dof_handler;
 
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
 
  Vector<double> solution;
  Vector<double> system_rhs;

  const std::string initial_mesh_filename;                                          
  const std::string output_filename; 
};
 
 
// @sect4{tetElast::tetElast}
//
// In the constructor, we set the polynomial degree of the finite element and
// the number of quadrature points. Furthermore, we initialize the MappingFE
// object with a (linear) FE_SimplexP object so that it can work on simplex
// meshes.
template <int dim>     
tetElast<dim>::tetElast(
    const std::string &  initial_mesh_filename,                                       
    const std::string &  output_filename)
  : initial_mesh_filename(initial_mesh_filename)
  , output_filename(output_filename) 
  , mapping(FE_SimplexP<dim>(1))
  , fe(FE_SimplexP<dim>(2),dim)
  , quadrature_formula(3)
  , dof_handler(triangulation)
{}
 
// @sect4{tetElast::make_grid}

template <int dim>
  void tetElast<dim>::read_domain()                                              
  {
    std::ifstream in;                                                                 
    in.open(initial_mesh_filename);                                                   
    GridIn<3> gi;
    gi.attach_triangulation(triangulation);
    gi.read_msh(in);
  }

template <int dim>
void tetElast<dim>::make_grid()
{
  GridIn<dim>(triangulation).read("input/Flap_aq.msh");
  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;
}

template <int dim>
void tetElast<dim>::output_mesh()
  {
    std::ofstream logfile(output_filename);
    GridOut       grid_out;
    grid_out.write_vtk(triangulation, logfile);
  }


template <int dim>
  void tetElast<dim>::run_mesh()                                                      
  {                                                                                   
    // This function can be more complex for other jobs
    read_domain();
    output_mesh();
  }


template <int dim>
void tetElast<dim>::nodal_bcs()
{
  // Define location of contrained nodes (x,y,z)
    const Point<dim> proot1(367.138, -519.886, 827.035);
    const Point<dim> proot2( 383.448, -527.473, 827.035 );
    const Point<dim> proot3( 381.893, -533.753, 899.548);
    const Point<dim> load1( 393.015, -516.585, 865.016);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      for (unsigned int v = 0; v<cell->n_vertices() ; v++)  // might be able to use 'vertex_iterator'
        {
          if (cell->vertex(v).distance(proot1) < 0.001)   // This mess should be reduced once I get this working
            {
              constraints.add_line(cell->vertex_dof_index(v,0, cell->active_fe_index() ));
              constraints.add_line(cell->vertex_dof_index(v,1, cell->active_fe_index() ));
              constraints.add_line(cell->vertex_dof_index(v,2, cell->active_fe_index() ));
              std::cout << "Root 1" << std::endl;
            }
          if (cell->vertex(v).distance(proot2) < 0.001)
            {
              constraints.add_line(cell->vertex_dof_index(v,0, cell->active_fe_index() ));
              constraints.add_line(cell->vertex_dof_index(v,1, cell->active_fe_index() ));
              constraints.add_line(cell->vertex_dof_index(v,2, cell->active_fe_index() ));
              std::cout << "Root 2" << std::endl;
            }
          if (cell->vertex(v).distance(proot3) < 0.001)
            {
              constraints.add_line(cell->vertex_dof_index(v,0, cell->active_fe_index() ));
              constraints.add_line(cell->vertex_dof_index(v,1, cell->active_fe_index() ));
              constraints.add_line(cell->vertex_dof_index(v,2, cell->active_fe_index() ));
              std::cout << "Root 3" << std::endl;
            }
          if (cell->vertex(v).distance(load1) < 0.001)
            {
              constraints.add_line(cell->vertex_dof_index(v,0, cell->active_fe_index() ));
              constraints.set_inhomogeneity(cell->vertex_dof_index(v,0, cell->active_fe_index()), 1.5);
              std::cout << "Load 1" << std::endl;
            }

        }
    }


}





// See cadex for BoundaryValues function example (overloaded)
template <int dim>
 class BoundaryValues : public Function<dim>
 {
 public:
	BoundaryValues() : Function<dim>(dim)   // The (dim) is needed for matching the BoundaryValue function
	    {}
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  value) const override;
 };


template <int dim>
 double BoundaryValues<dim>::value(const Point<dim> & p,
                                   const unsigned int component) const
 {
   Assert(component < this->n_components,
          ExcIndexRange(component, 0, this->n_components));
   if (component == 1)
	   {
	   return 1.0;
	   }
//     return (p[0] < 0 ? -1 : (p[0] > 0 ? 1 : 0));
   return 0;
 }

 template <int dim>
 void BoundaryValues<dim>::vector_value(const Point<dim> &p,
                                        Vector<double> &  values) const
 {
   for (unsigned int c = 0; c < this->n_components; ++c)
     values(c) = BoundaryValues<dim>::value(p, c);
 }

  template <int dim>
  void right_hand_side(const std::vector<Point<dim>> &points,
  		       std::vector<Tensor<1, dim>> & values)
  {
    AssertDimension(values.size(), points.size());
    Assert(dim >= 2, ExcNotImplemented());

    // Force values
    // Point force at point 1
//    Point<dim> point_1;
//    point_1(0) =  413.0;
//    point_1(1) = -519.0;
//    point_1(2) = 1245.0;
	Point<dim> point_1;
//	point_1(0) = 389.257;
//	point_1(1) = -520.864;
//	point_1(2) = 836.817;
	point_1(0) = 389.0;
	point_1(1) = -520.0;
	point_1(2) = 836.0;
	int rad1 = 1.0;

    for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
      {
		if ( (points[point_n] - point_1).norm_square() < rad1)
		{
		  values[point_n][0] = 0.0;
//		  std::cout << "Load Point: " << point_n << std::endl;
		}
		else
		  values[point_n][2] = 0.0;
      }


  }
 
template <int dim>
void tetElast<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;
  
 
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
  
  constraints.clear();
  
  nodal_bcs();

  // Setting two point/sphere constraints
//  const Point<dim> p1(376.47 , -525.54 , 852.04);
////  const Point<dim> p1(0.0 , 0.0 , 0.0);   // for debug cube
//
//  const Point<dim> p2(413.0 , -495.0 , 1083.0);
//  const double rad1 = 5.0;
////  const double rad1 = 0.15;  // for debug cube
//  const double rad2 = 19.0;
//
//  const Point<dim> LP1(367.138, -519.886, 827.035);
//  const double radLP1 = 2.0;

  int bcfs = 0;
  
 // Fix Points
//
//  for (const auto &vertex : triangulation.vertex_iterator())
//  {
//	  std::cout << "vertex" << std::endl;
//  }

//  for (const auto &cell :   triangulation.active_cell_iterators())
//  {
//	  for (const auto &vertex : cell->vertex_indices())
//	  {
//		  std::cout << "vertex" << std::endl;
//	  }


//
//	  for (const auto &face : cell->face_iterators())
//    {
//    	if (face->at_boundary())
//    	{
//    if (  face->center().distance(p1) < rad1)
//    			{
//    			face->set_boundary_id(1);
//    			bcfs += 1;
//
//    			}
//    			else if (  face->center().distance(p2) < rad2)
//    				face->set_boundary_id(2);
//
//    			else if (  face->center().distance(LP1) < radLP1)
//    			{
//					face->set_boundary_id(3);
//					std::cout << "set face load boundary id" << std::endl;
//    			}
//    	}
//
//    }
//  }

//  std::cout << "Boundary condition faces" << std::endl;
//  std::cout << bcfs << std::endl;

//  for (const auto &cell : triangulation.active_cell_iterators())
//  {
//    for (const auto &face : cell->face_iterators())
//    {
//    	if (face->at_boundary())
//    	{
//    		if (  face->center()[2] < 1225.0)
//    			face->set_boundary_id(1);
//    	}
//
//    }
//  }


  
  // Setting fixed boundary conditions to two sets of BoundaryIDs
//  VectorTools::interpolate_boundary_values(dof_handler,
//  					     1,
//  					     Functions::ZeroFunction<dim>(dim),
//  					     constraints);
  
//  VectorTools::interpolate_boundary_values(dof_handler,
//  					     2,
//  					     Functions::ZeroFunction<dim>(dim),
//  					     constraints);
  
//    VectorTools::interpolate_boundary_values(dof_handler,
//    					     2,
//    					     Functions::ZeroFunction<dim>(dim),
//    					     constraints);
//    const FEValuesExtractors::Scalar   y_component(dim - 2);
//    VectorTools::interpolate_boundary_values(dof_handler,
//					     3,
//					     BoundaryValues<dim>(),
//					     constraints,
//						 fe.component_mask(y_component));

  constraints.close();		  
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
  
}
 
 
// @sect4{tetElast::assemble_system}
//
// Nothing has changed here.
template <int dim>     
void tetElast<dim>::assemble_system()
{
  FEValues<dim> fe_values(mapping,
                        fe,
                        quadrature_formula,
                        update_values | update_gradients |
			update_quadrature_points | update_JxW_values);
 
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points =    quadrature_formula.size();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double> lambda_values(n_q_points);
  std::vector<double> mu_values(n_q_points);
  Functions::ConstantFunction<dim> lambda(1150.6), mu(3565.1);
  std::vector<Tensor<1,dim>> rhs_values(n_q_points);

  
  // Original -------------------------
  // for (const auto &cell : dof_handler.active_cell_iterators())
  //   {
  //     fe_values.reinit(cell);
  //     cell_matrix = 0;
  //     cell_rhs    = 0;

      
  //     for (const unsigned int q_index : fe_values.quadrature_point_indices())
  //       {
  //         for (const unsigned int i : fe_values.dof_indices())
  //           for (const unsigned int j : fe_values.dof_indices())
  //             cell_matrix(i, j) +=
  //               (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
  //                fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
  //                fe_values.JxW(q_index));           // dx

  //         for (const unsigned int i : fe_values.dof_indices())
  //           cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
  //                           1. *                                // f(x_q)
  //                           fe_values.JxW(q_index));            // dx
  //       }
  //     cell->get_dof_indices(local_dof_indices);
 
  //     for (const unsigned int i : fe_values.dof_indices())
  //       for (const unsigned int j : fe_values.dof_indices())
  //         system_matrix.add(local_dof_indices[i],
  //                           local_dof_indices[j],
  //                           cell_matrix(i, j));
 
  //     for (const unsigned int i : fe_values.dof_indices())
  //       system_rhs(local_dof_indices[i]) += cell_rhs(i);
  //   }

  // std::map<types::global_dof_index, double> boundary_values;
  // VectorTools::interpolate_boundary_values(
  //   mapping, dof_handler, 0, Functions::ZeroFunction<dim>(), boundary_values);

  // MatrixTools::apply_boundary_values(boundary_values,
  //                                    system_matrix,
  //                                    solution,
  //                                    system_rhs);

  // --------------------------------------------------
  // -------- End Original ----------------------------
  // --------------------------------------------------


  // =======================================
  // ========= From cadex ==================
  // =======================================

  for (const auto &cell : dof_handler.active_cell_iterators())
      {
	cell_matrix=0;
	cell_rhs   =0;
	fe_values.reinit(cell);

	// Get values for lambda, mu, and rhs
	lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
	mu.value_list(fe_values.get_quadrature_points(), mu_values);
	right_hand_side(fe_values.get_quadrature_points(), rhs_values);

	for (const unsigned int i : fe_values.dof_indices())
	  {
	    const unsigned int component_i =
	      fe.system_to_component_index(i).first;
	    for (const unsigned int j : fe_values.dof_indices())
	      {
		const unsigned int component_j =
		  fe.system_to_component_index(j).first;

		for (const unsigned int q_point :
		       fe_values.quadrature_point_indices())
		  {
		    cell_matrix(i,j) +=

		      (
		       (fe_values.shape_grad(i, q_point)[component_i] *
			fe_values.shape_grad(j, q_point)[component_j] *
			lambda_values[q_point])
		       +
		       (fe_values.shape_grad(i, q_point)[component_j] *
			fe_values.shape_grad(j, q_point)[component_i] *
			mu_values[q_point])
		       +
		       ((component_i == component_j) ?
			(fe_values.shape_grad(i, q_point) *
			 fe_values.shape_grad(j, q_point) *
			 mu_values[q_point]) 
			:
			0) ) *
		      fe_values.JxW(q_point);
		  }
	      }
	  }


  // Assembling RHS (from cadex)
  for (const unsigned int i : fe_values.dof_indices())
	  {
	    const unsigned int component_i =
	      fe.system_to_component_index(i).first;

	    for (const unsigned int q_point :
		   fe_values.quadrature_point_indices())
	      cell_rhs(i) += fe_values.shape_value(i, q_point) *
		             rhs_values[q_point][component_i] *
		             fe_values.JxW(q_point);
	  }
  cell->get_dof_indices(local_dof_indices);
  constraints.distribute_local_to_global(
		       cell_matrix, cell_rhs,
		       local_dof_indices,
		       system_matrix,
		       system_rhs);
      }      
  
  // =============================
  // === End import from cadex ===
  // =============================
 

  // Manually modifying the RHS with a point force
  // Need to move into a function
//  int nodenum = 1;
//  int locdofnum = 0;
//  double loadval = 1.0;
//  int dofnum;
//  dofnum = (nodenum)*3 + locdofnum;
//
//  system_rhs(dofnum) += loadval;
}
 
// @sect4{tetElast::solve}
//
// Nothing has changed here.
template <int dim>     
void tetElast<dim>::solve()
{
  SolverControl            solver_control(4000, 1e-2); //what are these controls?
  SolverCG<Vector<double>> cg(solver_control);

  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);
  cg.solve(system_matrix, solution, system_rhs, preconditioner);
  constraints.distribute(solution);


}
 
 
// @sect4{tetElast::output_results}
//
// Nothing has changed here.
template <int dim>     
void tetElast<dim>::output_results() const
{
  DataOut<dim> data_out;
 
//  DataOutBase::VtkFlags flags;
//  flags.write_higher_order_cells = true;
//  data_out.set_flags(flags);
 
  data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> solution_names;
    switch (dim)
      {
	case 1:
          solution_names.emplace_back("displacement");
          break;
        case 2:
          solution_names.emplace_back("x_displacement");
          solution_names.emplace_back("y_displacement");
          break;
        case 3:
          solution_names.emplace_back("x_displacement");
          solution_names.emplace_back("y_displacement");
          solution_names.emplace_back("z_displacement");
          break;
        default:
          Assert(false, ExcNotImplemented());
      }

    data_out.add_data_vector(solution, solution_names);
    //    data_out.build_patches();



  // data_out.attach_dof_handler(dof_handler);
  // data_out.add_data_vector(solution, "solution");
  data_out.build_patches(mapping, 2);
  // std::ofstream output("solution.vtu");
  // data_out.write_vtu(output);

  // The '2' in .build_patches is not the dimension.  I think it's the degree
  // Changed vtk to vtu

  std::ofstream output("partial-flap-tetelast.vtu");
  data_out.write_vtu(output);
}
 
 
// @sect4{tetElast::run}
//
// Nothing has changed here.
template <int dim>     
void tetElast<dim>::run()
{
  //  make_grid();
  setup_system();
    std::cout << "Complete: setup_system" << std::endl;
  assemble_system();
    std::cout << "Complete: assemble_system" << std::endl;
  solve();
    std::cout << "Complete: solve" << std::endl;
  output_results();
    std::cout << "Complete: output_results" << std::endl;

}
 
 
// @sect3{The <code>main</code> function}
//
// Nothing has changed here.
int main()
{

  deallog.depth_console(2);
  Timer timer;

//  const std::string in_mesh_filename = "input/Flap_aq.msh";
  std::string in_mesh_filename = "input/partial_flap_aq.msh";

//  const std::string in_mesh_filename = "input/trial.msh";
  const std::string out_mesh_filename = ("output/3d_mesh.vtk");

  tetElast<3> elastic3D(in_mesh_filename,out_mesh_filename);
  elastic3D.run_mesh();
  elastic3D.run();

  std::cout << " ======= Done with program ========" << std::endl;
  timer.stop();

  std::cout << "Elapsed CPU time: " << timer.cpu_time() << " seconds.\n";
  std::cout << "Elapsed wall time: " << timer.wall_time() << " seconds.\n";

  timer.reset();
  return 0;
}
