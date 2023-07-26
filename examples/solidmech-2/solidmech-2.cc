// ----------- Header file inclusion ---------------

#include <deal.II/base/quadrature_lib.h>                                                  
#include <deal.II/base/function.h>                                                        
#include <deal.II/base/logstream.h>                                                       
#include <deal.II/lac/vector.h>                                                           
#include <deal.II/lac/full_matrix.h>                                                      
#include <deal.II/lac/sparse_matrix.h>                                                    
#include <deal.II/lac/dynamic_sparsity_pattern.h>                                         
#include <deal.II/lac/solver_cg.h>                                                        
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>                                                            
#include <deal.II/dofs/dof_handler.h>                                                     
#include <deal.II/dofs/dof_tools.h>                                                       
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>                                                              
#include <deal.II/fe/fe_values.h>                                                         
#include <deal.II/numerics/vector_tools.h>                                                
#include <deal.II/numerics/matrix_tools.h>                                                
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/grid/grid_in.h>                                                     

#include <fstream>                                                                        
#include <iostream>

#include <cmath>                                                                      
#include <string>  

// ----------- Specifying Name Spaces --------------
namespace solidmech
{
  using namespace dealii;

// ----------- Create the Analysis Class -----------

template <int dim>   // Apply this template to classes and functions to specify the dimension 'dim'
  class ElasticProblem
  {
  public:

    ElasticProblem<dim>(
			const std::string & initial_mesh_filename);
    void run();

  private:
    void read_domain();
    void nodal_bcs();
    void setup_system();
    void assemble_system();
    void solve();
    void output_results();

    Triangulation<dim>               tria;
    FESystem<dim>                      fe;
    DoFHandler<dim>           dof_handler;
    AffineConstraints<double> constraints;
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double>    system_matrix;

    Vector<double>               solution;
    Vector<double>             system_rhs;  // Note that these Vectors are different than std::vector

    const std::string initial_mesh_filename;
};  // End of Creating Analysis Class


// ---------- Constructor Method ---------------
template <int dim>
  ElasticProblem<dim>::ElasticProblem(
     const std::string & initial_mesh_filename)
    :initial_mesh_filename(initial_mesh_filename)
    ,fe(FE_Q<dim>(1), dim)
    ,dof_handler(tria)
{}

// --------- Start the Methods -------------------
template <int dim>
  void ElasticProblem<dim>::read_domain()
  {
    std::ifstream in;                                                                 
    in.open(initial_mesh_filename);                                                   
    GridIn<3> gi;
    gi.attach_triangulation(tria);
    gi.read_abaqus(in, true);
  }

template <int dim>
void ElasticProblem<dim>::nodal_bcs()
{
  // ** Note: see bc-reader.cc for importing nodal BCs ***************

  // Define location of contrained nodes (x,y,z)
    const Point<dim> proot1(0., 0.0209, -0.0101);
    const Point<dim> proot2(10., 0.0209, -0.0101);
    const Point<dim> proot3(0., 34.9272, 1.04056);


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
    	}
    }

      
}

template <int dim>
  void ElasticProblem<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
  nodal_bcs();

  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs() );
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
}

template <int dim>
  void ElasticProblem<dim>::assemble_system()
{
  QGauss<dim> quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe,
			  quadrature_formula,
			  update_values | update_gradients |
			  update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  int num_ptLoads = 2;

  Vector<double>     ptld_rhs(num_ptLoads);
  std::vector<types::global_dof_index> ptld_idx(num_ptLoads);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double> lambda_values(n_q_points);
  std::vector<double> mu_values(n_q_points);
  Functions::ConstantFunction<dim> lambda(84.0), mu(84.0);


  std::vector<Tensor<1,dim>> rhs_values(n_q_points);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_matrix=0;
      cell_rhs   =0;
      fe_values.reinit(cell);

      lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
      mu.value_list(fe_values.get_quadrature_points(), mu_values);

      for (const unsigned int i : fe_values.dof_indices())
	{
	  const unsigned int component_i =
	    fe.system_to_component_index(i).first;

	  for (const unsigned int j : fe_values.dof_indices())
	    {
	      const unsigned int component_j = fe.system_to_component_index(j).first;

	      for (const unsigned int q_point : fe_values.quadrature_point_indices())
		{
		  cell_matrix(i,j) += (
				       (fe_values.shape_grad(i,q_point)[component_i] *
					fe_values.shape_grad(j,q_point)[component_j] *
					lambda_values[q_point])
				       +
				       (fe_values.shape_grad(i,q_point)[component_j] *                                                                       
                                        fe_values.shape_grad(j,q_point)[component_i] *                                                                       
                                        lambda_values[q_point])
				       +
				       ((component_i == component_j) ?
					(fe_values.shape_grad(i,q_point) *
					 fe_values.shape_grad(j,q_point) *
					 mu_values[q_point])
				       :
					0) ) * fe_values.JxW(q_point);
		}
	    }
	}

      for (const unsigned int i : fe_values.dof_indices())
	{
	  const unsigned int component_i = fe.system_to_component_index(i).first;
	  for (const unsigned int q_point : fe_values.quadrature_point_indices())
	    {
	      cell_rhs(i)+= fe_values.shape_value(i,q_point) *
     		            rhs_values[q_point][component_i] *
                	    fe_values.JxW(q_point);
	    }
	}

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
					     cell_matrix, cell_rhs,
					     local_dof_indices,
					     system_matrix,
					     system_rhs);


    const Point<dim> load1(10., 34.9272, 1.04056);
    for (unsigned int v = 0; v<cell->n_vertices() ; v++)  // might be able to use 'vertex_iterator'
	{
    	if (cell->vertex(v).distance(load1) < 0.001)
		{
    		ptld_rhs(0) = 33.33;   // assigns okay... but this could maybe be pre-assigned from file
    		ptld_idx[0] = local_dof_indices[v+2];
		}
	}

    // this was missing for some reason...
       // make the loads and the indices vectors
   // Assign Point Force
	constraints.distribute_local_to_global(
					       ptld_rhs,
					       ptld_idx,
					       system_rhs);
    }  
}


template <int dim>
void ElasticProblem<dim>::solve()
{
  SolverControl             solver_control(1000, 1e-2);
  SolverCG<Vector<double>>          cg(solver_control);

  PreconditionSSOR<SparseMatrix<double>>  preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  cg.solve(system_matrix, solution, system_rhs, preconditioner);
  constraints.distribute(solution);
}

template <int dim>
void ElasticProblem<dim>::output_results()
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  std::vector<std::string> solution_names;
  switch (dim)
    {
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
  data_out.build_patches();
  std::ofstream output("solution_sm2.vtu");
  data_out.write_vtu(output);
}
    
    

 
// ========== Run Function =========================
template <int dim>
void ElasticProblem<dim>::run()
{
  read_domain();
  setup_system();
  assemble_system();
  solve();
  output_results();
}
}  // close the namespace solidmech
// ===========  Main Function  ======================

int main()                                                                                                                                               
{                                                                                                                                                        
  using namespace solidmech;                                                                                                                   
  const std::string in_mesh_filename = "input/Mesh.inp";                                                                                       
  ElasticProblem<3> stran_3d(in_mesh_filename);                                                                                                 
  stran_3d.run();                                                                                                                              
}   
