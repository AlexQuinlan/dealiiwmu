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
#include <deal.II/fe/fe_simplex_p.h>   // added for tet
#include <deal.II/fe/mapping_fe.h>   // added for tet
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

#include <string>
#include <fstream>                                                                        
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <cwchar>
#include <cmath>
#include <tuple>

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
    void read_bcs();
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
    const MappingFE<dim>     mapping;            // added for tet
    const QGaussSimplex<dim> quadrature_formula;                // added for tet

    const std::string initial_mesh_filename;
};  // End of Creating Analysis Class


// ---------- Constructor Method ---------------
template <int dim>
  ElasticProblem<dim>::ElasticProblem(
     const std::string & initial_mesh_filename)
    :initial_mesh_filename(initial_mesh_filename)
    ,fe(FE_SimplexP<dim>(2), dim)   // fe(FE_Q<dim>(1), dim) // added for tet
    ,dof_handler(tria)
	, quadrature_formula(3)
	,mapping(FE_SimplexP<dim>(1))// added for tet
{}

// --------- Start the Methods -------------------
template <int dim>
  void ElasticProblem<dim>::read_domain()
  {
    std::ifstream in;                                                                 
    in.open(initial_mesh_filename);                                                   
    GridIn<3> gi;
    gi.attach_triangulation(tria);
//    gi.read_abaqus(in, true);
    gi.read_msh(in);
  }

// ChatGPT function
std::vector<std::string> parse_csv_line(const std::string& line)
{
std::vector<std::string> fields;
std::stringstream line_stream(line);
std::string field;
while (getline(line_stream, field, ',')) {
  fields.push_back(field);
}
return fields;
}

template <int dim>
void ElasticProblem<dim>::read_bcs()
{
	std::cout << "Entering read bcs()" << std::endl;
//	std::ifstream csv_file("input/genfile.csv");
	std::ifstream csv_file("../aq-mpi-stress/input/cube-bcs.csv");

	std::vector<dealii::Point<dim>> bcpoints(0);
	std::vector< std::tuple<int,double> > dofmag(0);
	// Read the file line by line
	std::string line;
	while (getline(csv_file, line)) {
	// Parse each line and add the resulting fields to an array
    std::vector<std::string> fields = parse_csv_line(line);

 	const Point<dim> bcn(std::strtod(fields[0].c_str(), NULL) ,
			std::strtod(fields[1].c_str(), NULL), std::strtod(fields[2].c_str(), NULL) );

	bcpoints.push_back(bcn);

	// Add something asserting that the field has 5, 7, or 9 entries

	for (int i = 3; i < fields.size(); i++){
		if (i%2){
			auto ctup = std::make_tuple (std::stoi(fields[i].c_str(), NULL)
					,  std::strtod(fields[i+1].c_str(), NULL) );
			dofmag.push_back(ctup);
    		}
    	}
	}
}




template <int dim>
void ElasticProblem<dim>::nodal_bcs()
{
	std::cout << "Entered into nodal_bcs" << std::endl;
	// --------------------------------------
	//              Read the BCS from csv file
	// --------------------------------------
	std::ifstream csv_file("../aq-mpi-stress/input/cube-bcs.csv"); //nodalbc_filename
	std::vector<dealii::Point<dim>> bcpoints(0);  // vector of BC points
	std::vector< std::tuple<int,double> > dofmag(0);   // vector holding tuple of DOF and magnitude
	std::vector<std::vector< std::tuple<int,double>>> bcmags(0);   // vector holding tuple of DOF and magnitude
	std::vector<int> i_rmBC(0); // temp vector for removing BC's from main BC lists
	std::string line;    // read file line by line
	while (getline(csv_file, line))  //(csv_file
	{
		// Parse each line and add the resulting fields to an array
		std::vector<std::string> fields = parse_csv_line(line);
		// Add something asserting that the field has 5, 7, or 9 entries

		const Point<dim> bcn(std::strtod(fields[0].c_str(), NULL) ,
		std::strtod(fields[1].c_str(), NULL), std::strtod(fields[2].c_str(), NULL) );

		bcpoints.push_back(bcn);
		std::vector< std::tuple<int,double> > dofmag(0);   // vector holding tuple of DOF and magnitude

		for (int i = 3; i < fields.size(); i++){
			if (i%2){
				auto ctup = std::make_tuple (std::stoi(fields[i].c_str(), NULL)
							,  std::strtod(fields[i+1].c_str(), NULL) );
				dofmag.push_back(ctup);
			}

		}
		bcmags.push_back(dofmag)        ;
	}

    int locdof;
	double locmag;
	// Set acceptable radius around node
	const double nrad = 0.001;  // Need a better way to make this

	// Print the starting number of BC point and mags
	std::cout << "# of BCs (pt,mag):  (" << bcpoints.size() << \
			", " << bcmags.size() << ")" << std::endl;

 for (const auto &cell : dof_handler.active_cell_iterators())
 {
   for (unsigned int v = 0; v<cell->n_vertices() ; v++)  // might be able to use 'vertex_iterator'
     {
       for (unsigned int i = 0; i < bcpoints.size(); i++)
       {
		   if (cell->vertex(v).distance(bcpoints[i]) < nrad)
		   {
			   for (auto t : bcmags[i])
			   {
			// remove BC from list after finding
			// check x, check y, check z, then check distance
			// look for library functions
			//               std::tie ( myint,  mydoub) = bcmags[1][0];
				   std::tie ( locdof,  locmag) = t;
				   if (locdof <= 2)
				   {
//					   std::cout << "DOF " << locdof << " = " << locmag << std::endl;
					   constraints.add_line(cell->vertex_dof_index(v,locdof, cell->active_fe_index() ));
					   if (locmag != 0)
					   {
						   constraints.set_inhomogeneity(cell->vertex_dof_index(v,locdof,
										   cell->active_fe_index()), locmag);
						   std::cout << "Non-zero BC of " << locmag << " applied to DOF "
								   << locdof << std::endl;
					   }
				   }
			   }
		   // Although a BC has been matched to the vertex, we can't stop looping
		   // because their might be another BC that matches.
		   // However, we need to mark this BC for deletion

			   i_rmBC.push_back(i);  // list of BCs to remove
		   }
	   }
       // Now that this vertex is done, remove any BCs from the point and magnitude vectors
       // Then clear the list of indices to be removed.  Reprint the number of BCs
       if (!i_rmBC.empty())
       {
    	   if (i_rmBC.size() > 10)
    	   {
    		   std::cout << "uh oh" << std::endl;
    	   }
    	   for (auto irm : i_rmBC)
    	   {
//    		   auto i = std::find(begin(bcpoints), end(bcpoints), bcpoints[irm]);
    		   bcpoints.erase (std::find(begin(bcpoints), end(bcpoints), bcpoints[irm]));
//    		   auto j = std::find(begin(bcmags), end(bcmags), bcmags[irm]);
			   bcmags.erase (std::find(begin(bcmags), end(bcmags), bcmags[irm]));
    	   }
    	   i_rmBC.clear();
//    		std::cout << "# of BCs (pt,mag):  (" << bcpoints.size() << \
//    				", " << bcmags.size() << ")" << std::endl;
       }
     }
 }
 std::cout << "Final BC count:" << std::endl;
 std::cout << "# of unresolved BCs (pt,mag):  (" << bcpoints.size() << \
     				", " << bcmags.size() << ")" << std::endl;

}








// Original function
//template <int dim>
//void ElasticProblem<dim>::nodal_bcs()
//{
//// --------------------------------------
//// 		Read the BCS from csv file
//// --------------------------------------
//	std::ifstream csv_file("input/file.csv");
//	std::vector<dealii::Point<dim>> bcpoints(0);  // vector of BC points
//	std::vector< std::tuple<int,double> > dofmag(0);   // vector holding tuple of DOF and magnitude
//	std::vector<std::vector< std::tuple<int,double> > > bcmags(0);   // vector holding tuple of DOF and magnitude
//	std::string line;    // read file line by line
//	while (getline(csv_file, line)) {
//	// Parse each line and add the resulting fields to an array
//		std::vector<std::string> fields = parse_csv_line(line);
//		// Add something asserting that the field has 5, 7, or 9 entries
//
//		const Point<dim> bcn(std::strtod(fields[0].c_str(), NULL) ,
//			std::strtod(fields[1].c_str(), NULL), std::strtod(fields[2].c_str(), NULL) );
//
//		bcpoints.push_back(bcn);
//		std::vector< std::tuple<int,double> > dofmag(0);   // vector holding tuple of DOF and magnitude
//
//		for (int i = 3; i < fields.size(); i++){
//			if (i%2){
//				auto ctup = std::make_tuple (std::stoi(fields[i].c_str(), NULL)
//						,  std::strtod(fields[i+1].c_str(), NULL) );
//				dofmag.push_back(ctup);
//    		}
//
//    	}
//		bcmags.push_back(dofmag)	;
//	}
//
//	int locdof;
//	double locmag;
//	// Set acceptable radius around node
//	const double nrad = 0.001;
//
//    for (const auto &cell : dof_handler.active_cell_iterators())
//    {
//      for (unsigned int v = 0; v<cell->n_vertices() ; v++)  // might be able to use 'vertex_iterator'
//    	{
//    	  for (int i = 0; i < bcpoints.size(); i++)
//    	  {
//    		  if (cell->vertex(v).distance(bcpoints[i]) < nrad)
//    		  {
//    			  for (auto t : bcmags[i])
//    			  {
//// remove BC from list after finding
//// check x, check y, check z, then check distance
//// look for library functions
//    	//		  std::tie ( myint,  mydoub) = bcmags[1][0];
//				  std::tie ( locdof,  locmag) = t;
//
//				  std::cout << "DOF " << locdof << " = " << locmag << std::endl;
//				  constraints.add_line(cell->vertex_dof_index(v,locdof, cell->active_fe_index() ));
//
//				  if (locmag != 0)
//				  	  {
//					  constraints.set_inhomogeneity(cell->vertex_dof_index(v,locdof,
//							  cell->active_fe_index()), locmag);
//					  std::cout << "Non-zero BC applied" << std::endl;
//				  	  }
//    			  }
//    		  }
//    	  }
//
//
////    	  if (cell->vertex(v).distance(proot1) < 0.001)   // This mess should be reduced once I get this working
////	    {
////	      constraints.add_line(cell->vertex_dof_index(v,0, cell->active_fe_index() ));
////	      constraints.add_line(cell->vertex_dof_index(v,1, cell->active_fe_index() ));
////	      constraints.add_line(cell->vertex_dof_index(v,2, cell->active_fe_index() ));
////	      std::cout << "Root 1" << std::endl;
////	    }
//
//
//    	}
//    }
//}

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
  std::ofstream output("cube-solution.vtu");
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
//  const std::string in_mesh_filename = "input/Mesh.inp";
//  const std::string in_mesh_filename = "input/Flap_aq.msh";
  const std::string in_mesh_filename = "../aq-mpi-stress/input/tet-cube.msh";

  ElasticProblem<3> stran_3d(in_mesh_filename);                                                                                                 
  stran_3d.run();                                                                                                                              
}   

