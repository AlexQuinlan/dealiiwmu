/* --------------------------
   Demo job by Alex Quinlan 
   -------------------------- */


// Include files
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
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/grid/grid_generator.h>                                              
#include <deal.II/grid/grid_in.h>                                                     
#include <deal.II/grid/grid_out.h>
#include <deal.II/opencascade/manifold_lib.h>                                         
#include <deal.II/opencascade/utilities.h> 

#include <fstream>                                                                        
#include <iostream>

#include <cmath>                                                                      
#include <string>                                                                     


// Set namespace
namespace cadex
{
  using namespace dealii;

  // Template allows for 2D or 3D without rewriting the code
  template <int dim>
  // Analysis Class
  class ElasticProblem
  {
  public:
    ElasticProblem();
    void run();

  private:
    void setup_system();
    void assemble_system();
    void solve();
    void output_results(const unsigned int cycle) const;

    Triangulation<dim> triangulation;
    //    FE_Q<dim>          fe;  // this will change to FE system

    DoFHandler<dim>    dof_handler;
    FESystem<dim>      fe;
    AffineConstraints<double> constraints;
  
    SparsityPattern    sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double>   solution;
    Vector<double>   system_rhs;
  };

  /////////////
   class CADtriangu                                                            
  {                                                                                   
  public:                                                                             
    enum ProjectionType                                                               
    {                                                                                 
      NormalProjection       = 0,                                                     
      DirectionalProjection  = 1,                                                     
      NormalToMeshProjection = 2                                                      
    };                                                                                
                                                                                      
                                                                                      
    CADtriangu(                                                               
      const std::string &  initial_mesh_filename,                                     
      const std::string &  cad_file_name,                                             
      const std::string &  output_filename,                                           
      const ProjectionType surface_projection_kind = NormalProjection);               
                                                                                      
    void run();

    private:                                                                            
    void read_domain();                                                               
                                                                                      
    void refine_mesh();                                                               
                                                                                      
    void output_results(const unsigned int cycle);                                    
                                                                                      
    Triangulation<3> tria;
    //    Triangulation<2, 3> tria;                                                             
                                                                                      
    const std::string initial_mesh_filename;                                          
    const std::string cad_file_name;                                                  
    const std::string output_filename;                                                
                                                                                      
    const ProjectionType surface_projection_kind;                                     
  };

    CADtriangu::CADtriangu(                                             
    const std::string &  initial_mesh_filename,                                       
    const std::string &  cad_file_name,                                               
    const std::string &  output_filename,                                             
    const ProjectionType surface_projection_kind)                                     
    : initial_mesh_filename(initial_mesh_filename)                                    
    , cad_file_name(cad_file_name)                                                    
    , output_filename(output_filename)                                                
    , surface_projection_kind(surface_projection_kind)                                
  {}


  void CADtriangu::read_domain()                                              
  {                                                                                   
    TopoDS_Shape block_surface = OpenCASCADE::read_IGES(cad_file_name, 1);
    const double tolerance = OpenCASCADE::get_shape_tolerance(block_surface) * 5;       
    std::vector<TopoDS_Compound>  compounds;                                          
    std::vector<TopoDS_CompSolid> compsolids;                                         
    std::vector<TopoDS_Solid>     solids;                                             
    std::vector<TopoDS_Shell>     shells;                                             
    std::vector<TopoDS_Wire>      wires;

    OpenCASCADE::extract_compound_shapes(                                             
    block_surface, compounds, compsolids, solids, shells, wires);                     


    // Lets try extracting faces, edges, and vertices
    std::vector<TopoDS_Face>     ex_faces;                                             
    std::vector<TopoDS_Edge>     ex_edges;                                             
    std::vector<TopoDS_Vertex>   ex_vertices;

    OpenCASCADE::extract_geometrical_shapes(
					    block_surface, ex_faces, ex_edges, ex_vertices);
					    
    
    // The next few steps are more familiar, and allow us to import an existing       
    // mesh from an external VTK file, and convert it to a deal triangulation.        
    std::ifstream in;                                                                 


    // need to convert this to abaqus format
    in.open(initial_mesh_filename);                                                   
                                                                                      
    GridIn<3> gi;
    //        GridIn<2, 3> gi;                                                                  
    gi.attach_triangulation(tria);                                                    
    gi.read_abaqus(in);                                                                  
                                                                                      
    // We output this initial mesh saving it as the refinement step 0.                
    output_results(0);  


    ///
    //    Triangulation<2, 3>::active_cell_iterator cell = tria.begin_active();
    // Triangulation<3>::active_cell_iterator cell = tria.begin_active();             
    // cell->set_manifold_id(1);                                                         
    
    unsigned int cellid = 0;
    unsigned int faceid = 0;
    unsigned int boundfaceid = 0;
    // for (const auto &face : cell->face_iterators())                                   
    //   {
    // 	faceid++;
    // 	face->set_manifold_id(1 + faceid);
    //   }

    
    for (const auto &cell : tria.active_cell_iterators())
      {
	cellid++;
	cell->set_all_manifold_ids(1);                                                         
	for (const auto &face : cell->face_iterators())
	  {
	    faceid++;
	    if (face->at_boundary())
	      {	    
		//		if (std::fabs(face->center()[1] - proot[1]) < 1e-6)
		boundfaceid++;
		face->set_all_manifold_ids(2);
	      }                                      
	  }
      }





    
    Assert(                                                                           
      wires.size() > 0,                                                               
      ExcMessage(                                                                     
        "I could not find any wire in the CAD file you gave me. Bailing out."));      
                                                                                      
    //    OpenCASCADE::ArclengthProjectionLineManifold<2, 3> line_projector(
    OpenCASCADE::ArclengthProjectionLineManifold<3,3> line_projector(
      wires[0], tolerance);        // orignally wires[0]
    
    tria.set_manifold(2, line_projector);
    
    //    OpenCASCADE::NormalToMeshProjectionManifold<2, 3>
    //OpenCASCADE::NormalToMeshProjectionManifold<3,3>                               
    //        normal_to_mesh_projector(block_surface, tolerance);                       
    //    tria.set_manifold(1, normal_to_mesh_projector);

    OpenCASCADE::NormalProjectionManifold<3,3> normal_manifold(block_surface, tolerance);
    tria.set_manifold(2, normal_manifold);
      
							       

    
  }
  
  void CADtriangu::refine_mesh()                                              
  {                                                                                   
    tria.refine_global(1);                                                            
  } 



  void CADtriangu::output_results(const unsigned int cycle)                   
  {                                                                                   
    const std::string filename =                                                      
      (output_filename + "_" + Utilities::int_to_string(cycle) + ".vtk");             
    std::ofstream logfile(filename);                                                  
    GridOut       grid_out;                                                           
    grid_out.write_vtk(tria, logfile);                                                
  }

  void CADtriangu::run()                                                      
  {                                                                                   
    read_domain();                                                                    
                                                                                      
    const unsigned int n_cycles = 1;                                                  
    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)                           
      {                                                                               
        refine_mesh();                                                                
        output_results(cycle + 1);                                                    
      }                                                                               
  }                                                                                   

    
  // ------------------------------------------------------------
  // template <int dim>
  // class BoundaryValues : public Function<dim>
  // {
  // public:
  //   virtual double value(const Point<dim> & p,
  // 			 const unsigned int component = 0) const override;
  //   // How can I set BCs for more than one component?
  // };

  // template <int dim>
  // double BoundaryValues<dim>::value(const Point<dim> &p,
  // 				    const unsigned int component) const
  // {
  //   for (unsigned int i=0; i < value.size(); ++i)
  //     {
  // 	if (p.square() < (0.1 * 0.1))
  // 	  {
  // 	  value[i][0]= 0.0; 
  //         value[i][1]= 0.0;
  // 	  }
  //     }
  //     return value;

  //     // How does it work for points that do not have an applied BC?6
  // }

  template <int dim>
  void right_hand_side(const std::vector<Point<dim>> &points,
		       std::vector<Tensor<1, dim>> & values)
  {
    AssertDimension(values.size(), points.size());
    Assert(dim >= 2, ExcNotImplemented());

    // Force values
    // Point force at (1,1)
    Point<dim> point_1;
    point_1(0) = 5.0;
    point_1(1) = 33.0;    
    point_1(2) = 3.0;
    
    for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
      {
	if ( (points[point_n] - point_1).norm_square() < (0.5 * 0.5) )
	  values[point_n][2] = 1.0;
	else
	  values[point_n][2] = 0.0;

	// The force value of 1 and the radius 0.5 around the point
	// could both be made into variables

	// Also, I need to see what the zero means in values[][0]
	// I think it's direction 1, or 'x'
      }
  }

  //------------------------------------------------------------
  template <int dim>
  ElasticProblem<dim>::ElasticProblem()
    : dof_handler(triangulation)
    , fe(FE_Q<dim>(1), dim)
  {}

  //-------------------- Set-up System ----------------------------------
  template <int dim>
  void ElasticProblem<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    constraints.clear();

    // Assign the face with y=0 to BoundaryID=1
    const Point<dim> proot(0., 0.020919, 0.);
    for (const auto &cell : triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	if (face->at_boundary())
	  {
	    if (std::fabs(face->center()[1] - proot[1]) < 1e-6)
	      face->set_boundary_id(1);
	  }
  
    
    
    VectorTools::interpolate_boundary_values(dof_handler,
					     1,
					     Functions::ZeroFunction<dim>(dim),
					     constraints);
    constraints.close();
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs() );
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
    

    // Not sure where this should go... but I'll write it here for now
    //    FlatManifold<dim> manifold_description();

    // Above is the method from step 5.  Below is commented out, but is more related
    // to step 8
    /*
    // hanging node constraints omitted, since we're not planning to
    // have any in this example

    // I expected constraints to be dealt with in the .assemble_system() method
    constraints.clear();
    VectorTools::interpolate_boundary_values(dof_handler,
					     0,
					     Functions::ZeroFunction<dim>(dim),
					     constraints);
    // dof handler object used, component of boundary where boundary values should
    // be interpretted, boundary value function, output object

    // I think this is the BC application
    MatrixTools::apply_boundary_values(boundary_values,
				       system_matrix,
				       solution,
				       system_rhs);
    constraints.close();

    // ^^ is the zero the displacement?  Could this be changed to a non-zero value?
    // what is this ZeroFunction? per step-3, ZeroFunction applies zero 

    DynamicSparsityPattern dsp( dof_handler.n_dofs(), dof_handler.n_dofs() );
    DoFTools:make_sparsity_pattern(dof_handler,
				   dsp,
				   constraints,
				   false);

    // False refers to 'keep_constrained_dofs;
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);

    */
  }

  //------------------ Assemble System ------------------------------
  template <int dim>
  void ElasticProblem<dim>::assemble_system()
  {
    QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
			  quadrature_formula,
			  update_values | update_gradients |
			  update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points =    quadrature_formula.size();

    // This is the local matrix
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Perhaps these lambda and mu values could be calculated from E and nu
    std::vector<double> lambda_values(n_q_points);
    std::vector<double> mu_values(n_q_points);
    // supposedly the next line makes the previous 2 redundant
    Functions::ConstantFunction<dim> lambda(84.0), mu(84.0);

    std::vector<Tensor<1,dim>> rhs_values(n_q_points);

    // Loop over all the cells
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
	cell_matrix=0;
	cell_rhs   =0;

	fe_values.reinit(cell);

	// Get values for lambda, mu, and rhs
	lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
	mu.value_list(fe_values.get_quadrature_points(), mu_values);
	right_hand_side(fe_values.get_quadrature_points(), rhs_values);


	// need to lookup the C++ for loop syntax
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

	// Assembline RHS
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
  }

  // Solver --------------------------------------
  template <int dim>
  void ElasticProblem<dim>::solve()
  {
    SolverControl            solver_control(1000, 1e-12);
    SolverCG<Vector<double>> cg(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    cg.solve(system_matrix, solution, system_rhs, preconditioner);

    constraints.distribute(solution);
  }


  template <int dim>
  void ElasticProblem<dim>::output_results(const unsigned int cycle) const
  {
    DataOut<dim> data_out;
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
    data_out.build_patches();

    std::ofstream output("solution-" + std::to_string(cycle) + ".vtu");
    data_out.write_vtu(output);
  }
      
  template <int dim>
  void ElasticProblem<dim>::run()
  {
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation);
    std::ifstream input_file("input/demo_hmMesh.inp");
    Assert(dim == 3, ExcInternalError());
    grid_in.read_abaqus(input_file); 

    for (unsigned int cycle=0; cycle < 1; ++cycle)
      {
	std::cout << "Cycle " << cycle << ":" << std::endl;
	// refinement code...
	triangulation.refine_global(1);
	std::cout << "   Number of active cells:    "
	          << triangulation.n_active_cells() << std::endl;
	setup_system();
	std::cout << "   Number of DOFs:            " << dof_handler.n_dofs()
	          << std::endl;
	assemble_system();
	solve();
	output_results(cycle);
      }
  }
}
	    
// Main Function
int main()
{
  try
    {
      using namespace cadex;


      // First make the triangulation from CAD
      
      const std::string in_mesh_filename = "input/demo_hmMesh.inp";               
      const std::string cad_file_name    = "input/demo_solid.iges";
      const std::string out_mesh_filename = ("output/3d_mesh");

      CADtriangu cad_tria__norm(in_mesh_filename,                           
                                          cad_file_name,                              
                                          out_mesh_filename,                          
                                          CADtriangu::NormalProjection);
      cad_tria__norm.run();
      


      // Then try to the elastic stuff
      ElasticProblem<3> abq_3d;
        abq_3d.run();

      
      
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
		<< std::endl
		<< "----------------------------------------------------"
		<< std::endl;
      std::cerr << "Exception on processing: " << std::endl
		<< exc.what() << std::endl
		<< "Aborting!" << std::endl
		<< "----------------------------------------------------"
		<< std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
		<< "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;

}
