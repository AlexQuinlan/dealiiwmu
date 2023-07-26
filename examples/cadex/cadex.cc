/* --------------------------
   Demo job by Alex Quinlan 
   -------------------------- */
// you can revert to the original_cadex.cc

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
#include <deal.II/fe/component_mask.h>

#include <fstream>                                                                        
#include <iostream>

#include <cmath>                                                                      
#include <string>                                                                     

// Set namespace
namespace cadex
{
  using namespace dealii;

  // Template allows for 2D or 3D without rewriting the code

  // Analysis Class
template <int dim>
  class ElasticProblem
  {
  public:
    enum ProjectionType                                                               
    {                                                                                 
      NormalProjection       = 0,                                                     
      DirectionalProjection  = 1,                                                     
      NormalToMeshProjection = 2                                                      
    };                                                                                
    
    ElasticProblem<dim>(
      const std::string &  initial_mesh_filename,                                     
      const std::string &  cad_file_name,                                             
      const std::string &  output_filename,                                           
      const ProjectionType surface_projection_kind = NormalProjection);               
    void run_mesh();
    void run();    
    
  private:
    void setup_system();
    void assemble_system();
    void solve();
    void output_results(const unsigned int cycle) const;
    void read_domain();                                                               
    void refine_mesh();                                                               
    void output_mesh(const unsigned int cycle);
    
    Triangulation<dim> tria;
    //    FE_Q<dim>          fe;  // this will change to FE system

    DoFHandler<dim>    dof_handler;
    FESystem<dim>      fe;
    AffineConstraints<double> constraints;
  
    SparsityPattern    sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double>   solution;
    Vector<double>   system_rhs;
                                                                                      
    const std::string initial_mesh_filename;                                          
    const std::string cad_file_name;                                                  
    const std::string output_filename;                                                
    const ProjectionType surface_projection_kind;                                     
  };

 template <int dim>  
    ElasticProblem<dim>::ElasticProblem(                                             
    const std::string &  initial_mesh_filename,                                       
    const std::string &  cad_file_name,                                               
    const std::string &  output_filename,                                             
    const ProjectionType surface_projection_kind)                                     
    : initial_mesh_filename(initial_mesh_filename)                                    
    , cad_file_name(cad_file_name)                                                    
    , output_filename(output_filename)                                                
    , surface_projection_kind(surface_projection_kind)
    , fe(FE_Q<dim>(1), dim)
    , dof_handler(tria)
  {}

template <int dim>
  void ElasticProblem<dim>::read_domain()                                              
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

    // Lets try extracting faces, edges, and vertices (might not be necessary AQ)
    std::vector<TopoDS_Face>     ex_faces;                                             
    std::vector<TopoDS_Edge>     ex_edges;                                             
    std::vector<TopoDS_Vertex>   ex_vertices;

    OpenCASCADE::extract_geometrical_shapes(
					    block_surface, ex_faces, ex_edges, ex_vertices);
					    

    std::ifstream in;                                                                 
    in.open(initial_mesh_filename);                                                   
    GridIn<3> gi;
    gi.attach_triangulation(tria);


    gi.read_abaqus(in, true);
//    gi.read_msh(in);
    output_mesh(0);      // We oapply_all_indicators_to_manifolds =utput this initial mesh saving it as the refinement step 0.

    // Triangulation<2, 3>::active_cell_iterator cell = tria.begin_active();
    // Triangulation<3>::active_cell_iterator cell = tria.begin_active();             
    // cell->set_manifold_id(1);                                                         
    
    unsigned int cellid = 0;
//    unsigned int faceid = 0;
//    unsigned int boundfaceid = 0;
    // for (const auto &face : cell->face_iterators())                                   
    //   {
    // 	faceid++;
    // 	face->set_manifold_id(1 + faceid);
    //   }

    for (const auto &cell : tria.active_cell_iterators())
      {
	cellid++;
	std::cout << "Cell " << cellid << ", Material " << cell->material_id() << std::endl;
	std::cout << "Cell " << cellid << ", Manifold " << cell->manifold_id() << std::endl;
//	for (const auto &face : cell->face_iterators())
//	  {
//	    faceid++;
//	    if (face->at_boundary())
//	      {
//		boundfaceid++;
//	//	face->set_all_manifold_ids(2);
//		if (face->manifold_id()==101 || face->manifold_id()==102)
//		std::cout << "Face " << faceid << ", Manifold " << face->manifold_id() << std::endl;
//	      }
//	  }
      }

    Assert(                                                                           
      wires.size() > 0,                                                               
      ExcMessage(                                                                     
        "I could not find any wire in the CAD file you gave me. Bailing out."));      
                                                                                      
    OpenCASCADE::ArclengthProjectionLineManifold<3,3> line_projector(
      wires[0], tolerance);
    
    tria.set_manifold(2, line_projector);
    
    OpenCASCADE::NormalProjectionManifold<3,3> normal_manifold(block_surface, tolerance);
    tria.set_manifold(2, normal_manifold);
    
  }
template <int dim>
  void ElasticProblem<dim>::refine_mesh()                                              
  {                                                                                   
    tria.refine_global(1);                                                            
  } 

template <int dim>
  void ElasticProblem<dim>::output_mesh(const unsigned int cycle)                   
  {                                                                                   
    const std::string filename =                                                      
      (output_filename + "_" + Utilities::int_to_string(cycle) + ".vtk");             
    std::ofstream logfile(filename);                                                  
    GridOut       grid_out;                                                           
    grid_out.write_vtk(tria, logfile);                                                
  }

template <int dim>
  void ElasticProblem<dim>::run_mesh()                                                      
  {                                                                                   
    read_domain();                                                                    
                                                                                      
    const unsigned int n_cycles = 1;                                                  
    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)                           
      {                                                                               
        refine_mesh();                                                                
        output_mesh(cycle + 1);                                                    
      }                                                                               
  }                                                                                   


// I need to get this part working

//   ------------------------------------------------------------
//   template <int dim>
//   class BoundaryValues : public Function<dim>
//   {
//   public:
//     virtual void value(const Point<dim> & p,
//   			 const unsigned int component = 0) const override;
//     // How can I set BCs for more than one component?
//   };
//
//   template <int dim>
//   void BoundaryValues<dim>::value(const Point<dim> &p,
//   				    const unsigned int component) const
//   {
////     for (unsigned int i=0; i < value.size(); ++i)
////       {
////   	if (p.square() < (0.1 * 0.1))
////   	  {
////   	  value[i][0]= 0.0;
////           value[i][1]= 0.0;
////   	  }
////       }
////	   value = 1;
////	   value[0] = 1;
////	   value[1] = 1;
////       return value;
//
//       // How does it work for points that do not have an applied BC?6
//   }

//  ---- From Step-22 -------------
template <int dim>
 class BoundaryValues : public Function<dim>
 {
 public:
	BoundaryValues()
	      : Function<dim>(dim)   // The (dim) here is super important for matching the BoundaryValue function
	    {}


    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  value) const override;

//   virtual double value(const Point<dim> & p,
//                        const unsigned int component = 0) const override;

//   virtual void value_list(const std::vector<Point<dim>> &points,
//                           std::vector<double> &          values,
//                           const unsigned int component = 0) const override;
 };

//template <int dim>
//double BoundaryValues<dim>::value(const Point<dim> & p,
//                                  const unsigned int component) const
//{
//  Assert(component == 0, ExcIndexRange(component, 0, 1));
//  (void)component;
//
//  // Set boundary to 1 if $x=1$, or if $x>0.5$ and $y=-1$.
//  if (std::fabs(p[0] - 1) < 1e-8 ||
//      (std::fabs(p[1] + 1) < 1e-8 && p[0] >= 0.5))
//    {
//      return 1.0;
//    }
//  else
//    {
//      return 0.0;
//    }
//}
template <int dim>
 double BoundaryValues<dim>::value(const Point<dim> & p,
                                   const unsigned int component) const
 {
   Assert(component < this->n_components,
          ExcIndexRange(component, 0, this->n_components));

   if (component == 2)
	   {
	   return 0.77;
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
//template <int dim>
//void BoundaryValues<dim>::value_list(const std::vector<Point<dim>> &points,
//                                     std::vector<double> &          values,
//                                     const unsigned int component) const
//{
//  AssertDimension(values.size(), points.size());
//
// // for (unsigned int i = 0; i < points.size(); ++i)
//	  values[0] = 1;
//  	  values[1] = 1;
//  	  values[2] = 1;
//
//}


  template <int dim>
  void right_hand_side(const std::vector<Point<dim>> &points,
  		       std::vector<Tensor<1, dim>> & values)
  {
    AssertDimension(values.size(), points.size());
    Assert(dim >= 2, ExcNotImplemented());

    // Force values
    // Point force at point 1
    Point<dim> point_1(5.0, 33.0, 3.0);

    
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

  //-------------------- Set-up System ----------------------------------
  template <int dim>
  void ElasticProblem<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    constraints.clear();


    std::cout << "Root Points Here." <<  std::endl;
    // Assign the face with y=0 to BoundaryID=1
    const Point<dim> proot1(0., 0.0209, -0.0101);
    const Point<dim> proot2(10., 0.0209, -0.0101);
    const Point<dim> proot3(0., 34.9272, 1.04056);
    const Point<dim> load1(10., 34.9272, 1.04056);  // single cell
//    const Point<dim> load1(9.1666, 34.3191, 1.23999);  // part of 4 cells


    for (const auto &cell : dof_handler.active_cell_iterators())
    {

    	for (unsigned int v = 0; v<cell->n_vertices() ; v++)  // might be able to use 'vertex_iterator'
    	{
    		if (cell->vertex(v).distance(proot1) < 0.001)
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
						  constraints.add_line(cell->vertex_dof_index(v,2, cell->active_fe_index() ));
						  constraints.set_inhomogeneity(cell->vertex_dof_index(v,2, cell->active_fe_index()), 0.5);
						  std::cout << "Load 1" << std::endl;
				  }

    	}

    /* for (const auto &face : cell->face_iterators())
//	if (face->at_boundary())
//	  {
//	    if (std::fabs(face->center()[1] - proot[1]) < 1e-6)
//	      face->set_boundary_id(1);
//	  }
      {
    	  if (face->manifold_id() == 201)
    	  {
    		  face->set_boundary_id(201);
       	  }
    	  else if (face->manifold_id() == 202)
    	  {
    		  face->set_boundary_id(202);
       	  }
      } */
    }

    /*  Original constraints
     *
    const FEValuesExtractors::Scalar   z_component(dim - 1);
    // Using the ZeroFunction to constrain displacement
    VectorTools::interpolate_boundary_values(dof_handler,
					     201,
					     Functions::ZeroFunction<dim>(dim),
					     constraints);

    VectorTools::interpolate_boundary_values(dof_handler,
					     202,
					     BoundaryValues<dim>(),
					     constraints,
						 fe.component_mask(z_component));

*/


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

    // Create a vector to hold the nodal loads and the dof indices
    int num_ptLoads = 2;
    Vector<double>     ptld_rhs(num_ptLoads);
    std::vector<types::global_dof_index> ptld_idx(num_ptLoads);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Perhaps these lambda and mu values could be calculated from E and nu
    std::vector<double> lambda_values(n_q_points);
    std::vector<double> mu_values(n_q_points);
    // supposedly the next line makes the previous 2 redundant
    Functions::ConstantFunction<dim> lambda(84.0), mu(84.0);

    std::vector<Tensor<1,dim>> rhs_values(n_q_points);
    std::vector<Tensor<1,dim>> rhs_nodal_values(8); // 8 vertices in hex element.  Need to make automatic




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
//	right_hand_side(cell->vertex  , rhs_nodal_values);  // try to get for nodes


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

//	const Point<dim> load1(10., 34.9272, 1.04056);  // single cell
	    const Point<dim> load1(9.1666, 34.3191, 1.23999);  // part of 4 cells
//    const Point<dim> load1( 0.0, 12.4891, 7.83696);
//    const Point<dim> load2(10.0, 12.4891, 7.83696);
//    const std::vector<int> loadDeg2;
//    const std::vector<double> loadMag2;
//    int nL_idx = 0;

//    loadDeg2.add(1, 2);
//    loadDeg2.add(1,2);
//    loadMag2 = 10.1;

//    Vector<types::global_dof_index> loadDoFindex;

	// Assembling RHS
	for (const unsigned int i : fe_values.dof_indices())   // fe_values has be reinitialized to the cell.
	  {
	    const unsigned int component_i =
	      fe.system_to_component_index(i).first;

	    for (const unsigned int q_point :
		   fe_values.quadrature_point_indices())
	      cell_rhs(i) += fe_values.shape_value(i, q_point) *
		             rhs_values[q_point][component_i] *
		             fe_values.JxW(q_point);

	  }


    // Point Force -------------
// It seems like this will be multiplied by the number of cells touching the node
    // Check vertex location
    // i is the dof... not the vertex
//	for (unsigned int v = 0; v<cell->n_vertices() ; v++)  // might be able to use 'vertex_iterator'
//		if (cell->vertex(v).distance(load2) < 0.001)
//			// Check direction
//			for (unsigned int i = v*3; v*3+3; v++)
//				if (fe.system_to_component_index(i).first == 2)
//					loadDoFindex =	i;



	cell->get_dof_indices(local_dof_indices);
	constraints.distribute_local_to_global(
					       cell_matrix, cell_rhs,
					       local_dof_indices,
					       system_matrix,
					       system_rhs);


	 // Nodal Forces AQ

	// We will need to read the magnitude of the force and the direction (0,1,2) from file
//	for (unsigned int v = 0; v<cell->n_vertices() ; v++)  // might be able to use 'vertex_iterator'
//		{
//		    	if (cell->vertex(v).distance(load1) < 0.001)
//				{
//		    		ptld_rhs(0) = 0.0;   // assigns okay... but this could maybe be pre-assigned from file
//		    		ptld_idx[0] = local_dof_indices[v+2];
//				}
//
////				if (cell->vertex(v).distance(load2) < 0.001)
////				{
////					ptld_rhs(1) = -33.33;   // assigns okay
////					ptld_idx[1] = local_dof_indices[v+2];  // offset of 2 for z direction
////				}
//		}

      }
   // make the loads and the indices vectors
   // Assign Point Force
//	constraints.distribute_local_to_global(
//					       ptld_rhs,
//					       ptld_idx,
//					       system_rhs);



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
    std::cout << "Here in the elastic run subroutine" << std::endl;
    //  This grid in stuff has been taken care of by the code
    //  copied from step-54
    
    // GridIn<dim> grid_in;
    // grid_in.attach_triangulation(tria);
    // std::ifstream input_file("input/demo_hmMesh.inp");
    // Assert(dim == 3, ExcInternalError());
    //    grid_in.read_abaqus(input_file); 

    for (unsigned int cycle=0; cycle < 1; ++cycle)
      {
    	std::cout << "Cycle " << cycle << ":" << std::endl;
    	// refinement code...
        refine_mesh();                                                                	
	//    	tria.refine_global(1); // I have a local function for this...
    	std::cout << "   Number of active cells:    "
    	          << tria.n_active_cells() << std::endl;
    	setup_system();
    	std::cout << "   Number of DOFs:            " << dof_handler.n_dofs()
    	          << std::endl;
    	assemble_system();
    	solve();
    	output_results(cycle);
         }
  }
}

// --------------------------------------	    
// Main Function
// --------------------------------------
int main()
{
  try
    {
      using namespace cadex;


      // First make the triangulation from CAD
      const std::string in_mesh_filename = "input/section_hmMesh.inp";
//      const std::string in_mesh_filename = "input/demo_hmMesh_mix.inp";
//      const std::string in_mesh_filename = "input/demo_solid.msh";
//      const std::string in_mesh_filename = "input/trial.msh";
//      const std::string in_mesh_filename = "input/tet.inp";
      const std::string cad_file_name    = "input/demo_solid.iges";
      const std::string out_mesh_filename = ("output/3d_mesh");

      ElasticProblem<3> abq_3d(in_mesh_filename,                           
                                          cad_file_name,                              
                                          out_mesh_filename,                          
                                          ElasticProblem<3>::NormalProjection);
      abq_3d.run_mesh();
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
	  clock_t finish_t = clock();

  std::cout << "Finished!"  << std::endl;


  return 0;

}
