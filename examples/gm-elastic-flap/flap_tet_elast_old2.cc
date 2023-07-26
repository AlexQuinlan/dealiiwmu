#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
// These next two are from step-18
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/conditional_ostream.h>

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
#include <algorithm>
#include <deal.II/base/timer.h>
#include <ctime>   // different timer for debugging and optimization
// Include files that contain appropriate quadrature rules, finite elements, and mapping objects for simplex meshes.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
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
// These are added for step-18

#include <deal.II/base/utilities.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/physics/transformations.h>


#include <fstream>                                                                        
#include <iostream>
#include <iomanip>


#include <cmath>                                                                      
#include <string>    

namespace GM
{
using namespace dealii;
 

  template <int dim>
  struct PointHistory
    {
     SymmetricTensor<2,dim> old_stress;
    };

  template <int dim>
  SymmetricTensor<4,dim> get_stress_strain_tensor(const double lambda,
						  const double mu)
  {
    SymmetricTensor<4, dim> tmp;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        for (unsigned int k = 0; k < dim; ++k)
          for (unsigned int l = 0; l < dim; ++l)
            tmp[i][j][k][l] = (((i == k) && (j == l) ? mu : 0.0) +
                               ((i == l) && (j == k) ? mu : 0.0) +
                               ((i == j) && (k == l) ? lambda : 0.0));
    return tmp;
  }

  template <int dim>
  inline SymmetricTensor<2, dim> get_strain(const FEValues<dim> &fe_values,
                                            const unsigned int   shape_func,
                                            const unsigned int   q_point)
  {
    SymmetricTensor<2, dim> tmp;
    for (unsigned int i = 0; i < dim; ++i)
      tmp[i][i] = fe_values.shape_grad_component(shape_func, q_point, i)[i];

    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = i + 1; j < dim; ++j)
        tmp[i][j] =
          (fe_values.shape_grad_component(shape_func, q_point, i)[j] +
           fe_values.shape_grad_component(shape_func, q_point, j)[i]) /
          2;

    return tmp;  // temp is the 6 strain values [true or engineering strain?] AQ
  }

  template <int dim>
  inline SymmetricTensor<2, dim>
  get_strain(const std::vector<Tensor<1, dim>> &grad)
  {
    Assert(grad.size() == dim, ExcInternalError());
    SymmetricTensor<2, dim> strain;
    for (unsigned int i = 0; i < dim; ++i)
      strain[i][i] = grad[i][i];

    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = i + 1; j < dim; ++j)
        strain[i][j] = (grad[i][j] + grad[j][i]) / 2;

    return strain;
  }

  Tensor<2, 2> get_rotation_matrix(const std::vector<Tensor<1, 2>> &grad_u)
  {
    const double curl = (grad_u[1][0] - grad_u[0][1]);
    const double angle = std::atan(curl);
    return Physics::Transformations::Rotations::rotation_matrix_2d(-angle);
  }
    
  Tensor<2, 3> get_rotation_matrix(const std::vector<Tensor<1, 3>> &grad_u)
  {
    const Tensor<1, 3> curl({grad_u[2][1] - grad_u[1][2],
                             grad_u[0][2] - grad_u[2][0],
                             grad_u[1][0] - grad_u[0][1]});
    const double tan_angle = std::sqrt(curl * curl);
    const double angle     = std::atan(tan_angle);
    if (std::abs(angle) < 1e-9)
      {
        static const double rotation[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        static const Tensor<2, 3> rot(rotation);
        return rot;
      }
    const Tensor<1, 3> axis = curl / tan_angle;
    return Physics::Transformations::Rotations::rotation_matrix_3d(axis,
                                                                   -angle);
  }



template <int dim>
class tetElast
{
public:
  tetElast<dim>(
      const std::string &  initial_mesh_filename,
	  const std::string &  nodalbc_filename,
      const std::string &  output_filename);
  void run_mesh();
  void run();
 
private:

//  void make_grid();
  unsigned int nodal_bcs();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;
  void read_domain();
  void output_mesh();
  void read_bcs();
 
  void setup_quadrature_point_history();
  void update_quadrature_point_history();

  parallel::shared::Triangulation<dim> triangulation;
  const FESystem<dim>   fe;
  DoFHandler<dim> dof_handler;
  const QGaussSimplex<dim> quadrature_formula;
  
  // Here, we select a mapping object, a finite element, and a quadrature rule
  // that are compatible with simplex meshes.
  const MappingFE<dim>     mapping;
  //  const FE_SimplexP<dim>   fe;
  

  AffineConstraints<double> constraints;
  // ** Triangulation<dim> triangulation;

  std::vector<PointHistory<dim>> quadrature_point_history;
  PETScWrappers::MPI::SparseMatrix system_matrix;
  PETScWrappers::MPI::Vector system_rhs;
  Vector<double> incremental_displacement;  // Not sure if I'll need this for the static job

  //  SparsityPattern      sparsity_pattern;  // A dynamic sparsity patter is introduced in setup
  // ** SparseMatrix<double> system_matrix;

  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  ConditionalOStream pcout;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  static const SymmetricTensor<4, dim> stress_strain_tensor;
 
  Vector<double> solution;
  // ** Vector<double> system_rhs;

  const std::string initial_mesh_filename;
  const std::string nodalbc_filename;
  const std::string output_filename; 
};
 
 
// @sect4{tetElast::tetElast}
//
// In the constructor, we set the polynomial degree of the finite element and
// the number of quadrature points. Furthermore, we initialize the MappingFE
// object with a (linear) FE_SimplexP object so that it can work on simplex
// meshes.

// class BodyForce
  template <int dim>
  class BodyForce : public Function<dim>
  {
  public:
    BodyForce();

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  values) const override;

    virtual void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  value_list) const override;
  };

  template <int dim>
  BodyForce<dim>::BodyForce()
    : Function<dim>(dim)
  {}


  template <int dim>
  inline void BodyForce<dim>::vector_value(const Point<dim> & /*p*/,
                                           Vector<double> &values) const
  {
    AssertDimension(values.size(), dim);
    const double g   = 9.81;
    const double rho = 7700;
    values          = 0;
    values(dim - 1) = -rho * g;
  }


  template <int dim>
  void BodyForce<dim>::vector_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>> &  value_list) const
  {
    const unsigned int n_points = points.size();

    AssertDimension(value_list.size(), n_points);

    for (unsigned int p = 0; p < n_points; ++p)
      BodyForce<dim>::vector_value(points[p], value_list[p]);
  }


   template <int dim>
   const SymmetricTensor<4, dim> tetElast<dim>::stress_strain_tensor =
   get_stress_strain_tensor<dim>(/*lambda = */ 9.695e10,
                                  /*mu     = */ 7.617e10);



  template <int dim>     
tetElast<dim>::tetElast(
    const std::string &  initial_mesh_filename,
	const std::string &  nodalbc_filename,
    const std::string &  output_filename)
  : triangulation(MPI_COMM_WORLD)
    , fe(FE_SimplexP<dim>(2),dim)    //    , fe(FE_Q<dim>(1), dim)
    , dof_handler(triangulation)
    , quadrature_formula(fe.degree + 1)
    // , present_time(0.0)
    // , present_timestep(1.0)
    // , end_time(10.0)
    // , timestep_no(0) 
    , initial_mesh_filename(initial_mesh_filename)
    , nodalbc_filename(nodalbc_filename)
    , output_filename(output_filename) 
    , mapping(FE_SimplexP<dim>(1))
    , mpi_communicator(MPI_COMM_WORLD)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
    , pcout(std::cout, this_mpi_process == 0)
  //
  // initial_mesh_filename(initial_mesh_filename)
  // , output_filename(output_filename) 
  // , mapping(FE_SimplexP<dim>(1))

    //  , quadrature_formula(3)
    //, dof_handler(triangulation)
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

//template <int dim>
//void tetElast<dim>::make_grid()
//{
//  GridIn<dim>(triangulation).read("input/Flap_aq.msh"); //initial_mesh_filename
//  pcout << "Number of active cells: " << triangulation.n_active_cells()
//            << std::endl;
//}

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


// ==============================================================
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
void tetElast<dim>::read_bcs()
{

//	std::ifstream csv_file("input/genfile.csv");
	std::ifstream csv_file(nodalbc_filename);
	std::vector<dealii::Point<dim>> bcpoints(0);
	std::vector< std::tuple<int,double> > dofmag(0);
	// Read the file line by line
	std::string line;
	while (getline(csv_file, line))
	{
        // Parse each line and add the resulting fields to an array
		std::vector<std::string> fields = parse_csv_line(line);

        const Point<dim> bcn(std::strtod(fields[0].c_str(), NULL) ,
                        std::strtod(fields[1].c_str(), NULL), std::strtod(fields[2].c_str(), NULL) );

        bcpoints.push_back(bcn);

        // Add something asserting that the field has 5, 7, or 9 entries

        for (int i = 3; i < fields.size(); i++)
		{
			if (i%2)
			{
				auto ctup = std::make_tuple (std::stoi(fields[i].c_str(), NULL)
								,  std::strtod(fields[i+1].c_str(), NULL) );
				dofmag.push_back(ctup);
			}
        }
	}
}


template <int dim>
unsigned int tetElast<dim>::nodal_bcs()
{
	pcout << "Entered into nodal_bcs" << std::endl;
	// --------------------------------------
	//              Read the BCS from csv file
	// --------------------------------------
	std::ifstream csv_file(nodalbc_filename);
	std::vector<dealii::Point<dim>> bcpoints(0);  // vector of BC points
//	std::vector< std::tuple<const Point<dim>,int,double> > dofmag(0);   // vector holding tuple of DOF and magnitude
	std::vector<std::vector< std::tuple<const Point<dim>, int,double>>> nbcs(0);   // vector holding tuple of DOF and magnitude
	std::vector<int> i_rmBC(0); // temp vector for removing BC's from main BC lists
	std::string line;    // read file line by line
	while (getline(csv_file, line))
	{
		// Parse each line and add the resulting fields to an array
		std::vector<std::string> fields = parse_csv_line(line);
		// Add something asserting that the field has 5, 7, or 9 entries

		const Point<dim> bcn(std::strtod(fields[0].c_str(), NULL) ,
		std::strtod(fields[1].c_str(), NULL), std::strtod(fields[2].c_str(), NULL) );

		bcpoints.push_back(bcn);
		std::vector< std::tuple<const Point<dim>, int,double> > tmpPt(0);   // vector holding tuple of DOF and magnitude

		for (int i = 3; i < fields.size(); i++){
			if (i%2){
				auto ctup = std::make_tuple (bcn, std::stoi(fields[i].c_str(), NULL)
							,  std::strtod(fields[i+1].c_str(), NULL) );
				tmpPt.push_back(ctup);
			}
		}

		nbcs.push_back(tmpPt)        ;
	}

    int locdof;
	double locmag;
	Point<dim> locpt;
	// Set acceptable radius around node
	const double nrad = 0.01;  // Need a better way to make this.  Maybe tie to bcpoints[0].norm()

	// Print the starting number of BC point and mags
	pcout << "# of BCs (pt,mag):  (" << bcpoints.size() << \
			", " << nbcs.size() << ")" << std::endl;

 for (const auto &cell : dof_handler.active_cell_iterators())
 {
   for (unsigned int v = 0; v<cell->n_vertices() ; v++)  // might be able to use 'vertex_iterator'
     {
       for (unsigned int i = 0; i < bcpoints.size(); i++)
       {
		   if (cell->vertex(v).distance(bcpoints[i]) < nrad)
		   {
			   for (auto t : nbcs[i])
			   {
			// remove BC from list after finding
			// check x, check y, check z, then check distance
			// look for library functions
			//               std::tie ( myint,  mydoub) = bcmags[1][0];
				   std::tie (locpt, locdof,  locmag) = t;
				   if (locdof <= 2)
				   {
//					   pcout << "DOF " << locdof << " = " << locmag << std::endl;
					   constraints.add_line(cell->vertex_dof_index(v,locdof, cell->active_fe_index() ));
					   if (locmag != 0)
					   {

						   constraints.set_inhomogeneity(cell->vertex_dof_index(v,locdof,
										   cell->active_fe_index()), locmag);
						   pcout << "Non-zero BC of " << locmag << " applied to DOF.   "
								   << locdof << std::flush;
					   }
					   pcout << ".    BC number " << i << " of " << bcpoints.size() << " matched." << std::endl;

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
    		   pcout << "uh oh" << std::endl;
    	   }
    	   for (auto irm : i_rmBC)
    	   {
//    		   auto ii = std::find(begin(bcpoints), end(bcpoints), bcpoints[irm]);
//    		   bcmags.erase (bcmags(ii));
//    		   bcpoints.erase (bcpoints[ii]);

//    		   bcmags.erase (std::find(begin(bcpoints), end(bcpoints), bcpoints[irm]))
//    		   std::vector<dealii::Point<dim>> tmpPt = bcpoints[irm]
    		   bcpoints.erase (std::find(begin(bcpoints), end(bcpoints), bcpoints[irm]));
////    		   auto j = std::find(begin(bcmags), end(bcmags), bcmags[irm]);
//    		   std::vector<std::vector< std::tuple<int,double>>> tmpM = bcmags[irm];
//			   bcmags.erase (std::find(begin(tmpM), end(tmpM), bcmags[irm]));

			   nbcs.erase (std::find(begin(nbcs), end(nbcs), nbcs[irm]));
    	   }
    	   i_rmBC.clear();
//    		pcout << "# of BCs (pt,mag):  (" << bcpoints.size() << \
//    				", " << bcmags.size() << ")" << std::endl;
       }
     }
 }
 pcout << std::endl;
 pcout << "Final BC count:" << std::endl;
 pcout << "# of unresolved BCs (pt,mag):  (" << bcpoints.size() << \
     				", " << nbcs.size() << ")" << std::endl;

 return nbcs.size();

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
void tetElast<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);


  
  // Not sure where to add this setup_quad_pt_hist.  Came from step18 create_coarse_grid
  setup_quadrature_point_history();

  // From step 18 -----------
  DynamicSparsityPattern sparsity_pattern(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler,
				  sparsity_pattern,
				  constraints,
				  false); //,
				  /*hanging_node_constraints,*/
                                  /*keep constrained dofs*/ // false);
  SparsityTools::distribute_sparsity_pattern(sparsity_pattern,
                                               locally_owned_dofs,
                                               mpi_communicator,
                                               locally_relevant_dofs);

  system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         sparsity_pattern,
                         mpi_communicator);

  // ------------------

    
  // solution.reinit(dof_handler.n_dofs());    // Original, replaced by step 18 MPI
  //  system_rhs.reinit(dof_handler.n_dofs());   // Not sure if this is replaced by incrementat_displacement

  system_rhs.reinit(locally_owned_dofs, mpi_communicator);
  incremental_displacement.reinit(dof_handler.n_dofs());
  
  constraints.clear();
  
  unsigned int missed_bcs =  nodal_bcs();
  Assert(missed_bcs == 0, ExcInternalError());


//  int bcfs = 0;

//  pcout << "Boundary condition faces" << std::endl;
//  pcout << bcfs << std::endl;
//  for (const auto &cell : triangulation.active_cell_iterators())
//  {
//    for (const auto &face : cell->face_iterators())
//    {
//    	if (face->at_boundary())
//    	{
//    		if (  face->center()[2] < 1225.0)
//    			face->set_boundary_id(1);
//    	}
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

  // constraints.close();		  
  // DynamicSparsityPattern dsp(dof_handler.n_dofs());
  // DoFTools::make_sparsity_pattern(dof_handler, dsp);
  // sparsity_pattern.copy_from(dsp);
  // system_matrix.reinit(sparsity_pattern);
  
}
 
 
// @sect4{tetElast::assemble_system}
//
// Nothing has changed here.
template <int dim>     
void tetElast<dim>::assemble_system()
{
	//  system_rhs    = 0;  // from 18
	//  system_matrix = 0;  // from 18

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


	  // From 18, saving for later
	   BodyForce<dim>              body_force;
	   std::vector<Vector<double>> body_force_values(n_q_points,
	                                                   Vector<double>(dim));



	  std::vector<double> lambda_values(n_q_points);   // w33  can this be removed??
	  std::vector<double> mu_values(n_q_points);
	  Functions::ConstantFunction<dim> lambda(96.95e9), mu(76.17e9);
	  std::vector<Tensor<1,dim>> rhs_values(n_q_points);  // this is different in Step-17

	//  // This is needed for point loads
	  int num_ptLoads = 0;  //
	  Vector<double>     ptld_rhs(num_ptLoads);
	  std::vector<types::global_dof_index> ptld_idx(num_ptLoads);
	  const Point<dim> load1(1.0, 1.0, 1.0);
	  int matchver = 0; // used to trigger the application of nodal forces


	  const FEValuesExtractors::Scalar          z_component(dim - 1);
	  std::map<types::global_dof_index, double> boundary_values;
	  // Newly commented.  Now moved up before loads applied
	//  VectorTools::interpolate_boundary_values(dof_handler,
	//                                           0,
	//                                           Functions::ZeroFunction<dim>(dim),
	//                                           boundary_values);


	  // =======================================
	  // ========= From cadex ==================
	  // =======================================
		pcout << "Starting Matrix assembly" <<std::endl;

	  for (const auto &cell : dof_handler.active_cell_iterators())
	    if (cell->is_locally_owned()) // Added from S18
	    {
		cell_matrix=0;
		cell_rhs   =0;
		fe_values.reinit(cell);

		// Get values for lambda, mu, and rhs   // Maybe not needed now
		 lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
		 mu.value_list(fe_values.get_quadrature_points(), mu_values);
	//	right_hand_side(fe_values.get_quadrature_points(), rhs_values);

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

			    // Added from S18  ----------------  // This method seems to take a little bit longer
			    const SymmetricTensor<2, dim>
	                    eps_phi_i = get_strain(fe_values, i, q_point),
	                    eps_phi_j = get_strain(fe_values, j, q_point);

	                  cell_matrix(i, j) += (eps_phi_i *            //
	                                        stress_strain_tensor * //
	                                        eps_phi_j              //
	                                        ) *                    //
	                                       fe_values.JxW(q_point); //
			    // End added from S18 --------------


			  // original
	//  		    cell_matrix(i,j) +=
	//
	//  		      (
	//  		       (fe_values.shape_grad(i, q_point)[component_i] *
	//  			fe_values.shape_grad(j, q_point)[component_j] *
	//  			lambda_values[q_point])
	//  		       +
	//  		       (fe_values.shape_grad(i, q_point)[component_j] *
	//  			fe_values.shape_grad(j, q_point)[component_i] *
	//  			mu_values[q_point])
	//  		       +
	//  		       ((component_i == component_j) ?
	//  			(fe_values.shape_grad(i, q_point) *
	//  			 fe_values.shape_grad(j, q_point) *
	//  			 mu_values[q_point])
	//  			:
	//  			0) ) *
	//  		      fe_values.JxW(q_point);
			  }
		      }
		  }

		// From S18
		const PointHistory<dim> *local_quadrature_points_data =
	            reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
//		body_force.vector_value_list(fe_values.get_quadrature_points(),
//	                                        body_force_values);

		// Assembling RHS (from cadex)
		for (const unsigned int i : fe_values.dof_indices())
		  {
		    const unsigned int component_i =
		      fe.system_to_component_index(i).first;

		    for (const unsigned int q_point :
			   fe_values.quadrature_point_indices())
		      {
			const SymmetricTensor<2, dim> &old_stress =
	                    local_quadrature_points_data[q_point].old_stress;

//			const SymmetricTensor<2, dim> &old_strain =
//			                    local_quadrature_points_data[q_point].old_strain;
			// below is neglecting the body force
			cell_rhs(i) += ( fe_values.shape_value(i, q_point) *
					 rhs_values[q_point][component_i] -
					 old_stress * get_strain(fe_values, i, q_point)
					 ) * fe_values.JxW(q_point);

//			// This is from S-18
//	        cell_rhs(i) +=
//	          (body_force_values[q_point](component_i) *
//	             fe_values.shape_value(i, q_point) -
//	           old_stress * get_strain(fe_values, i, q_point)) *
//	          fe_values.JxW(q_point);

	// Merged 18 and cfrm  [could combine rhs_values, but leaving until debug done]
	//        cell_rhs(i) +=
	//                  (body_force_values[q_point](component_i) *
	//                     fe_values.shape_value(i, q_point) +
	//					 rhs_values[q_point][component_i] *
	//					 fe_values.shape_value(i, q_point) -
	//                   old_stress * get_strain(fe_values, i, q_point)) *
	//                  fe_values.JxW(q_point);

		      }
		  }


			// nothing happends to cell rhs

	//	cell->get_dof_indices(local_dof_indices);
	//	constraints.distribute_local_to_global(
	//					       cell_matrix, cell_rhs,
	//					       local_dof_indices,
	//					       system_matrix,
	//					       system_rhs);


	  ////// ***** Applied nodal forces *************
			if (num_ptLoads > 0)
			{
			   for (unsigned int v = 0; v<cell->n_vertices() ; v++)  // might be able to use 'vertex_iterator'
				   {
				   if (cell->vertex(v).distance(load1) < 0.001)
					  {
					  matchver = 1;
					  ptld_rhs(0) = 10000.02;   // assigns okay... but this could maybe be pre-assigned from file
					  ptld_idx[0] = local_dof_indices[v+0];
					  pcout << "Nodal Load added to constraints" << std::endl;
					  num_ptLoads = 0;  // stopgap for nodal load application
					  }
				   }
			}
	       // make the loads and the indices vectors
	       // Assign Point Force
	       if (matchver > 0)
	       {
	 		  constraints.distribute_local_to_global(
	 							   ptld_rhs,
	 							   ptld_idx,
	 							   system_rhs);
	 		  matchver = 0;
	       }
	// 	  pcout << "Nodal Loads applied to matrix" << std::endl;

		cell->get_dof_indices(local_dof_indices);


	//	pcout << cell_rhs.l1_norm() << " is cell_rhs l1_norm" << std::endl;
	//	pcout << cell_rhs.l2_norm() << " is cell_rhs l2_norm" << std::endl;
	//	pcout << cell_matrix.l1_norm() << " is cell_matrix l1_norm" << std::endl;
	//	pcout << cell_matrix.linfty_norm() << " is cell_matrix linfty_norm" << std::endl;
	//	pcout << cell_matrix.trace() << " is cell_matrix trace" << std::endl;

	//	constraints.distribute_local_to_global(cell_matrix,
	//										  local_dof_indices,
	//										  system_matrix);
		// for some reason, this goes to the same function as below.  Maybe
		// the overloaded function wasn't written for these args?  I don't see
		// it in affine_constraint


		// what exactly happens here
		constraints.distribute_local_to_global(cell_matrix,
											  cell_rhs,
											  local_dof_indices,
											  system_matrix,
											  system_rhs);


	    }
		    // Now compress the vector and the system matrix:  S18
	    system_matrix.compress(VectorOperation::add);
	    system_rhs.compress(VectorOperation::add);







	//     This is from S18, but I am not working with the time steps
	//     VectorTools::interpolate_boundary_values(
	//       dof_handler,
	//       1,
	//       IncrementalBoundaryValues<dim>(present_time, present_timestep),
	//       boundary_values,
	//       fe.component_mask(z_component));

	    PETScWrappers::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);
	        MatrixTools::apply_boundary_values(
	          boundary_values, system_matrix, tmp, system_rhs, false);
	    incremental_displacement = tmp;


}
 
// @sect4{tetElast::solve}
//
// Nothing has changed here.
template <int dim>     
void tetElast<dim>::solve()
{
    PETScWrappers::MPI::Vector distributed_incremental_displacement(
      locally_owned_dofs, mpi_communicator);
    distributed_incremental_displacement = incremental_displacement;

    pcout << "system_rhs.l2_norm = " << system_rhs.l2_norm() << std::endl;
//    pcout << "system_rhs.l2_norm not used.  Force to 1e-2" << std::endl;

    SolverControl solver_control(dof_handler.n_dofs(),
                                 1e-16 * system_rhs.l2_norm());  // no forces used right now


//    SolverControl solver_control(6,   1e-16 * system_rhs.l2_norm());


    PETScWrappers::SolverCG cg(solver_control, mpi_communicator);

    PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);

    cg.solve(system_matrix,
             distributed_incremental_displacement,
             system_rhs,
             preconditioner);

    incremental_displacement = distributed_incremental_displacement;

    // hanging_node_constraints.distribute(incremental_displacement);

    //    return solver_control.last_step();  // This was S18. We don't
    // have anything to return and the function returns void


  /// Original ------------

  // SolverControl            solver_control(50, 1e-3); //what are these controls?
  // SolverCG<Vector<double>> cg(solver_control);

  // PreconditionSSOR<SparseMatrix<double>> preconditioner;
  // preconditioner.initialize(system_matrix, 1.2);
  // cg.solve(system_matrix, solution, system_rhs, preconditioner);
   constraints.distribute(distributed_incremental_displacement);
}
 
 
// @sect4{tetElast::output_results}
//
// Nothing has changed here.
template <int dim>     
void tetElast<dim>::output_results() const
{
    pcout << "L928" << std::endl;
DataOut<dim> data_out;
//  DataOutBase::VtkFlags flags;
//  flags.write_higher_order_cells = true;
//  data_out.set_flags(flags);
pcout << "L933" << std::endl;
  data_out.attach_dof_handler(dof_handler);
  pcout << "L935" << std::endl;
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

    data_out.add_data_vector(incremental_displacement, solution_names);
    //    data_out.build_patches();

    pcout << "L958" << std::endl;
    // Norm of Stress from S18

    Vector<double> norm_of_stress(triangulation.n_active_cells());
    {
      // Loop over all the cells...
      for (auto &cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            // On these cells, add up the stresses over all quadrature
            // points...
            SymmetricTensor<2, dim> accumulated_stress;
            for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
              accumulated_stress +=
                reinterpret_cast<PointHistory<dim> *>(cell->user_pointer())[q]
                  .old_stress;

            // ...then write the norm of the average to their destination:
            norm_of_stress(cell->active_cell_index()) =
              (accumulated_stress / quadrature_formula.size()).norm();
          }
      else
          norm_of_stress(cell->active_cell_index()) = -1e+20;
    }

    pcout << "L983" << std::endl;


    data_out.add_data_vector(norm_of_stress, "norm_of_stress");
    std::vector<types::subdomain_id> partition_int(
      triangulation.n_active_cells());
    GridTools::get_subdomain_association(triangulation, partition_int);
    const Vector<double> partitioning(partition_int.begin(),
                                      partition_int.end());
    data_out.add_data_vector(partitioning, "partitioning");
    
    pcout << "L994" << std::endl;
  // data_out.attach_dof_handler(dof_handler);
  // data_out.add_data_vector(solution, "solution");
  data_out.build_patches(mapping, 2);
  // std::ofstream output("solution.vtu");
  // data_out.write_vtu(output);

  // The '2' in .build_patches is not the dimension.  I think it's the degree
  // Changed vtk to vtu

const std::string pvtu_filename = data_out.write_vtu_with_pvtu_record(
      "./", "solution", 7, mpi_communicator, 4);

     if (this_mpi_process == 0)
      {
        // Finally, we write the paraview record, that references all .pvtu
        // files and their respective time. Note that the variable
        // times_and_names is declared static, so it will retain the entries
        // from the previous timesteps.
        static std::vector<std::pair<double, std::string>> times_and_names;
        times_and_names.push_back(
          std::pair<double, std::string>(7, pvtu_filename));
        std::ofstream pvd_output("flaptet-solution.pvd");
        DataOutBase::write_pvd_record(pvd_output, times_and_names);
      }
  
  // std::ofstream output("flap-tetelast-bc2.vtu");
  // data_out.write_vtu(output);
}


  template <int dim>
  void tetElast<dim>::setup_quadrature_point_history()
  {
        triangulation.clear_user_data();
    {
      std::vector<PointHistory<dim>> tmp;
      quadrature_point_history.swap(tmp);
    }
    quadrature_point_history.resize(
      triangulation.n_locally_owned_active_cells() * quadrature_formula.size());
    unsigned int history_index = 0;
    for (auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->set_user_pointer(&quadrature_point_history[history_index]);
          history_index += quadrature_formula.size();
        }
        Assert(history_index == quadrature_point_history.size(),
           ExcInternalError());
  }

  template <int dim>
  void tetElast<dim>::update_quadrature_point_history()
  {
    // First, set up an <code>FEValues</code> object by which we will evaluate
    // the incremental displacements and the gradients thereof at the
    // quadrature points, together with a vector that will hold this
    // information:
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients);

    std::vector<std::vector<Tensor<1, dim>>> displacement_increment_grads(
      quadrature_formula.size(), std::vector<Tensor<1, dim>>(dim));

    // Then loop over all cells and do the job in the cells that belong to our
    // subdomain:
    for (auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          // Next, get a pointer to the quadrature point history data local to
          // the present cell, and, as a defensive measure, make sure that
          // this pointer is within the bounds of the global array:
          PointHistory<dim> *local_quadrature_points_history =
            reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
          Assert(local_quadrature_points_history >=
                   &quadrature_point_history.front(),
                 ExcInternalError());
          Assert(local_quadrature_points_history <=
                   &quadrature_point_history.back(),
                 ExcInternalError());

          // Then initialize the <code>FEValues</code> object on the present
          // cell, and extract the gradients of the displacement at the
          // quadrature points for later computation of the strains
          fe_values.reinit(cell);
          fe_values.get_function_gradients(incremental_displacement,
                                           displacement_increment_grads);
	  // Then loop over the quadrature points of this cell:
          for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
	    {
	    const SymmetricTensor<2, dim> new_stress =
                (local_quadrature_points_history[q].old_stress +
                 (stress_strain_tensor *
                  get_strain(displacement_increment_grads[q])));
	    const Tensor<2, dim> rotation =
                get_rotation_matrix(displacement_increment_grads[q]);
	    const SymmetricTensor<2, dim> rotated_new_stress =
                symmetrize(transpose(rotation) *
                           static_cast<Tensor<2, dim>>(new_stress) * rotation);
	    local_quadrature_points_history[q].old_stress =
                rotated_new_stress;
	    }
	}
  }



template <int dim>     
void tetElast<dim>::run()
{
  //  make_grid();
	clock_t runstart = clock();

    pcout << "Starting setup_system" << std::endl;
    pcout << (clock()-runstart)/CLOCKS_PER_SEC << std::endl;
    pcout << "---------------------" << std::endl;
  setup_system();
    pcout << "Complete: setup_system" << std::endl;
    pcout << (clock()-runstart)/CLOCKS_PER_SEC << " seconds" << std::endl;
    pcout << "---------------------" << std::endl;
    pcout << "Starting assemble_system" << std::endl;
  assemble_system();
    pcout << "Complete: assemble_system" << std::endl;
    pcout << (clock()-runstart)/CLOCKS_PER_SEC <<  " seconds" << std::endl;
    pcout << "---------------------" << std::endl;
    pcout << "Start Solver" << std::endl;
  solve();
    pcout << "Complete: solve" << std::endl;
    pcout << (clock()-runstart)/CLOCKS_PER_SEC << " seconds" <<  std::endl;
    pcout << "---------------------" << std::endl;
  output_results();
    pcout << "Complete: output_results" << std::endl;
    pcout << (clock()-runstart)/CLOCKS_PER_SEC <<  " seconds" << std::endl;
    pcout << "---------------------" << std::endl;

}
 

}  // Closing the namespace

// @sect3{The <code>main</code> function}
//
// Nothing has changed here.
int main(int argc, char **argv)
{
  using namespace GM;
  
  deallog.depth_console(2);
  Timer timer;

  const std::string in_mesh_filename = "input/Flap_aq.msh";
//  std::string in_mesh_filename = "input/partial_f	flap_aq.msh";
  const std::string nodalbc_filename = "input/simplebcs.csv";
//  const std::string nodalbc_filename = "../aq-mpi-stressinput/cube-bcs.csv";
//  const std::string in_mesh_filename = "../aq-mpi-stress/input/tet-cube.msh";
  const std::string out_mesh_filename = ("output/tet-flap_mesh.vtk");

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  tetElast<3> elastic3D(in_mesh_filename,nodalbc_filename,out_mesh_filename);
  elastic3D.run_mesh();
  elastic3D.run();

  std::cout << " ======= Done with program ========" << std::endl;
  timer.stop();

  std::cout << "Elapsed CPU time: " << timer.cpu_time() << " seconds.\n";
  std::cout << "Elapsed wall time: " << timer.wall_time() << " seconds.\n";

  timer.reset();
  return 0;
}
