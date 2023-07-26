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
#include <deal.II/fe/fe_shell.h>
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
#include <deal.II/base/parameter_handler.h>
#include <deal.II/fe/fe_enriched.h>
#include <deal.II/fe/fe_nothing.h>
#include <fstream>                                                                        
#include <iostream>
#include <iomanip>
#include "../aqtools/shell_tools.h"

#include <cmath>                                                                      
#include <string>    

namespace shell
{
using namespace dealii;


//template <int dim>
//struct PointHistory
//  {
//   SymmetricTensor<2,dim> old_stress;
//   SymmetricTensor<2,dim> old_strain;
//
//  };

template <int dim, int spacedim>  //// From cube_hex
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

// Original
template <int dim, int spacedim>   //// From cube hex
inline SymmetricTensor<2, dim> get_strain(const FEValues<dim, spacedim> & fe_values,
                                          const unsigned int   shape_func,
                                          const unsigned int   q_point)
{
  SymmetricTensor<2, dim> tmp;
  double foo;
  for (unsigned int i = 0; i < dim; ++i)
  {
    tmp[i][i] = fe_values.shape_grad_component(shape_func, q_point, i)[i];
    foo = shell_tools::get_lagr_deriv(0, 0.211, 0.211 );
  }
  // shape functions are wrt to the global coordinate system
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = i + 1; j < dim; ++j)
      tmp[i][j] =
        (fe_values.shape_grad_component(shape_func, q_point, i)[j] +
         fe_values.shape_grad_component(shape_func, q_point, j)[i]) /
        2;

  return tmp;  // temp is the 6 strain values [true or engineering strain?] AQ
}

template <int dim, int spacedim>          //// from cube Hex
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


// -------------------------------------------------------
// -------------------------------------------------------
// -------------------------------------------------------
//template <int dim>
//class EnrichFunc : Function<dim>
//{
//public:
//	EnrichFunc(const Point<dim> &point,
//	        const unsigned int     &   component )
//	    : Function<dim>(1)
//	    , point(point)
//		, component(component)
//  {}
//	  virtual double
//	  value(const dealii::Point<dim> &point,
//	        const unsigned int        component = 0) const;
//
//	  virtual typename dealii::Tensor<1, dim, double>
//	  gradient(const dealii::Point<dim> &point,
//	        const unsigned int        component = 0) const;
//
//private:
//	const dealii::Point<dim> &point;
//	const unsigned int        component;
//};



// -------------------------------------------------------
// -------------------------------------------------------
// -------------------------------------------------------


template <int dim, int spacedim>
class shElast
{
public:
  shElast<dim,spacedim>(
      const std::string &  initial_mesh_filename,
      const std::string &  nodalbc_filename,
      const std::string &  output_filename);
  void run_mesh();
  void run();
//  static double enrichment_function();


private:

  void nodal_bcs();
  void setup_system();
  void assemble_system();
  unsigned int solve();
  void output_results() const;
  void read_domain();
  void output_mesh();
  void read_bcs();
  void setup_quadrature_point_history();
  void update_quadrature_point_history();


//  parallel::shared::Triangulation<dim> triangulation;
//  Triangulation<dim> triangulation;
//  Triangulation<dim,spacedim> triangulation;  // <dim, spacedim>
  Triangulation<dim> triangulation;  // <dim, spacedim>


  const FE_Enriched<dim>   fenr;
  const FESystem<dim,spacedim>   fe;

  DoFHandler<dim> dof_handler; //,spacedim
  const QGauss<dim> quadrature_formula;
//  const MappingFE<dim,spacedim>     mapping;
  const MappingFE<dim>     mapping;

  AffineConstraints<double> constraints;
//  std::vector<PointHistory<dim>> quadrature_point_history;
  SparsityPattern    sparsity_pattern;
  SparseMatrix<double> system_matrix;
//  PETScWrappers::MPI::SparseMatrix system_matrix;
//  PETScWrappers::MPI::Vector system_rhs;
  Vector<double> solution;  // This is the solution vector
  Vector<double> system_rhs;


//  MPI_Comm mpi_communicator;
//  const unsigned int n_mpi_processes;
//  const unsigned int this_mpi_process;
//  ConditionalOStream std::cout;
//  ParameterHandler & parameters;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  static const SymmetricTensor<4, dim> stress_strain_tensor;
  static const SymmetricTensor<4, spacedim> shell_stress_strain_tensor; // space
  static const Tensor<2,9> H_matrix; // this makes a 9x9 when I only need a 6x9
  static int H2[6][9];

  static const unsigned int ui3 = 3;

//  const FE_Q<dim> fe_base;
//  const FE_Q<dim> fe_enriched;
  WMUshellFunction<dim> wmuSh;



//  EnrichFunc<dim> enrichment;
//  EnrichmentFunction<dim> & enrichment;

  // Probably need to mod this for shell, so that z is related to x and y

  const std::string initial_mesh_filename;
  const std::string nodalbc_filename;
  const std::string output_filename;
};

  
template <int dim, int spacedim>   // from cube hex
const SymmetricTensor<4, dim> shElast<dim, spacedim>::stress_strain_tensor =
 get_stress_strain_tensor<dim, spacedim>(/*lambda = */ 84e9,
                               /*mu     = */ 84e9);
  

template <int dim, int spacedim>
shElast<dim, spacedim>::shElast(
    const std::string &  initial_mesh_filename,
    const std::string &  nodalbc_filename,
    const std::string &  output_filename)
 // : fe(FE_Shell<dim,spacedim>(1), 5) // 2nd arg changed to 6, # of elements AQ
    : fenr( FE_Q<dim>(1),
    	    FE_Q<dim>(1),
            &wmuSh)
	, fe(fenr, 6 )
	, dof_handler(triangulation)
    , quadrature_formula(fe.degree + 1)
    , initial_mesh_filename(initial_mesh_filename)
    , nodalbc_filename(nodalbc_filename)
	, mapping( FE_Q<dim>(1))
    , output_filename(output_filename)
//    ,  mapping(FE_Q<dim>(1),  FE_Q<dim>(1) ,EnrichmentFunction<dim>(Point<dim>(),
//            1.0,
//            2.5)  )   //MappingFE


{}


//template <int dim, int spacedim>
//void shElast<dim, spacedim>::enrichment_function( )
//{
//	std::cout << "enrichment function" << std::endl;
//}

//template <int dim, int spacedim>
//double shElast<dim, spacedim>::enrichment_function(  Triangulation<dim, spacedim>::cell_iterator  & cell )
//{
//	std::cout << "enrichment function" << std::endl;
//
//	return 2.0;
//}

template <int dim, int spacedim>
  void shElast<dim, spacedim>::read_domain()
  {
    std::ifstream in;
    in.open(initial_mesh_filename);
    GridIn<dim,spacedim> gi;
    gi.attach_triangulation(triangulation);
    //    gi.read_msh(in);  // if using GMSH input l
    gi.read_abaqus(in);  //

  }

template <int dim, int spacedim>
void shElast<dim, spacedim>::output_mesh()
  {
    std::ofstream logfile(output_filename);
//    GridOut       grid_out;
//    grid_out.write_vtu(triangulation, logfile);
  }

template <int dim, int spacedim>
  void shElast<dim, spacedim>::run_mesh()
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

template <int dim, int spacedim>
void shElast<dim, spacedim>::nodal_bcs()
{
  // Written by Alex Quinlan
  std::cout << "Entered into nodal_bcs" << std::endl;
  // --------------------------------------
  //              Read the BCS from csv file
  // --------------------------------------
  std::ifstream csv_file(nodalbc_filename);
  std::vector<dealii::Point<spacedim>> bcpoints(0);  // vector of BC points
  std::vector<std::vector< std::tuple<const Point<spacedim>, int,double>>> nbcs(0);   // vector holding tuple of DOF and magnitude
  std::vector<int> i_rmBC(0); // temp vector for removing BC's from main BC lists
  std::string line;    // read file line by line
  while (getline(csv_file, line))
    {
      // Parse each line and add the resulting fields to an array
      std::vector<std::string> fields = parse_csv_line(line);
      // Add something asserting that the field has 5, 7, or 9 entries

//      const Point<dim> bcn(std::strtod(fields[0].c_str(), NULL) ,
//                           std::strtod(fields[1].c_str(), NULL), std::strtod(fields[2].c_str(), NULL) );

      const Point<spacedim> bcn(std::strtod(fields[0].c_str(), NULL) ,
                            std::strtod(fields[1].c_str(), NULL),
							std::strtod(fields[2].c_str(), NULL));

      bcpoints.push_back(bcn);
      std::vector< std::tuple<const Point<spacedim>, int,double> > tmpPt(0);   // vector holding tuple of DOF and magnitude

      for (int i = 3; i < fields.size(); i++){
        if (i%2){
          auto ctup = std::make_tuple (bcn, std::stoi(fields[i].c_str(), NULL)
                                       ,  std::strtod(fields[i+1].c_str(), NULL) );
          tmpPt.push_back(ctup);
        }
      }

      nbcs.push_back(tmpPt);
    }
  int locdof;
   double locmag;
   Point<spacedim> locpt;
   // Set acceptable radius around node
   const double nrad = 0.001;  // Need a better way to make this.
   // Maybe tie to bcpoints[0].norm() or use the parameter file

   // Print the starting number of BC point and mags
   std::cout << "# of BCs (pt,mag):  (" << bcpoints.size() << \
     ", " << nbcs.size() << ")" << std::endl;

   for (const auto &cell : dof_handler.active_cell_iterators())
     {
    for (unsigned int v = 0; v<cell->n_vertices() ; v++)  // might be able to use 'vertex_iterator'
      {
        for (unsigned int i = 0; i < bcpoints.size(); i++)
        {
          if (cell->vertex(v).distance(bcpoints[i]) < nrad)  // Distance fails even when BCs are 2D
//        	if (cell->vertex(v)[0]        	 < 0.1)
            {
              for (auto t : nbcs[i])
                {
                  std::tie (locpt, locdof,  locmag) = t;
                  if (locdof <= 5)  // <= 2 for full 3D, 5 for shells with 3 rots
                    {
                      constraints.add_line(cell->vertex_dof_index(v,locdof, cell->active_fe_index() ));
                      if (locmag != 0 )
                        {
                          constraints.set_inhomogeneity(cell->vertex_dof_index(v,locdof,
                                                                               cell->active_fe_index()), locmag);
                          std::cout << "Non-zero BC of " << locmag << " applied to DOF   "
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
          for (auto irm : i_rmBC)
            {
              bcpoints.erase (std::find(begin(bcpoints), end(bcpoints), bcpoints[irm]));
              nbcs.erase (std::find(begin(nbcs), end(nbcs), nbcs[irm]));
            }
            i_rmBC.clear();
        }
      }
  }
  std::cout << std::endl;
  std::cout << "Final BC count:" << std::endl;
  std::cout << "# of unresolved BCs (pt,mag):  (" << bcpoints.size() << \
                                 ", " << nbcs.size() << ")" << std::endl;

 }

template <int dim, int spacedim>
void shElast<dim, spacedim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

//  setup_quadrature_point_history();
  constraints.clear();
  nodal_bcs();  // Applying the nodal constraints using WMU function

  constraints.close();


  // From step 18 -----------
  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  false); //,
  // 'false' means that we will never write into the matrix for the dofs corresponding
   // with the constraints


   sparsity_pattern.copy_from(dsp);
   system_matrix.reinit(sparsity_pattern);

   system_rhs.reinit(dof_handler.n_dofs()); //locally_owned_dofs, mpi_communicator);
   solution.reinit(dof_handler.n_dofs());

   std::ofstream out("output/sparsity_pattern1.svg");
   sparsity_pattern.print_svg(out);

 }

//-------- Assemble --------------
template <int dim, int spacedim>
void shElast<dim, spacedim>::assemble_system()
{
  system_rhs    = 0;  // from step-18
  system_matrix = 0;  // from step-18

  FEValues<dim, spacedim> fe_values(mapping,
                        fe,
                        quadrature_formula,
                        update_values | update_gradients |
                        update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points =    quadrature_formula.size();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);



  std::cout << "Starting Matrix assembly" <<std::endl;

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned()) // Added from S18
    {
        cell_matrix=0;
        cell_rhs   =0;
        for (unsigned int i = 0; i < 4 ; i++)
            wmuSh.set_corners(cell->vertex(i));
//        wmuSh.print_corners();
//        wmuSh.calc_corner_norms();
        fe_values.reinit(cell); // <--- here's the problem

//        wmuSh.set_curcell(cell);
//        wmuSh.set_magic(cell);
//        wmuSh.print_magic();

        for (const unsigned int i : fe_values.dof_indices())
          {
            const unsigned int component_i =
              fe.system_to_component_index(i).first;
            for (const unsigned int j : fe_values.dof_indices())
              {
                const unsigned int component_j =
                  fe.system_to_component_index(j).first;

                for (const unsigned int q_point : fe_values.quadrature_point_indices())
                  {

                    // Added from S18  ----------------  // This method seems to take a little bit longer
                    const SymmetricTensor<2, spacedim> eps_phi_i = get_strain(fe_values, i, q_point);
                    const SymmetricTensor<2, spacedim> eps_phi_j = get_strain(fe_values, j, q_point);
//                    So, I think these are the B matrix

                    // New AQ function to get shell jacobian at a certian point
//                    get_J(fe_values, q_point, 0.0 /*=zeta*/, 1.0 /*=thickness*/);



                  cell_matrix(i, j) += (eps_phi_i *            // B matrix?
                                        stress_strain_tensor * // Constitutive mtx
                                        eps_phi_j              // B matrix transpose?
                                        ) *                    //
                                       fe_values.JxW(q_point); //
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
                    {
              cell_rhs(i) += 0;
                    }
                 }
              cell->get_dof_indices(local_dof_indices);
               constraints.distribute_local_to_global(cell_matrix,
                                                      cell_rhs,
                                                      local_dof_indices,
                                                      system_matrix,
                                                      system_rhs);
    }
  
}




// ------------------- Solve -------------------
template <int dim, int spacedim>
unsigned int shElast<dim, spacedim>::solve()
{
//    SolverControl solver_control(dof_handler.n_dofs(),
//                                 1e-16 * system_rhs.l2_norm());
//
    SolverControl            solver_control(1000, 1e-12);
    SolverCG<Vector<double>> cg(solver_control);

    // Temporarily printing out the System Matrix
    std::ofstream outFile;
    outFile.open("output/SystemMatrix.txt");
    system_matrix.print_formatted(outFile, 5, true, 5, "z", 1.);
    outFile.close();

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    cg.solve(system_matrix, solution, system_rhs, preconditioner);

    constraints.distribute(solution);

    return 0;
}

template <int dim, int spacedim>
void shElast<dim, spacedim>::output_results() const
{
  DataOut<dim,spacedim> data_out;
  data_out.attach_dof_handler(dof_handler);

  std::vector<std::string> solution_names;


	solution_names.emplace_back("x_displacement");
	solution_names.emplace_back("y_displacement");
	solution_names.emplace_back("z_displacement");
	solution_names.emplace_back("rot1");
	solution_names.emplace_back("rot2");
	solution_names.emplace_back("rot3");

  data_out.add_data_vector(solution, solution_names);
  data_out.build_patches();
  std::ofstream output("solution.vtu");
  data_out.write_vtu(output);
}




template <int dim, int spacedim>
void shElast<dim, spacedim>::run()
{
  //  make_grid();
        clock_t runstart = clock();

        std::cout << std::endl;
        std::cout << "-----------------------------------" << std::endl;
        std::cout << std::endl;
        std::cout << "Input Mesh: " << initial_mesh_filename  << std::endl;
        std::cout << "Nodal BC File: " << nodalbc_filename  << std::endl;
        std::cout << "Output Mesh File: " << output_filename  << std::endl;

        std::cout << std::endl;
  
  
        std::cout << "Starting setup_system" << std::endl;
        std::cout << (clock()-runstart)/CLOCKS_PER_SEC << std::endl;
        std::cout << "---------------------" << std::endl;
      setup_system();
        std::cout << "Complete: setup_system" << std::endl;
        std::cout << (clock()-runstart)/CLOCKS_PER_SEC << " seconds" << std::endl;
        std::cout << "---------------------" << std::endl;
        std::cout << "Starting assemble_system" << std::endl;
      assemble_system();
        std::cout << "Complete: assemble_system" << std::endl;
        std::cout << (clock()-runstart)/CLOCKS_PER_SEC <<  " seconds" << std::endl;
        std::cout << "---------------------" << std::endl;
        std::cout << "Start Solver" << std::endl;
      const unsigned int n_iterations = solve();
        std::cout << "Complete: solve" << std::endl;
        std::cout << "Converged in " << n_iterations << " iterations." << std::endl;
        std::cout << (clock()-runstart)/CLOCKS_PER_SEC << " seconds" <<  std::endl;
        std::cout << "---------------------" << std::endl;
//        std::cout << "    Updating quadrature point data..." << std::endl;
//      update_quadrature_point_history();
////        std::cout << "    Quadrature point data updated" << std::endl;
//        std::cout << (clock()-runstart)/CLOCKS_PER_SEC << " seconds" <<  std::endl;
//        std::cout << "---------------------" << std::endl;
      output_results();
        std::cout << "Complete: output_results" << std::endl;
        std::cout << (clock()-runstart)/CLOCKS_PER_SEC <<  " seconds" << std::endl;
        std::cout << "---------------------" << std::endl;
  
  

}


  
}  // end namespace

int main(int argc, char **argv)
{
  using namespace shell;
  deallog.depth_console(2);
    Timer timer;

    const std::string in_mesh_filename = "input/3d1cell.inp";
    const std::string nodalbc_filename = "input/shell-bcs-min.csv";
    const std::string out_mesh_filename = ("output/3d1cell.vtu");


    shElast<3,3> shellTest(in_mesh_filename, nodalbc_filename, out_mesh_filename);


    shellTest.run_mesh();
    shellTest.run();

    timer.stop();

    {
            std::cout << " ======= Done with program ========" << std::endl;
            std::cout << "Elapsed CPU time: " << timer.cpu_time() << " seconds.\n";
            std::cout << "Elapsed wall time: " << timer.wall_time() << " seconds.\n";
    }

    timer.reset();
    return 0;
}

