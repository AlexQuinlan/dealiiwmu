/*
Written by Alex Quinlan
*/

#ifndef dealii_nodal_bcs_h
#define dealii_nodal_bcs_h


DEAL_II_NAMESPACE_OPEN





// /home/quinlan/Github/dealii/examples/thermal_shell/thermal-shell.cc:423:12: error: no matching function for call to ‘

// nodal_bcs(const string&, dealii::DoFHandler<2, 3>&, dealii::AffineConstraints<double>&)’
//   423 |   nodal_bcs(nodalbc_filename ,  dof_handler   ,constraints);  // Applying the nodal constraints using WMU function
//       |   ~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// In file included from /home/quinlan/Github/dealii/examples/thermal_shell/thermal-shell.cc:60:
// /home/quinlan/Github/dealii/examples/thermal_shell/../aqtools/nodal_bcs.h:32:6: note: candidate:

// ‘template<int dim, int spacedim>

//              nodal_bcs(const string&, dealii::DoFHandler<2, 3>&, dealii::AffineConstraints<double>&)’
// void dealii::nodal_bcs(const string&, dealii::DoFHandler<2, 3>&, dealii::AffineConstraints<double>&)’

//  nodal_bcs(const std::string&  nodalbc_filename , dealii::DoFHandler<2,3>&  dof_handler   , dealii::AffineConstraints<double>&  constraints   )
//       |      ^~~~~~~~~



// Helper ChatGPT Function
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


void bogo( void)
{
  std::cout << "bogo" << std::endl;  // Test function
}

template <int dim, int spacedim>    
void nodal_bcs(const std::string&  nodalbc_filename , dealii::DoFHandler<2,3>&  dof_handler   , dealii::AffineConstraints<double>&  constraints   )
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

      for (unsigned int i = 3; i < fields.size(); i++){
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






// Ending
DEAL_II_NAMESPACE_CLOSE

#endif
