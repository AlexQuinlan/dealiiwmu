#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <deal.II/base/parameter_handler.h>

/* This class declars the parameters.  The parameters are
   specified in the parameter.prm file
*/

namespace Parameters
{
  using namespace dealii;

  struct Tester
  {
    std::string t_phrase = "yeah, no" ;

    void
    add_output_parameters(ParameterHandler &prm);
    
  };






  struct AllParameters : public Tester                                                                                                                

  {                                                                                                                                                    
    AllParameters(const std::string &input_file);                                                                                                      
  };                                                                                                                                                   
} // namespace Parameters                                                                                                                              
                                                                                                                                                       
#endif // PARAMETERS_H    
  

