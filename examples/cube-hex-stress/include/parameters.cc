#include <include/parameters.h>

namespace Parameters
{
  void
  Tester::add_output_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Tester");
    {
      prm.add_parameter("Test phrase", t_phrase, "Yeah, no", Patterns::Anything());
    }
    prm.leave_subsection();
  }

