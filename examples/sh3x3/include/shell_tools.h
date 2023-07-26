/*
Written by Alex Quinlan
*/

#ifndef dealii_shell_tools_h
#define dealii_shell_tools_h


DEAL_II_NAMESPACE_OPEN

  void ext_func_shtool()
{
  std::cout << "Testing External Function library" << std::endl;
}


template <int dim, int spacedim>  //// From cube_hex
SymmetricTensor<4,spacedim> get_shell_stress_strain_tensor(const double lambda,
                                                const double mu)
{
  SymmetricTensor<4, spacedim> tmp;
  for (unsigned int i = 0; i < spacedim; ++i)
    for (unsigned int j = 0; j < spacedim; ++j)
      for (unsigned int k = 0; k < spacedim; ++k)
        for (unsigned int l = 0; l < spacedim; ++l)
//        	tmp[i][j][k][l] = 0;
        	tmp[i][j][k][l] = (((i == k) && (j == l) ? mu : 0.0) +
                             ((i == l) && (j == k) ? mu : 0.0) +
                             ((i == j) && (k == l) ? lambda : 0.0));

//  const double Eprime = ( E / (1 - nu*nu));
//
//  tmp[0][0][0][0] = Eprime;
//  tmp[0][0][1][1] = nu * Eprime;
//  tmp[1][1][0][0] = nu * Eprime;
//  tmp[1][1][1][1] = Eprime;
//  tmp[2][2][2][2] = Eprime;
//  tmp[0][1][0][1] = G;
//  tmp[1][2][1][2] = 5*G/6;
//  tmp[2][0][2][0] = 5*G/6;

  return tmp;
}

// ****** Replacing FE_Values ***************

double get_lagrange(unsigned int i, double xsi, double eta)
{
  if (i == 1)
    return 1/4 * (1-xsi) * (1-eta);
  else if (i == 2)
      return 1/4 * (1+xsi) * (1-eta);
  else if (i == 3)
      return 1/4 * (1+xsi) * (1+eta);
  else if (i == 4)
      return 1/4 * (1-xsi) * (1+eta);
  else
    return 0.0;
 // this one should match the inbuilt shape functions
}

double get_Ja(double xsi, double eta, dealii::TriaAccessor<2, 2, 3> &cell)
{

  return 0.0;
}

// // template <int dim, int spacedim>
// double get_glob_coords(double xsi, double eta,
//   const typename dealii::TriaActiveIterator::cell_iterator &cell)
// // double get_glob_coords(double xsi, double eta,  const typename Triangulation<dim, spacedim>::cell_iterator &cell)
// {
// std::vector<double> tmp ={0, 0, 0};
// // std::cout << cell.vertex(1) << std::endl;
// //
// // for(unsigned int i=0 ; i < cell.n_vertices() ; i++)
// // {
// //   for (unsigned int j=0 ; j < 3 ; j++)
// //   {
// //   //  tmp[j] += get_lagrange(i, xsi, eta) * cell.vertex(i).values[j].value;
// //     tmp[j] += get_lagrange(i, xsi, eta) * 1;
// //
// //   }
// // }
// return 0.0;
// }

/**
 * Coulomb potential
 */
template <int dim>
class PotentialFunction : public dealii::Function<dim>
{
public:
  PotentialFunction()
    : dealii::Function<dim>(1)
  {}

  virtual double
  value(const dealii::Point<dim> &point,
        const unsigned int        component = 0) const;

  virtual typename dealii::Tensor<1, dim, double>
  gradient(const dealii::Point<dim> &point,
        const unsigned int        component = 0) const;

//  dealii::TriaActiveIterator<dealii::DoFCellAccessor<3, 3, false> >  cur_cell;
  double magic_dub = 0.0;

  virtual void
  set_curcell(const dealii::TriaActiveIterator<dealii::DoFCellAccessor<3, 3, false> >	 ccell);

  virtual void
  set_corners( const dealii::Point<3,double> & , const unsigned int);

  virtual void
  print_magic();

};

template <int dim>
void
PotentialFunction<dim>::set_curcell(const dealii::TriaActiveIterator<dealii::DoFCellAccessor<3, 3, false> > ccell)
{
	Point<3,double> & tmp = ccell->vertex(0);

//	tmp = ccell->vertex(0);
	std::cout << "Vertex 0 x-position:  " << std::endl;
	std::cout << tmp << std::endl;
//cur_cell = ccell;

}

template <int dim>
void
PotentialFunction<dim>::set_corners( const dealii::Point<3,double> &pt, const unsigned int v)
{
	std::cout << "Point " << v << " coordinates: " ;
	std::cout << pt << std::endl;
}

template <int dim>
void
PotentialFunction<dim>::print_magic( void)
{
	std::cout << magic_dub << std::endl;
}

template <int dim>
double
PotentialFunction<dim>::value(const dealii::Point<dim> &p,
                              const unsigned int) const
{
  return -1.0 / std::sqrt(p.square());
}

template <int dim>
typename dealii::Tensor<1, dim, double>
PotentialFunction<dim>::gradient(const dealii::Point<dim> &p,
                              const unsigned int) const
{
	dealii::Tensor<1, dim, double> tmp;
	for (unsigned int i=0 ; i < dim ; i++)
	{
		tmp[i] =  0.2;
	}

	return tmp;
}




// Ending
DEAL_II_NAMESPACE_CLOSE

#endif
