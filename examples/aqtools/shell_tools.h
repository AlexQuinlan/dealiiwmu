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
//  tmp[0][0][0][0] = Eprime;//  tmp[0][0][1][1] = nu * Eprime;//  tmp[1][1][0][0] = nu * Eprime;
//  tmp[1][1][1][1] = Eprime;//  tmp[2][2][2][2] = Eprime;//  tmp[0][1][0][1] = G;
//  tmp[1][2][1][2] = 5*G/6;//  tmp[2][0][2][0] = 5*G/6;
  return tmp;
}

template <int dim, int spacedim>  //// From cube_hex
FullMatrix<double> get_shell_constituative_matrix(const double E,
                                                  const double nu)
{
  FullMatrix<double> tmp(6,6);

  const double Eprime = ( E / (1 - nu*nu));
  const double G = E / (2.*(1+nu));
  tmp[0][0] = Eprime;
  tmp[0][1] = nu * Eprime;
  tmp[1][0] = nu * Eprime;
  tmp[1][1] = Eprime;
  tmp[2][2] = Eprime; // could remove per 16.5-12
  tmp[3][3] = G;
  tmp[4][4] = 5*G/6;
  tmp[5][5] = 5*G/6;
  return tmp;
}




// ****** Replacing FE_Values ***************

//double get_lagrange(unsigned int i, double xsi, double eta) // for -1 to 1 bounds
//{
//  if (i == 1)
//    return 0.25 * (1-xsi) * (1-eta);
//  else if (i == 2)
//      return 0.25 * (1+xsi) * (1-eta);
//  else if (i == 4)  // deal.ii order swaps 3 and 4
//      return 0.25 * (1+xsi) * (1+eta);
//  else if (i == 3)  // deal.ii order swaps 3 and 4
//      return 0.25 * (1-xsi) * (1+eta);
//  else
//    return 0.0;
// // this one should match the inbuilt shape functions
//}
//
//double get_lagr_deriv(unsigned int i, double xsi, double eta, unsigned int wrt) // for -1 to 1 bounds
//{
//  if (wrt == 0)   // with respect to xsi
//  {
//    if (i == 1)
//        return -0.25 * (1-eta);
//    else if (i == 2)
//        return  0.25 * (1-eta);
//    else if (i == 4)  // deal.ii order swaps 3 and 4
//        return  0.25 * (1+eta);
//    else if (i == 3)  // deal.ii order swaps 3 and 4
//        return -0.25 * (1+eta);
//  }
//
//  else if (wrt == 1)   // with respect to eta
//  {
//    if (i == 1)
//        return -0.25 * (1-xsi);
//    else if (i == 2)
//        return -0.25 * (1+xsi);
//    else if (i == 4)  // deal.ii order swaps 3 and 4
//        return  0.25 * (1+xsi);
//    else if (i == 3)  // deal.ii order swaps 3 and 4
//        return  0.25 * (1-xsi);
//  }
//
//
//  else
//    return 0.0;
// // this one should match the inbuilt shape functions
//}

double get_lagrange(unsigned int i, double xsi, double eta) // for 0 to 1 bounds
{
  if (i == 0)
    return (1-xsi) * (1-eta);
  else if (i == 1)
      return (xsi) * (1-eta);
  else if (i == 3)  // deal.ii order swaps 3 and 4
      return (xsi) * (eta);
  else if (i == 2)  // deal.ii order swaps 3 and 4
      return (1-xsi) * (eta);
  else
    return 0.0;
 // this one should match the inbuilt shape functions
}

double get_lagr_deriv(unsigned int i, double xsi, double eta, unsigned int wrt) // for 0 to 1 bounds
{
  if (wrt == 0)   // with respect to xsi
  {
    if (i == 0)
        return  -(1-eta);
    else if (i == 1)
        return  (1-eta);
    else if (i == 3)  // deal.ii order swaps 3 and 4
        return   eta;
    else if (i == 2)  // deal.ii order swaps 3 and 4
        return  -eta;
  }
  else if (wrt == 1)   // with respect to eta
  {
    if (i == 0)
        return  -(1-xsi);
    else if (i == 1)
        return  -xsi;
    else if (i == 3)  // deal.ii order swaps 3 and 4
        return   xsi;
    else if (i == 2)  // deal.ii order swaps 3 and 4
        return  (1-xsi);
  }
  else
    return 0.0;
 // this one should match the inbuilt shape functions
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
 * Coulomb WMUshell
 */
template <int dim>
class WMUshellFunction : public dealii::Function<dim>
{
public:
  WMUshellFunction()
    : dealii::Function<dim>(1)
  {}

//  virtual double
//  value(const dealii::Point<dim> &point,
//        const unsigned int        component = 0) const;
//
//  virtual typename dealii::Tensor<1, dim, double>
//  gradient(const dealii::Point<dim> &point,
//        const unsigned int        component = 0) const;

  virtual void  set_corners( const dealii::Point<3,double> & /*, const unsigned int */);
  virtual void  set_thick(double);
  virtual void  set_zeta(double);

  virtual void  calc_corner_norms();
  virtual void  calc_corner_mu();
  virtual void  print_corners();
  virtual double get_grad(const unsigned int , const unsigned int , const int );
//  std::vector<double> sum_shape_func(double , double);
  dealii::Tensor<1, dim, double> calc_norm( double , double );
  virtual void set_J(double , double , double ) ;
  virtual void calc_Jinv();
  virtual void set_H(void);
  dealii::Tensor<1, dim, double> sum_shape_func(double , double);
  dealii::Tensor<1, dim, double> sum_shape_grad(double , double, unsigned int);
  void print_matrices(std::ostream & ,double , double );
//  dealii::TriaActiveIterator<dealii::DoFCellAccessor<3, 3, false> >  cur_cell;
  std::vector<dealii::Point<3,double>> 			corners;
  std::vector<dealii::Tensor<1, dim, double>> 	corner_norms;
  std::vector<dealii::Tensor<1, dim, double>> 	lmn2;  // vectors orthoganal
  std::vector<dealii::Tensor<1, dim, double>> 	lmn1;  // to the normal
  double cell_thick;
  double zeta;
  FullMatrix<double> J;
  FullMatrix<double> HJinv;
  FullMatrix<double> H;   // this doesn't need to be double.  <int> was causing an error, tho
  FullMatrix<double> Jinv;
  FullMatrix<double> N;
  FullMatrix<double> B;
  void print_J(void);
  void print_Jinv(void);
  void print_3x3mat(FullMatrix<double>);
  void calc_HJinv(void);
  void set_N(double , double );
  void calc_B(void);
  void reset(void);
  double detJ;
};


template <int dim>
//std::vector<double>
dealii::Tensor<1, dim, double> WMUshellFunction<dim>::sum_shape_grad(double xsi, double eta, unsigned int wrt)
{
  dealii::Tensor<1, dim, double> Ngrad;
  dealii::Point<3, double> tmp;
  double LG;
  for (unsigned int i = 0; i < corners.size() ; i++ )
  {
    LG = get_lagr_deriv( i ,  xsi,  eta, wrt);  // LG could be removed; used here for debug
    tmp += LG * corners[i];

  }
  Ngrad = tmp;  // converting from Point to Tensor.  Not sure if necessary
  return Ngrad;
}

template <int dim>
void WMUshellFunction<dim>::set_H(void)
{
	FullMatrix<double> tmp5(6,9);  // This function is no longer needed

	tmp5(0,0) = 1;
	tmp5(1,4) = 1;
	tmp5(2,8) = 1;
	tmp5(3, 1) = 1;
	tmp5(3, 3) = 1;
	tmp5(4, 5) = 1;
	tmp5(4, 7) = 1;
	tmp5(5, 2) = 1;
	tmp5(5, 6) = 1;
	H = tmp5;
}

template <int dim>
void WMUshellFunction<dim>::set_J(double xsi, double eta, double zeta) //dealii::TriaAccessor<2, 2, 3> &cell)
{

  FullMatrix<double> tmp(3,3);  // Maybe this isn't needed, idk if writing directly to J has any issues

  for (unsigned int i_sh=0; i_sh < 4; i_sh++)
  {
 // FIX ZETA HERE.  Changed zeta from a mid-surface origin to a bottom surface origin
    //tmp[0][0] += get_lagr_deriv(i_sh,xsi,eta,0)*(corners[i_sh][0]+0.5*zeta*cell_thick*corner_norms[i_sh] );
    tmp(0,0) += get_lagr_deriv(i_sh,xsi,eta,0)*(corners[i_sh][0]+(zeta-0.5)*cell_thick*corner_norms[i_sh][0] );
    tmp(1,0) += get_lagr_deriv(i_sh,xsi,eta,1)*(corners[i_sh][0]+(zeta-0.5)*cell_thick*corner_norms[i_sh][0] );
    tmp(2,0) += get_lagrange(i_sh,xsi,eta)*(cell_thick*corner_norms[i_sh][0] );

    tmp(0,1) += get_lagr_deriv(i_sh,xsi,eta,0)*(corners[i_sh][1]+(zeta-0.5)*cell_thick*corner_norms[i_sh][1] );
    tmp(1,1) += get_lagr_deriv(i_sh,xsi,eta,1)*(corners[i_sh][1]+(zeta-0.5)*cell_thick*corner_norms[i_sh][1] );
    tmp(2,1) += get_lagrange(i_sh,xsi,eta)*(cell_thick*corner_norms[i_sh][1] );

    tmp(0,2) += get_lagr_deriv(i_sh,xsi,eta,0)*(corners[i_sh][2]+(zeta-0.5)*cell_thick*corner_norms[i_sh][2] );
    tmp(1,2) += get_lagr_deriv(i_sh,xsi,eta,1)*(corners[i_sh][2]+(zeta-0.5)*cell_thick*corner_norms[i_sh][2] );
    tmp(2,2) += get_lagrange(i_sh,xsi,eta)*(cell_thick*corner_norms[i_sh][2] );
  }

  J = tmp;
}

template <int dim>
void WMUshellFunction<dim>::calc_Jinv(void)
{
	FullMatrix<double> tmp(3,3);
	FullMatrix<double> tmp1 = J;
//	tmp = FullMatrix<double>::invert( J );
	tmp1.invert(tmp1);
//	for (int i=0; i<3; i++)
//	{
//		for (int j=0; j<3; j++)
//
//	}
	Jinv = tmp1;
	detJ = J.determinant();
}

template <int dim>
void WMUshellFunction<dim>::set_N(double xsi, double eta)
{
	double hzt = (zeta-0.5) * cell_thick ; //
	// Thickness assumed constant across the cell ** will need to be updated
	FullMatrix<double> tmpN(9,24);
	for (int i_sh = 0 ; i_sh<4 ; i_sh++)
	{
		// sparse columns 1, 2, 3
		tmpN(0,0 + i_sh*6)= get_lagr_deriv(i_sh, xsi, eta, 0);
		tmpN(1,0 + i_sh*6)= get_lagr_deriv(i_sh, xsi, eta, 1);

		tmpN(3,1 + i_sh*6)= get_lagr_deriv(i_sh, xsi, eta, 0);
		tmpN(4,1 + i_sh*6)= get_lagr_deriv(i_sh, xsi, eta, 1);

		tmpN(6,2 + i_sh*6)= get_lagr_deriv(i_sh, xsi, eta, 0);
		tmpN(7,2 + i_sh*6)= get_lagr_deriv(i_sh, xsi, eta, 1);

		// fully filled columns 4, 5
		tmpN(0,3 + i_sh*6)= -1*get_lagr_deriv(i_sh, xsi, eta, 0)
				*hzt * lmn2[i_sh][0];
		tmpN(0,4 + i_sh*6)= get_lagr_deriv(i_sh, xsi, eta, 0)
				*hzt * lmn1[i_sh][0];
		tmpN(1,3 + i_sh*6)= -1*get_lagr_deriv(i_sh, xsi, eta, 1)
				*hzt * lmn2[i_sh][0];
		tmpN(1,4 + i_sh*6)= get_lagr_deriv(i_sh, xsi, eta, 1)
				*hzt * lmn1[i_sh][0];
		tmpN(2,3 + i_sh*6)=-cell_thick*lmn2[i_sh][0]*get_lagrange(i_sh, xsi, eta); // previously 0.5*cell_thick
		tmpN(2,4 + i_sh*6)=cell_thick*lmn1[i_sh][0]*get_lagrange(i_sh, xsi, eta);

		tmpN(3,3 + i_sh*6)= -1*get_lagr_deriv(i_sh, xsi, eta, 0)
				*hzt * lmn2[i_sh][1];
		tmpN(3,4 + i_sh*6)= get_lagr_deriv(i_sh, xsi, eta, 0)
				*hzt * lmn1[i_sh][1];
		tmpN(4,3 + i_sh*6)= -1*get_lagr_deriv(i_sh, xsi, eta, 1)
				*hzt * lmn2[i_sh][1];
		tmpN(4,4 + i_sh*6)= get_lagr_deriv(i_sh, xsi, eta, 1)
				*hzt * lmn1[i_sh][1];
		tmpN(5,3 + i_sh*6)=-cell_thick*lmn2[i_sh][1]*get_lagrange(i_sh, xsi, eta);
		tmpN(5,4 + i_sh*6)=cell_thick*lmn1[i_sh][1]*get_lagrange(i_sh, xsi, eta);

		tmpN(6,3 + i_sh*6)= -1*get_lagr_deriv(i_sh, xsi, eta, 0)
				*hzt * lmn2[i_sh][2];
		tmpN(6,4 + i_sh*6)= get_lagr_deriv(i_sh, xsi, eta, 0)
				*hzt * lmn1[i_sh][2];
		tmpN(7,3 + i_sh*6)= -1*get_lagr_deriv(i_sh, xsi, eta, 1)
				*hzt * lmn2[i_sh][2];
		tmpN(7,4 + i_sh*6)= get_lagr_deriv(i_sh, xsi, eta, 1)
				*hzt * lmn1[i_sh][2];
		tmpN(8,3 + i_sh*6)=-cell_thick*lmn2[i_sh][2]*get_lagrange(i_sh, xsi, eta);
		tmpN(8,4 + i_sh*6)=cell_thick*lmn1[i_sh][2]*get_lagrange(i_sh, xsi, eta);

	}
	N = tmpN;
}

template <int dim>
void WMUshellFunction<dim>::calc_B(void)
{
	FullMatrix<double> tmpB(6,24);
	FullMatrix<double> tmphji = HJinv;
	tmphji.mmult(tmpB , N);
	B = tmpB;

}

template <int dim>
void WMUshellFunction<dim>::calc_HJinv(void)
{
	FullMatrix<double> hji(6,9);

	for (int i = 0; i<3; i++)
	{
		hji(0,i)  =Jinv(0,i);
		hji(1,i+3)=Jinv(1,i);
		hji(2,i+6)=Jinv(2,i);

		hji(3,i+0)=Jinv(1,i);
		hji(3,i+3)=Jinv(0,i);

		hji(4,i+3)=Jinv(2,i);
		hji(4,i+6)=Jinv(1,i);

		hji(5,i+0)=Jinv(2,i);
		hji(5,i+6)=Jinv(0,i);
	}

//	hji.print_formatted(std::cout, 5, false, 5, "  0  ", 1., 0.);
	HJinv = hji;

}
template <int dim>
void WMUshellFunction<dim>::print_3x3mat(FullMatrix<double> M)
{
	std::cout << "-------" << std::endl;
	for (int i=0; i<3; i++)
	{
		for (int j=0; j<3; j++)
			std::cout << M[i][j] << "  ";
		std::cout << std::endl;

	}
}

template <int dim>
void WMUshellFunction<dim>::print_J(void)
{
	std::cout << "Jacobian" << std::endl;
	for (int i=0; i<3; i++)
	{
		for (int j=0; j<3; j++)
			std::cout << J[i][j] << "  ";
		std::cout << std::endl;

	}
}

template <int dim>
void WMUshellFunction<dim>::print_Jinv(void)
{
	std::cout << "Inverse Jacobian" << std::endl;
	for (int i=0; i<3; i++)
	{
		for (int j=0; j<3; j++)
			std::cout << Jinv[i][j] << "  ";
		std::cout << std::endl;

	}
}

template <int dim>
//std::vector<double>
dealii::Tensor<1, dim, double> WMUshellFunction<dim>::sum_shape_func(double xsi, double eta)
{
  dealii::Tensor<1, dim, double> N2;
  dealii::Point<3, double> tmp;
  double LG;
  for ( int i = 0; i < corners.size() ; i++ )
  {
    LG = get_lagrange( i ,  xsi,  eta);  // LG could be removed; used here for debug
    tmp += LG * corners[i];

  }
  N2 = tmp;  // converting from Point to Tensor.  Not sure if necessary
  return N2;
}

template <int dim>
void
WMUshellFunction<dim>::set_thick( double t)
{
  // Assert that t > 0;
	cell_thick = t;
}

template <int dim>
void
WMUshellFunction<dim>::set_zeta( double z)
{
  // Assert that -1 < zeta < 1;
	zeta = z;
}

template <int dim>
double
WMUshellFunction<dim>::get_grad( const unsigned int sfno, const unsigned int q_pt, const int wrt)
{
	// Debugging tool
  double xsi=0.5;
  double eta=0.5;

  if ( q_pt == 0)
  {
    xsi = 0.5* (1.-sqrt(1./3) )  ;
    eta = 0.5* (1.-sqrt(1./3) )  ;
  }

  return get_lagr_deriv(sfno, xsi, eta, wrt);
}

template <int dim>
void
WMUshellFunction<dim>::reset( void)
{
	// std::cout << "Point " << v << " coordinates: " ;
	// std::cout << pt << std::endl;

	corners.clear();
	corner_norms.clear();
	lmn2.clear();
	lmn1.clear();
}

template <int dim>
void
WMUshellFunction<dim>::set_corners( const dealii::Point<3,double> &pt /*, const unsigned int v */)
{
	// std::cout << "Point " << v << " coordinates: " ;
	// std::cout << pt << std::endl;

	corners.push_back(pt);
}

template <int dim>
void
WMUshellFunction<dim>::calc_corner_norms( void)
{
  // Lagrange Jacobian; i.e. excluding the shell and enrichment determines
	std::vector<double> xsi = {0.0, 1.0, 1.0, 0.0};
	std::vector<double> eta = {0.0, 0.0, 1.0, 1.0};
	dealii::Tensor<1, dim, double> tmp;
  // at each corner derivative wrt xsi crossed by derivative wrt eta
//  std::vector<double> J;
	for (unsigned int i = 0 ; i < corners.size() ; i++)
	{
	  tmp = calc_norm(xsi[i] , eta[i]);
	  corner_norms.push_back( tmp );
 // std::cout << J3 << std::endl;
	}
}

template <int dim>
void
WMUshellFunction<dim>::calc_corner_mu( void)
{
  // Lagrange Jacobian; i.e. excluding the shell and enrichment determines


	dealii::Tensor<1, dim, double> tmp2;


	for (unsigned int i = 0 ; i < corners.size() ; i++)
	{
	  // cross global-y vector with norm
	  std::vector<double> n(0);
	  n.push_back( corner_norms[i][2] * 1);
	  n.push_back( 0);
	  n.push_back( -corner_norms[i][0] * 1 );


	  dealii::Point<dim, double> ptn = {n[0], n[1], n[2]};
	  dealii::Tensor<1, dim, double> tmp1 = ptn;
	  lmn1.push_back( tmp1 );


	  std::vector<double> m(0);
	  m.push_back( corner_norms[i][1]*lmn1[i][2] - corner_norms[i][2]*lmn1[i][1]);
	  m.push_back( corner_norms[i][2]*lmn1[i][0] - corner_norms[i][0]*lmn1[i][2]);
	  m.push_back( corner_norms[i][0]*lmn1[i][1] - corner_norms[i][1]*lmn1[i][0]);


	  dealii::Point<dim, double> ptm = {m[0], m[1], m[2]};
	  dealii::Tensor<1, dim, double> tmp2 = ptm;
	  lmn2.push_back( tmp2);
	}



}


template <int dim>
dealii::Tensor<1, dim, double>
WMUshellFunction<dim>::calc_norm( double xsi, double eta)
{
	  dealii::Tensor<1, dim, double> J1;
	  dealii::Tensor<1, dim, double> J2;
	  std::vector<double> n(0); //n.clear();
	  double nmag;

	  J1 = sum_shape_grad(xsi, eta, 0);
	  J2 = sum_shape_grad(xsi, eta, 1);
	  n.push_back( J1[1] * J2[2] - J1[2] * J2[1]);
	  n.push_back( J1[2] * J2[0] - J1[0] * J2[2]);
	  n.push_back( J1[0] * J2[1] - J1[1] * J2[0]);
	//  cp = cross_product_2d(J1, J2);
	 // J3 = sum_shape_func(xsi, eta); // this is the Shell Jacobian, not the Lagrange Jacobian

//	  std::cout << "d/dxsi: " << J1 << std::endl;
//	  std::cout << "d/deta: " << J2 << std::endl;

	  nmag = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
	  dealii::Point<dim, double> norm1 = {n[0]/nmag, n[1]/nmag, n[2]/nmag};
//    std::cout << "normal: " << norm1[0] << " " << norm1[1] << " " << norm1[2] << std::endl;
	  dealii::Tensor<1, dim, double> norm2 = norm1;

	  return  norm2;

}

template <int dim>
void
WMUshellFunction<dim>::print_corners( void)
{
  std::cout << "Recalling Corners" << std::endl;
  std::cout << "-----------------" << std::endl;
  for (unsigned int i = 0; i < corners.size() ; i++)
  {
    std::cout << "Point " << i << " coordinates: " ;
  	std::cout << corners[i] << std::endl;
  }

}


template <int dim>
void
WMUshellFunction<dim>::print_matrices(std::ostream & dbgout, double xsi, double eta)
{
	dbgout << "Local Coordinates:" << std::endl;
	dbgout << "xsi = " << xsi << ", eta = " << eta << ", zeta = " << zeta << std::endl;
	dbgout << "-----------------" << std::endl << std::endl;

	  dbgout << "Recalling Corners" << std::endl;
		dbgout << "-----------------" << std::endl;
	  for (unsigned int i = 0; i < corners.size() ; i++)
	  {
		  dbgout << "Point " << i << " coordinates: " ;
	    dbgout << corners[i] << std::endl;
	  }

	  dbgout << "V at Corners" << std::endl;
	  dbgout << "-----------------" << std::endl;
	  for (unsigned int i = 0; i < corners.size() ; i++)
	  {
		dbgout << "V at corner #" << i  << std::endl;
	    dbgout << lmn1[i] << std::endl;
	    dbgout << lmn2[i] << std::endl;
	    dbgout << corner_norms[i] << std::endl;
	  }

	  dbgout << "-----------------" << std::endl;
	  dbgout << "Jacobian" << std::endl;
	  dbgout << "-----------------" << std::endl;
	  J.print_formatted(dbgout, 1, true, 1, "  0  ", 1., 0.);

	  dbgout << "-----------------" << std::endl;
	  dbgout << "Inverse Jacobian" << std::endl;
	  dbgout << "-----------------" << std::endl;
	  Jinv.print_formatted(dbgout, 1, true, 1, "  0  ", 1., 0.);

	  dbgout << "-----------------" << std::endl;
	  dbgout << "H Jinv" << std::endl;
	  dbgout << "-----------------" << std::endl;
	  HJinv.print_formatted(dbgout, 1, true, 1, "  0  ", 1., 0.);

	  dbgout << "-----------------" << std::endl;
	  dbgout << "Shape Functions" << std::endl;
	  dbgout << "-----------------" << std::endl;
	  N.print_formatted(dbgout, 1, true, 1, "  0  ", 1., 0.);


	  dbgout << "-----------------" << std::endl;
	  dbgout << "Non-Transformed B Matrix" << std::endl;
	  dbgout << "-----------------" << std::endl;
	  B.print_formatted(dbgout, 1, true, 1, "  0  ", 1., 0.);



}



// Ending
DEAL_II_NAMESPACE_CLOSE

#endif
