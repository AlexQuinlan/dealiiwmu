// ---------------------------------------------------------------------
//
// Copyright (C) 2019 - 2021 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#ifndef dealii_function_signed_distance_h
#define dealii_function_signed_distance_h

#include <deal.II/base/config.h>

#include <deal.II/base/function.h>

#include <array>

DEAL_II_NAMESPACE_OPEN

namespace Functions
{
  namespace SignedDistance
  {
    /**
     * Signed-distance level set function of a sphere:
     * $\psi(x) = \| x - x^c \| - R$.
     * Here, $x^c$ is the center of the sphere and $R$ is its radius. This
     * function is thus zero on the sphere, negative "inside" the ball having
     * the sphere as its boundary, and positive in the rest of
     * $\mathbb{R}^{dim}$.
     *
     * This function has gradient and Hessian equal to
     * $\partial_i \psi(x) = (x - x^c)/\| x - x^c \|$,
     * $\partial_i \partial_j \psi =
     * \delta_{ij}/\| x - x^c \| - (x_i - x_i^c)(x_j - x_j^c)/\| x - x^c \|^3$,
     * where $\delta_{ij}$ is the Kronecker delta function.
     *
     * @ingroup functions
     */
    template <int dim>
    class Sphere : public Function<dim>
    {
    public:
      /**
       * Constructor, takes the center and radius of the sphere.
       */
      Sphere(const Point<dim> &center = Point<dim>(), const double radius = 1);

      double
      value(const Point<dim> & point,
            const unsigned int component = 0) const override;

      /**
       * @copydoc Function::gradient()
       *
       * @note The gradient is singular at the center of the sphere. If the
       * incoming @p point is too close to the center, a floating-point
       * exception may be thrown or entries in the gradient may be +inf/-inf
       * or +nan/-nan, depending on how the point is located relative to the
       * singularity.
       */
      Tensor<1, dim>
      gradient(const Point<dim> & point,
               const unsigned int component = 0) const override;

      /**
       * @copydoc Function::hessian()
       *
       * @note The Hessian is singular at the center of the sphere. If the
       * incoming @p point is too close to the center, a floating-point
       * exception may be thrown or entries in the Hessian may be +inf/-inf
       * or +nan/-nan, depending on how the point is located relative to the
       * singularity.
       */
      SymmetricTensor<2, dim>
      hessian(const Point<dim> & point,
              const unsigned int component = 0) const override;

    private:
      const Point<dim> center;
      const double     radius;
    };


    /**
     * Signed level set function of a plane in $\mathbb{R}^{dim}$:
     * $\psi(x) = n \cdot (x - x_p)$.
     * Here, $n$ is the plane normal and $x_p$ is a point in the plane.
     * Thus, with respect to the direction of the normal, this function is
     * positive above the plane, zero in the plane, and negative below the
     * plane. If the normal is normalized, $\psi$ will be the signed distance to
     * the closest point in the plane.
     *
     * @ingroup functions
     */
    template <int dim>
    class Plane : public Function<dim>
    {
    public:
      /**
       * Constructor, takes a point in the plane and the plane normal.
       */
      Plane(const Point<dim> &point, const Tensor<1, dim> &normal);

      double
      value(const Point<dim> & point,
            const unsigned int component = 0) const override;

      Tensor<1, dim>
      gradient(const Point<dim> &,
               const unsigned int component = 0) const override;

      SymmetricTensor<2, dim>
      hessian(const Point<dim> &,
              const unsigned int component = 0) const override;

    private:
      const Point<dim>     point_in_plane;
      const Tensor<1, dim> normal;
    };


    /**
     * Signed-distance level set function to an ellipsoid defined by:
     *
     * @f[
     * \sum_{i=1}^{dim} \frac{(x_i - c_i)^2}{R_i^2} = 1
     * @f]
     *
     * Here, $c_i$ are the coordinates of the center of the ellipsoid and $R_i$
     * are the elliptic radii. This function is zero on the ellipsoid, negative
     * inside the ellipsoid and positive outside the ellipsoid.
     *
     * @ingroup functions
     */
    template <int dim>
    class Ellipsoid : public Function<dim>
    {
    public:
      /**
       * Constructor, takes the center and radii of the ellipsoid.
       *
       * @param center Center of the ellipsoid.
       * @param radii Array of radii of the ellipsoid.
       * @param tolerance Tolerance of the distance computation.
       * @param max_iter Max. number of iteration of the distance computation algorithm.
       */
      Ellipsoid(const Point<dim> &             center,
                const std::array<double, dim> &radii,
                const double                   tolerance = 1e-14,
                const unsigned int             max_iter  = 10);

      double
      value(const Point<dim> & point,
            const unsigned int component = 0) const override;

    private:
      /**
       * Evaluates the ellipsoid function:
       *
       * @f[
       * f(\vec{x}) = \sum_{i=1}^{dim} \frac{(x_i - c_i)^2}{R_i^2} - 1
       * @f]
       */
      double
      evaluate_ellipsoid(const Point<dim> &point) const;

      /**
       * Compute the signed distance to a 2D ellipsoid i.e. ellipse.
       */
      double
      compute_signed_distance_ellipse(const Point<dim> &point) const;

      const Point<dim>              center;
      const std::array<double, dim> radii;
      const double                  tolerance;
      const unsigned int            max_iter;
    };
  } // namespace SignedDistance
} // namespace Functions

DEAL_II_NAMESPACE_CLOSE

#endif
