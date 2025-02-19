// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2022 by the deal.II authors
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


#ifndef dealii_cuda_matrix_free_h
#define dealii_cuda_matrix_free_h

#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_CUDA

#  include <deal.II/base/cuda_size.h>
#  include <deal.II/base/memory_space.h>
#  include <deal.II/base/mpi_stub.h>
#  include <deal.II/base/partitioner.h>
#  include <deal.II/base/quadrature.h>
#  include <deal.II/base/tensor.h>
#  include <deal.II/base/utilities.h>

#  include <deal.II/dofs/dof_handler.h>

#  include <deal.II/fe/fe_update_flags.h>
#  include <deal.II/fe/mapping.h>

#  include <deal.II/grid/filtered_iterator.h>

#  include <deal.II/lac/affine_constraints.h>
#  include <deal.II/lac/cuda_vector.h>
#  include <deal.II/lac/la_parallel_vector.h>

#  include <Kokkos_Core.hpp>



DEAL_II_NAMESPACE_OPEN

// Forward declaration
namespace internal
{
  namespace MatrixFreeFunctions
  {
    enum class ConstraintKinds : std::uint16_t;
  }
} // namespace internal

namespace CUDAWrappers
{
  // forward declaration
#  ifndef DOXYGEN
  namespace internal
  {
    template <int dim, typename Number>
    class ReinitHelper;
  }
#  endif

  /**
   * This class collects all the data that is stored for the matrix free
   * implementation. The storage scheme is tailored towards several loops
   * performed with the same data, i.e., typically doing many matrix-vector
   * products or residual computations on the same mesh.
   *
   * This class does not implement any operations involving finite element basis
   * functions, i.e., regarding the operation performed on the cells. For these
   * operations, the class FEEvaluation is designed to use the data collected in
   * this class.
   *
   * This class implements a loop over all cells (cell_loop()). This loop is
   * scheduled in such a way that cells that share degrees of freedom
   * are not worked on simultaneously, which implies that it is possible to
   * write to vectors in parallel without having to explicitly synchronize
   * access to these vectors and matrices. This class does not implement any
   * shape values, all it does is to cache the respective data. To implement
   * finite element operations, use the class CUDAWrappers::FEEvalutation.
   *
   * This class traverse the cells in a different order than the usual
   * Triangulation class in deal.II.
   *
   * @note Only float and double are supported.
   *
   * @ingroup CUDAWrappers
   */
  template <int dim, typename Number = double>
  class MatrixFree : public Subscriptor
  {
  public:
    using jacobian_type = Tensor<2, dim, Tensor<1, dim, Number>>;
    using point_type    = Point<dim, Number>;
    using CellFilter =
      FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

    /**
     * Standardized data struct to pipe additional data to MatrixFree.
     */
    struct AdditionalData
    {
      /**
       * Constructor.
       */
      AdditionalData(const UpdateFlags mapping_update_flags =
                       update_gradients | update_JxW_values |
                       update_quadrature_points,
                     const bool use_coloring                      = false,
                     const bool overlap_communication_computation = false)
        : mapping_update_flags(mapping_update_flags)
        , use_coloring(use_coloring)
        , overlap_communication_computation(overlap_communication_computation)
      {
#  ifndef DEAL_II_MPI_WITH_DEVICE_SUPPORT
        AssertThrow(
          overlap_communication_computation == false,
          ExcMessage(
            "Overlapping communication and computation requires CUDA-aware MPI."));
#  endif
        if (overlap_communication_computation == true)
          AssertThrow(
            use_coloring == false || overlap_communication_computation == false,
            ExcMessage(
              "Overlapping communication and coloring are incompatible options. Only one of them can be enabled."));
      }
      /**
       * This flag is used to determine which quantities should be cached. This
       * class can cache data needed for gradient computations (inverse
       * Jacobians), Jacobian determinants (JxW), quadrature points as well as
       * data for Hessians (derivative of Jacobians). By default, only data for
       * gradients and Jacobian determinants times quadrature weights, JxW, are
       * cached. If quadrature points of second derivatives are needed, they
       * must be specified by this field.
       */
      UpdateFlags mapping_update_flags;

      /**
       * If true, use graph coloring. Otherwise, use atomic operations. Graph
       * coloring ensures bitwise reproducibility but is slower on Pascal and
       * newer architectures.
       */
      bool use_coloring;

      /**
       * Overlap MPI communications with computation. This requires CUDA-aware
       * MPI and use_coloring must be false.
       */
      bool overlap_communication_computation;
    };

    /**
     * Structure which is passed to the kernel. It is used to pass all the
     * necessary information from the CPU to the GPU.
     */
    struct Data
    {
      /**
       * Kokkos::View of the quadrature points.
       */
      Kokkos::View<point_type **, MemorySpace::Default::kokkos_space> q_points;

      /**
       * Map the position in the local vector to the position in the global
       * vector.
       */
      Kokkos::View<types::global_dof_index **,
                   MemorySpace::Default::kokkos_space>
        local_to_global;

      /**
       * Kokkos::View of the inverse Jacobian.
       */
      Kokkos::View<Number **[dim][dim], MemorySpace::Default::kokkos_space>
        inv_jacobian;

      /**
       * Kokkos::View of the Jacobian times the weights.
       */
      Kokkos::View<Number **, MemorySpace::Default::kokkos_space> JxW;

      /**
       * Mask deciding where constraints are set on a given cell.
       */
      Kokkos::View<dealii::internal::MatrixFreeFunctions::ConstraintKinds *,
                   MemorySpace::Default::kokkos_space>
        constraint_mask;

      /**
       * Values of the shape functions.
       */
      Kokkos::View<Number *, MemorySpace::Default::kokkos_space> shape_values;

      /**
       * Gradients of the shape functions.
       */
      Kokkos::View<Number *, MemorySpace::Default::kokkos_space>
        shape_gradients;

      /**
       * Gradients of the shape functions for collocation methods.
       */
      Kokkos::View<Number *, MemorySpace::Default::kokkos_space>
        co_shape_gradients;

      /**
       * Weights used when resolving hanginf nodes.
       */
      Kokkos::View<Number *, MemorySpace::Default::kokkos_space>
        constraint_weights;

      /**
       * Number of cells.
       */
      unsigned int n_cells;

      /**
       * Length of the padding.
       */
      unsigned int padding_length;

      /**
       * Row start (including padding).
       */
      unsigned int row_start;

      /**
       * If true, use graph coloring has been used and we can simply add into
       * the destingation vector. Otherwise, use atomic operations.
       */
      bool use_coloring;
    };

    /**
     * Default constructor.
     */
    MatrixFree();

    /**
     * Return the length of the padding.
     */
    unsigned int
    get_padding_length() const;

    /**
     * Extracts the information needed to perform loops over cells. The
     * DoFHandler and AffineConstraints objects describe the layout of
     * degrees of freedom, the DoFHandler and the mapping describe the
     * transformation from unit to real cell, and the finite element
     * underlying the DoFHandler together with the quadrature formula
     * describe the local operations. This function takes an IteratorFilters
     * object (predicate) to loop over a subset of the active cells. When using
     * MPI, the predicate should filter out non locally owned cells.
     */
    template <typename IteratorFiltersType>
    void
    reinit(const Mapping<dim> &             mapping,
           const DoFHandler<dim> &          dof_handler,
           const AffineConstraints<Number> &constraints,
           const Quadrature<1> &            quad,
           const IteratorFiltersType &      iterator_filter,
           const AdditionalData &           additional_data = AdditionalData());

    /**
     * Same as above using Iterators::LocallyOwnedCell() as predicate.
     */
    void
    reinit(const Mapping<dim> &             mapping,
           const DoFHandler<dim> &          dof_handler,
           const AffineConstraints<Number> &constraints,
           const Quadrature<1> &            quad,
           const AdditionalData &           additional_data = AdditionalData());

    /**
     * Initializes the data structures. Same as above but using a Q1 mapping.
     */
    void
    reinit(const DoFHandler<dim> &          dof_handler,
           const AffineConstraints<Number> &constraints,
           const Quadrature<1> &            quad,
           const AdditionalData &           additional_data = AdditionalData());

    /**
     * Return the Data structure associated with @p color.
     */
    Data
    get_data(unsigned int color) const;

    // clang-format off
    /**
     * This method runs the loop over all cells and apply the local operation on
     * each element in parallel. @p func is a functor which is applied on each color.
     *
     * @p func needs to define
     * \code
     * DEAL_II_HOST_DEVICE void operator()(
     *   const unsigned int                                          cell,
     *   const typename CUDAWrappers::MatrixFree<dim, Number>::Data *gpu_data,
     *   CUDAWrappers::SharedData<dim, Number> *                     shared_data,
     *   const Number *                                              src,
     *   Number *                                                    dst) const;
     *   static const unsigned int n_dofs_1d;
     *   static const unsigned int n_local_dofs;
     *   static const unsigned int n_q_points;
     * \endcode
     */
    // clang-format on
    template <typename Functor, typename VectorType>
    void
    cell_loop(const Functor &   func,
              const VectorType &src,
              VectorType &      dst) const;

    /**
     * This method runs the loop over all cells and apply the local operation on
     * each element in parallel. This function is very similar to cell_loop()
     * but it uses a simpler functor.
     *
     * @p func needs to define
     * \code
     *  DEAL_II_HOST_DEVICE void operator()(
     *    const unsigned int                                          cell,
     *    const typename CUDAWrappers::MatrixFree<dim, Number>::Data *gpu_data);
     * static const unsigned int n_dofs_1d;
     * static const unsigned int n_local_dofs;
     * static const unsigned int n_q_points;
     * \endcode
     */
    template <typename Functor>
    void
    evaluate_coefficients(Functor func) const;

    /**
     * Copy the values of the constrained entries from @p src to @p dst. This is
     * used to impose zero Dirichlet boundary condition.
     */
    template <typename VectorType>
    void
    copy_constrained_values(const VectorType &src, VectorType &dst) const;

    /**
     * Set the entries in @p dst corresponding to constrained values to @p val.
     * The main purpose of this function is to set the constrained entries of
     * the source vector used in cell_loop() to zero.
     */
    template <typename VectorType>
    void
    set_constrained_values(const Number val, VectorType &dst) const;

    /**
     * Initialize a serial vector. The size corresponds to the number of degrees
     * of freedom in the DoFHandler object.
     */
    void
    initialize_dof_vector(
      LinearAlgebra::CUDAWrappers::Vector<Number> &vec) const;

    /**
     * Initialize a distributed vector. The local elements correspond to the
     * locally owned degrees of freedom and the ghost elements correspond to the
     * (additional) locally relevant dofs.
     */
    void
    initialize_dof_vector(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &vec) const;

    /**
     * Return the colored graph of locally owned active cells.
     */
    const std::vector<std::vector<CellFilter>> &
    get_colored_graph() const;

    /**
     * Return the partitioner that represents the locally owned data and the
     * ghost indices where access is needed to for the cell loop. The
     * partitioner is constructed from the locally owned dofs and ghost dofs
     * given by the respective fields. If you want to have specific information
     * about these objects, you can query them with the respective access
     * functions. If you just want to initialize a (parallel) vector, you should
     * usually prefer this data structure as the data exchange information can
     * be reused from one vector to another.
     */
    const std::shared_ptr<const Utilities::MPI::Partitioner> &
    get_vector_partitioner() const;

    /**
     * Return the DoFHandler.
     */
    const DoFHandler<dim> &
    get_dof_handler() const;

    /**
     * Return an approximation of the memory consumption of this class in bytes.
     */
    std::size_t
    memory_consumption() const;

  private:
    /**
     * Initializes the data structures.
     */
    template <typename IteratorFiltersType>
    void
    internal_reinit(const Mapping<dim> &             mapping,
                    const DoFHandler<dim> &          dof_handler,
                    const AffineConstraints<Number> &constraints,
                    const Quadrature<1> &            quad,
                    const IteratorFiltersType &      iterator_filter,
                    std::shared_ptr<const MPI_Comm>  comm,
                    const AdditionalData             additional_data);

    /**
     * Helper function. Loop over all the cells and apply the functor on each
     * element in parallel. This function is used when MPI is not used.
     */
    template <typename Functor, typename VectorType>
    void
    serial_cell_loop(const Functor &   func,
                     const VectorType &src,
                     VectorType &      dst) const;

    /**
     * Helper function. Loop over all the cells and apply the functor on each
     * element in parallel. This function is used when MPI is used.
     */
    template <typename Functor>
    void
    distributed_cell_loop(
      const Functor &                                                      func,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src,
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst) const;

    /**
     * This function should never be called. Calling it results in an internal
     * error. This function exists only because cell_loop needs
     * distributed_cell_loop() to exist for LinearAlgebra::CUDAWrappers::Vector.
     */
    template <typename Functor>
    void
    distributed_cell_loop(
      const Functor &                                    func,
      const LinearAlgebra::CUDAWrappers::Vector<Number> &src,
      LinearAlgebra::CUDAWrappers::Vector<Number> &      dst) const;

    /**
     * Unique ID associated with the object.
     */
    int my_id;

    /**
     * If true, use graph coloring. Otherwise, use atomic operations. Graph
     * coloring ensures bitwise reproducibility but is slower on Pascal and
     * newer architectures.
     */
    bool use_coloring;

    /**
     *  Overlap MPI communications with computation. This requires CUDA-aware
     *  MPI and use_coloring must be false.
     */
    bool overlap_communication_computation;

    /**
     * Total number of degrees of freedom.
     */
    types::global_dof_index n_dofs;

    /**
     * Degree of the finite element used.
     */
    unsigned int fe_degree;

    /**
     * Number of degrees of freedom per cell.
     */
    unsigned int dofs_per_cell;

    /**
     * Number of constrained degrees of freedom.
     */
    unsigned int n_constrained_dofs;

    /**
     * Number of quadrature points per cells.
     */
    unsigned int q_points_per_cell;

    /**
     * Number of colors produced by the graph coloring algorithm.
     */
    unsigned int n_colors;

    /**
     * Number of cells in each color.
     */
    std::vector<unsigned int> n_cells;

    /**
     * Vector of Kokkos::View to the quadrature points associated to the cells
     * of each color.
     */
    std::vector<Kokkos::View<point_type **, MemorySpace::Default::kokkos_space>>
      q_points;

    /**
     * Map the position in the local vector to the position in the global
     * vector.
     */
    std::vector<Kokkos::View<types::global_dof_index **,
                             MemorySpace::Default::kokkos_space>>
      local_to_global;

    /**
     * Vector of Kokkos::View of the inverse Jacobian associated to the cells of
     * each color.
     */
    std::vector<
      Kokkos::View<Number **[dim][dim], MemorySpace::Default::kokkos_space>>
      inv_jacobian;

    /**
     * Vector of Kokkos::View to the Jacobian times the weights associated to
     * the cells of each color.
     */
    std::vector<Kokkos::View<Number **, MemorySpace::Default::kokkos_space>>
      JxW;

    /**
     * Kokkos::View to the constrained degrees of freedom.
     */
    Kokkos::View<types::global_dof_index *, MemorySpace::Default::kokkos_space>
      constrained_dofs;

    /**
     * Mask deciding where constraints are set on a given cell.
     */
    std::vector<
      Kokkos::View<dealii::internal::MatrixFreeFunctions::ConstraintKinds *,
                   MemorySpace::Default::kokkos_space>>
      constraint_mask;

    /**
     * Values of the shape functions.
     */
    Kokkos::View<Number *, MemorySpace::Default::kokkos_space> shape_values;

    /**
     * Gradients of the shape functions.
     */
    Kokkos::View<Number *, MemorySpace::Default::kokkos_space> shape_gradients;

    /**
     * Gradients of the shape functions for collocation methods.
     */
    Kokkos::View<Number *, MemorySpace::Default::kokkos_space>
      co_shape_gradients;

    /**
     * Weights used when resolving hanginf nodes.
     */
    Kokkos::View<Number *, MemorySpace::Default::kokkos_space>
      constraint_weights;

    /**
     * Grid dimensions associated to the different colors. The grid dimensions
     * are used to launch the CUDA kernels.
     */
    std::vector<dim3> grid_dim;

    /**
     * Block dimensions associated to the different colors. The block dimensions
     * are used to launch the CUDA kernels.
     */
    std::vector<dim3> block_dim;

    /**
     * Shared pointer to a Partitioner for distributed Vectors used in
     * cell_loop. When MPI is not used the pointer is null.
     */
    std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;

    /**
     * Cells per block (determined by the function cells_per_block_shmem() ).
     */
    unsigned int cells_per_block;

    /**
     * Grid dimensions used to launch the CUDA kernels
     * in *_constrained_values-operations.
     */
    dim3 constraint_grid_dim;

    /**
     * Block dimensions used to launch the CUDA kernels
     * in *_constrained_values-operations.
     */
    dim3 constraint_block_dim;

    /**
     * Length of the padding (closest power of two larger than or equal to
     * the number of thread).
     */
    unsigned int padding_length;

    /**
     * Row start of each color.
     */
    std::vector<unsigned int> row_start;

    /**
     * Pointer to the DoFHandler associated with the object.
     */
    const DoFHandler<dim> *dof_handler;

    /**
     * Colored graphed of locally owned active cells.
     */
    std::vector<std::vector<CellFilter>> graph;

    friend class internal::ReinitHelper<dim, Number>;
  };



  // TODO We should rework this to use scratch memory
  /**
   * Structure to pass the shared memory into a general user function.
   */
  template <int dim, typename Number>
  struct SharedData
  {
    /**
     * Memory for dof and quad values.
     */
    Kokkos::Subview<Kokkos::View<Number *, MemorySpace::Default::kokkos_space>,
                    Kokkos::pair<int, int>>
      values;

    /**
     * Memory for computed gradients in reference coordinate system.
     */
    Kokkos::Subview<
      Kokkos::View<Number *[dim], MemorySpace::Default::kokkos_space>,
      Kokkos::pair<int, int>,
      Kokkos::pair<int, int>>
      gradients;
  };



  // This function determines the number of cells per block, possibly at compile
  // time (by virtue of being 'constexpr')
  // TODO this function should be rewritten using meta-programming
  DEAL_II_HOST_DEVICE constexpr unsigned int
  cells_per_block_shmem(int dim, int fe_degree)
  {
    /* clang-format off */
    // We are limiting the number of threads according to the
    // following formulas:
    //  - in 2d: `threads = cells * (k+1)^d <= 4*CUDAWrappers::warp_size`
    //  - in 3d: `threads = cells * (k+1)^d <= 2*CUDAWrappers::warp_size`
    return dim==2 ? (fe_degree==1 ? CUDAWrappers::warp_size :    // 128
                     fe_degree==2 ? CUDAWrappers::warp_size/4 :  //  72
                     fe_degree==3 ? CUDAWrappers::warp_size/8 :  //  64
                     fe_degree==4 ? CUDAWrappers::warp_size/8 :  // 100
                     1) :
           dim==3 ? (fe_degree==1 ? CUDAWrappers::warp_size/4 :  //  64
                     fe_degree==2 ? CUDAWrappers::warp_size/16 : //  54
                     1) : 1;
    /* clang-format on */
  }


  /*----------------------- Helper functions ---------------------------------*/
  /**
   * Compute the quadrature point index in the local cell of a given thread.
   *
   * @relates CUDAWrappers::MatrixFree
   */
  template <int dim>
  DEAL_II_HOST_DEVICE inline unsigned int
  q_point_id_in_cell(const unsigned int n_q_points_1d)
  {
    KOKKOS_IF_ON_DEVICE(
      return (dim == 1 ?
                threadIdx.x % n_q_points_1d :
              dim == 2 ?
                threadIdx.x % n_q_points_1d + n_q_points_1d * threadIdx.y :
                threadIdx.x % n_q_points_1d +
                  n_q_points_1d * (threadIdx.y + n_q_points_1d * threadIdx.z));)

    KOKKOS_IF_ON_HOST(AssertThrow(false, ExcInternalError()); return 0;)
  }



  /**
   * Return the quadrature point index local of a given thread. The index is
   * only unique for a given MPI process.
   *
   * @relates CUDAWrappers::MatrixFree
   */
  template <int dim, typename Number>
  DEAL_II_HOST_DEVICE inline unsigned int
  local_q_point_id(
    const unsigned int                                          cell,
    const typename CUDAWrappers::MatrixFree<dim, Number>::Data *data,
    const unsigned int                                          n_q_points_1d,
    const unsigned int                                          n_q_points)
  {
    return (data->row_start / data->padding_length + cell) * n_q_points +
           q_point_id_in_cell<dim>(n_q_points_1d);
  }



  /**
   * Return the quadrature point associated with a given thread.
   *
   * @relates CUDAWrappers::MatrixFree
   */
  template <int dim, typename Number>
  DEAL_II_HOST_DEVICE inline
    typename CUDAWrappers::MatrixFree<dim, Number>::point_type &
    get_quadrature_point(
      const unsigned int                                          cell,
      const typename CUDAWrappers::MatrixFree<dim, Number>::Data *data,
      const unsigned int                                          n_q_points_1d)
  {
    return data->q_points(cell, q_point_id_in_cell<dim>(n_q_points_1d));
  }

  /**
   * Structure which is passed to the kernel. It is used to pass all the
   * necessary information from the CPU to the GPU.
   */
  template <int dim, typename Number>
  struct DataHost
  {
    /**
     * Kokkos::View of quadrature points on the host.
     */
    typename Kokkos::View<Point<dim, Number> **,
                          MemorySpace::Default::kokkos_space>::HostMirror
      q_points;

    /**
     * Map the position in the local vector to the position in the global
     * vector.
     */
    typename Kokkos::View<types::global_dof_index **,
                          MemorySpace::Default::kokkos_space>::HostMirror
      local_to_global;

    /**
     * Kokkos::View of inverse Jacobians on the host.
     */
    typename Kokkos::View<Number **[dim][dim],
                          MemorySpace::Default::kokkos_space>::HostMirror
      inv_jacobian;

    /**
     * Kokkos::View of Jacobian times the weights on the host.
     */
    typename Kokkos::View<Number **,
                          MemorySpace::Default::kokkos_space>::HostMirror JxW;

    /**
     * Number of cells.
     */
    unsigned int n_cells;

    /**
     * Length of the padding.
     */
    unsigned int padding_length;

    /**
     * Row start (including padding).
     */
    unsigned int row_start;

    /**
     * Mask deciding where constraints are set on a given cell.
     */
    typename Kokkos::View<
      dealii::internal::MatrixFreeFunctions::ConstraintKinds *,
      MemorySpace::Default::kokkos_space>::HostMirror constraint_mask;

    /**
     * If true, use graph coloring has been used and we can simply add into
     * the destingation vector. Otherwise, use atomic operations.
     */
    bool use_coloring;
  };



  /**
   * Copy @p data from the @ref GlossDevice "device" to the host. @p update_flags should be
   * identical to the one used in MatrixFree::AdditionalData.
   *
   * @relates CUDAWrappers::MatrixFree
   */
  template <int dim, typename Number>
  DataHost<dim, Number>
  copy_mf_data_to_host(
    const typename dealii::CUDAWrappers::MatrixFree<dim, Number>::Data &data,
    const UpdateFlags &update_flags)
  {
    DataHost<dim, Number> data_host;

    data_host.n_cells        = data.n_cells;
    data_host.padding_length = data.padding_length;
    data_host.row_start      = data.row_start;
    data_host.use_coloring   = data.use_coloring;

    const unsigned int n_elements =
      data_host.n_cells * data_host.padding_length;
    if (update_flags & update_quadrature_points)
      {
        data_host.q_points = Kokkos::create_mirror(data.q_points);
        Kokkos::deep_copy(data_host.q_points, data.q_points);
      }

    data_host.local_to_global = Kokkos::create_mirror(data.local_to_global);
    Kokkos::deep_copy(data_host.local_to_global, data.local_to_global);

    if (update_flags & update_gradients)
      {
        data_host.inv_jacobian = Kokkos::create_mirror(data.inv_jacobian);
        Kokkos::deep_copy(data_host.inv_jacobian, data.inv_jacobian);
      }

    if (update_flags & update_JxW_values)
      {
        data_host.JxW = Kokkos::create_mirror(data.JxW);
        Kokkos::deep_copy(data_host.JxW, data.JxW);
      }

    data_host.constraint_mask = Kokkos::create_mirror(data.constraint_mask);
    Kokkos::deep_copy(data_host.constraint_mask, data.constraint_mask);

    return data_host;
  }



  /**
   * This function is the host version of local_q_point_id().
   *
   * @relates CUDAWrappers::MatrixFree
   */
  template <int dim, typename Number>
  inline unsigned int
  local_q_point_id_host(const unsigned int           cell,
                        const DataHost<dim, Number> &data,
                        const unsigned int           n_q_points,
                        const unsigned int           i)
  {
    return (data.row_start / data.padding_length + cell) * n_q_points + i;
  }



  /**
   * This function is the host version of get_quadrature_point(). It assumes
   * that the data in MatrixFree<dim, Number>::Data has been copied to the host
   * using copy_mf_data_to_host().
   *
   * @relates CUDAWrappers::MatrixFree
   */
  template <int dim, typename Number>
  inline Point<dim, Number>
  get_quadrature_point_host(const unsigned int           cell,
                            const DataHost<dim, Number> &data,
                            const unsigned int           i)
  {
    return data.q_points(cell, i);
  }


  /*----------------------- Inline functions ---------------------------------*/

#  ifndef DOXYGEN

  template <int dim, typename Number>
  inline const std::vector<std::vector<
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>>> &
  MatrixFree<dim, Number>::get_colored_graph() const
  {
    return graph;
  }



  template <int dim, typename Number>
  inline const std::shared_ptr<const Utilities::MPI::Partitioner> &
  MatrixFree<dim, Number>::get_vector_partitioner() const
  {
    return partitioner;
  }



  template <int dim, typename Number>
  inline const DoFHandler<dim> &
  MatrixFree<dim, Number>::get_dof_handler() const
  {
    Assert(dof_handler != nullptr, ExcNotInitialized());

    return *dof_handler;
  }

#  endif

} // namespace CUDAWrappers

DEAL_II_NAMESPACE_CLOSE

#endif
#endif
