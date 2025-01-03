#pragma once

#include "clough_tocher_patch.hpp"
#include "common.h"
#include "optimize_spline_surface.h"
#include "position_data.h"

class CloughTocherSurface {
public:
  typedef size_t PatchIndex;

  CloughTocherSurface();
  CloughTocherSurface(const Eigen::MatrixXd &V,
                      const AffineManifold &affine_manifold,
                      const OptimizationParameters &optimization_params,
                      Eigen::SparseMatrix<double> &fit_matrix,
                      Eigen::SparseMatrix<double> &energy_hessian,
                      Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>>
                          &energy_hessian_inverse);

  Eigen::Matrix<double, 1, 3> evaluate_patch(const PatchIndex &patch_index,
                                             const double &u, const double &v,
                                             const double &w);

  void write_cubic_surface_to_msh_no_conn(std::string filename);
  void write_coeffs_to_obj(std::string filename);
  void sample_to_obj(std::string filename, int sample_size = 10);

  void write_cubic_surface_to_msh_with_conn(std::string filename);
  void write_cubic_surface_to_msh_with_conn_from_lagrange_nodes(
      std::string filename);

private:
  void generate_face_normals(const Eigen::MatrixXd &V,
                             const AffineManifold &affine_manifold,
                             Eigen::MatrixXd &N);

private:
  std::vector<CloughTocherPatch> m_patches;

  AffineManifold m_affine_manifold;
  std::vector<std::array<TriangleCornerFunctionData, 3>> m_corner_data;
  std::vector<std::array<TriangleMidpointFunctionData, 3>> m_midpoint_data;
};