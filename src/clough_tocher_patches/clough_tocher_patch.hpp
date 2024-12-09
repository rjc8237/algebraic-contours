#pragma once

#include "common.h"
#include "convex_polygon.h"

#include "evaluate_surface_normal.h"
#include "optimize_spline_surface.h"

class CloughTocherPatch {
public:
  CloughTocherPatch(const Eigen::MatrixXd &V,
                    const AffineManifold &affine_manifold,
                    const OptimizationParameters &optimization_params,
                    std::vector<std::vector<int>> &face_to_patch_indices,
                    std::vector<int> &patch_to_face_indices,
                    Eigen::SparseMatrix<double> &fit_matrix,
                    Eigen::SparseMatrix<double> &energy_hessian,
                    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>>
                        &energy_hessian_inverse);

  //   CloughTocherPatch(const Eigen::MatrixXd surface_mapping_coeffs,
  //                     const ConvexPolygon &domain);

  int triangle_ind(const double &u, const double &v, const double &w);

  Eigen::Matrix<double, 1, 10>
  monomial_basis_eval(const double &u, const double &v, const double &w);

  Eigen::Matrix<double, 1, 3> CT_eval(const double &u, const double &v,
                                      const double &w);

  std::array<Eigen::Matrix<double, 10, 3>, 3> get_coeffs();

private:
  void generate_face_normals(const Eigen::MatrixXd &V,
                             const AffineManifold &affine_manifold,
                             Eigen::MatrixXd &N);

private:
  const static std::array<Eigen::Matrix<double, 3, 3>, 3>
      m_CTtri_bounds; // constant ct sub tri boundaries
  const static std::array<Eigen::Matrix<double, 10, 12>, 3>
      m_CT_matrices; // constant ct matrices for 3 sub tris

  std::array<Eigen::Matrix<double, 12, 1>, 3>
      m_boundary_data; // p0, p1, p2, G01, G10, G12, G21, G20, G02, N01, N12,
                       // N20, for x, y z

  AffineManifold m_affine_manifold;
  std::vector<std::array<TriangleCornerFunctionData, 3>> m_corner_data;
  std::vector<std::array<TriangleMidpointFunctionData, 3>> m_midpoint_data;
  Eigen::Matrix<double, 12, 3> m_boundary_data;
  std::array<Eigen::Matrix<double, 10, 3>, 3>
      m_CT_coeffs; // constant ct matrices for 3 sub tris
}