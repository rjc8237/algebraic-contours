#include "clough_tocher_patch.hpp"
#include "clough_tocher_matrices.hpp"

#include <fstream>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>

const std::array<Eigen::Matrix<double, 3, 3>, 3>
    CloughTocherPatch::CTtri_bounds = CT_subtri_bound_matrices();

const std::array<Eigen::Matrix<double, 10, 12>, 3>
    CloughTocherPatch::CT_matrices = CT_subtri_matrices();

CloughTocherPatch::CloughTocherPatch(
    const Eigen::MatrixXd &V, const AffineManifold &affine_manifold,
    const OptimizationParameters &optimization_params,
    std::vector<std::vector<int>> &face_to_patch_indices,
    std::vector<int> &patch_to_face_indices,
    Eigen::SparseMatrix<double> &fit_matrix,
    Eigen::SparseMatrix<double> &energy_hessian,
    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>>
        &energy_hessian_inverse)
    : m_affine_manifold(affine_manifold) {
  // Generate normals
  MatrixXr N;
  generate_face_normals(V, affine_manifold, N);

  // Generate fit matrix by setting the parametrized quadratic surface mapping
  // factor to zero
  double fit_energy;
  VectorXr fit_derivatives;
  OptimizationParameters optimization_params_fit = optimization_params;
  optimization_params_fit.parametrized_quadratic_surface_mapping_factor = 0.0;
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> fit_matrix_inverse;
  build_twelve_split_spline_energy_system(
      V, N, affine_manifold, optimization_params_fit, fit_energy,
      fit_derivatives, fit_matrix, fit_matrix_inverse);

  // Build full energy hessian system
  double energy;
  VectorXr derivatives;
  build_twelve_split_spline_energy_system(
      V, N, affine_manifold, optimization_params, energy, derivatives,
      energy_hessian, energy_hessian_inverse);

  // Build optimized corner and midpoint data
  generate_optimized_twelve_split_position_data(V, affine_manifold, fit_matrix,
                                                energy_hessian_inverse,
                                                m_corner_data, m_midpoint_data);

  // compute boundary data
  for (int i = 0; i < 3; ++i) {
    // p0,p1,p2
    m_boundary_data.row(i) = m_corner_data[i].function_value;
    // G01, G10, G12, G21, G20, G02
    m_boundary_data.row(3 + i * 2 + 0) = m_corner_data[i].first_edge_derivative;
    m_boundary_data.row(3 + i * 2 + 1) =
        m_corner_data[i].second_edge_derivative;
    // N01, N12, N20
    m_boundary_data.row(9 + i) = m_midpoint_data[i].normal_derivative;
  }

  // compute coeff matrices
  for (int i = 0; i < 3; ++i) {
    m_CT_coeffs[i] = m_CT_matrices[i] * m_boundary_data;
  }
}

void CloughTocherPatch::generate_face_normals(
    const Eigen::MatrixXd &V, const AffineManifold &affine_manifold,
    Eigen::MatrixXd &N) {
  Eigen::MatrixXi const &F = affine_manifold.get_faces();

  // Compute the cones of the affine manifold
  std::vector<AffineManifold::Index> cones;
  affine_manifold.compute_cones(cones);

  // Get vertex normals
  Eigen::MatrixXd N_vertices;
  igl::per_vertex_normals(V, F, N_vertices);

  // Set the face one ring normals of the cone vertices to the cone vertex
  // normal
  N.setZero(F.rows(), 3);
  for (size_t i = 0; i < cones.size(); ++i) {
    int ci = cones[i];
    VertexManifoldChart const &chart = affine_manifold.get_vertex_chart(ci);
    for (size_t j = 0; j < chart.face_one_ring.size(); ++j) {
      int fj = chart.face_one_ring[j];
      N.row(fj) = N_vertices.row(ci);
    }
  }
}

int CloughTocherPatch::triangle_ind(const double &u, const double &v,
                                    const double &w) {
  int idx = -1;
  for (int i = 0; i < 3; ++i) {
    if (m_CTtri_bounds[i](0, 0) * u + m_CTtri_bounds[i](0, 1) * v +
                m_CTtri_bounds[i](0, 2) * w >=
            -1e-7 &&
        m_CTtri_bounds[i](1, 0) * u + m_CTtri_bounds[i](1, 1) * v +
                m_CTtri_bounds[i](1, 2) * w >=
            -1e-7 &&
        m_CTtri_bounds[i](2, 0) * u + m_CTtri_bounds[i](2, 1) * v +
                m_CTtri_bounds[i](2, 2) * w >=
            -1e-7) {
      idx = i;
      break;
    }
  }

  assert(idx > -1);
  return idx;
}

Eigen::Matrix<double, 1, 10>
CloughTocherPatch::monomial_basis_eval(const double &u, const double &v,
                                       const double &w) {
  Eigen::Matrix<double, 1, 10> monomial_basis_values;
  monomial_basis_values(0, 0) = w * w * w; // w3
  monomial_basis_values(1, 0) = v * w * w; // vw2
  monomial_basis_values(2, 0) = v * v * w; // v2w
  monomial_basis_values(3, 0) = v * v * v; // v3
  monomial_basis_values(4, 0) = u * w * w; // uw2
  monomial_basis_values(5, 0) = u * v * w; // uvw
  monomial_basis_values(6, 0) = u * v * v; // uv2
  monomial_basis_values(7, 0) = u * u * w; // u2w
  monomial_basis_values(8, 0) = u * u * v; // u2v
  monomial_basis_values(9, 0) = u * u * u; // u3

  return monomial_basis_values;
}

Eigen::Matrix<double, 1, 3>
CloughTocherPatch::CT_eval(const double &u, const double &v, const double &w) {
  int idx = CloughTocherPatch::triangle_ind(u, v, w);
  Eigen::Matrix<double, 1, 10> bb_vector =
      CloughTocherPatch::monomial_basis_eval(u, v, w);

  Eigen::Matrix<double, 1, 3> val;
  val = bb_vector * m_CT_coeffs[idx];
  return val;
}

std::array<Eigen::Matrix<double, 10, 3>, 3> CloughTocherPatch::get_coeffs() {
  return m_CT_coeffs;
}