#include "clough_tocher_patch.hpp"
#include "clough_tocher_matrices.hpp"

#include <fstream>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>

const std::array<Eigen::Matrix<double, 3, 3>, 3>
    CloughTocherPatch::m_CTtri_bounds = CT_subtri_bound_matrices();

const std::array<Eigen::Matrix<double, 10, 12>, 3>
    CloughTocherPatch::m_CT_matrices = CT_subtri_matrices();

CloughTocherPatch::CloughTocherPatch(
    const std::array<TriangleCornerFunctionData, 3> &corner_data,
    const std::array<TriangleMidpointFunctionData, 3> &midpoint_data)
    : m_corner_data(corner_data), m_midpoint_data(midpoint_data) {

  // compute boundary data
  for (int i = 0; i < 3; ++i) {
    // p0,p1,p2
    m_boundary_data.row(i) = m_corner_data[i].function_value;
    // G01, G02, G12, G10, G20, G21
    m_boundary_data.row(3 + i * 2 + 0) = m_corner_data[i].first_edge_derivative;
    m_boundary_data.row(3 + i * 2 + 1) =
        m_corner_data[i].second_edge_derivative;
  }

  // p0, p1, p2, G10, G01, G02, G20, G21, G12, N01, N20, N12
  m_boundary_data.row(0) = m_corner_data[0].function_value; // p0
  m_boundary_data.row(1) = m_corner_data[1].function_value; // p1
  m_boundary_data.row(2) = m_corner_data[2].function_value; // p2

  m_boundary_data.row(3) = m_corner_data[0].first_edge_derivative;  // G01
  m_boundary_data.row(4) = m_corner_data[1].second_edge_derivative; // G10
  m_boundary_data.row(5) = m_corner_data[1].first_edge_derivative;  // G12
  m_boundary_data.row(6) = m_corner_data[2].second_edge_derivative; // G21
  m_boundary_data.row(7) = m_corner_data[2].first_edge_derivative;  // G20
  m_boundary_data.row(8) = m_corner_data[0].second_edge_derivative; // G02

  m_boundary_data.row(9) = m_midpoint_data[2].normal_derivative;  // N01
  m_boundary_data.row(10) = m_midpoint_data[0].normal_derivative; // N12
  m_boundary_data.row(11) = m_midpoint_data[1].normal_derivative; // N20

  // compute coeff matrices
  for (int i = 0; i < 3; ++i) {
    m_CT_coeffs[i] = m_CT_matrices[i] * m_boundary_data;
  }
}

int CloughTocherPatch::triangle_ind(const double &u, const double &v,
                                    const double &w) const {
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
                                       const double &w) const {
  Eigen::Matrix<double, 1, 10> monomial_basis_values;
  monomial_basis_values(0, 0) = w * w * w; // w3
  monomial_basis_values(0, 1) = v * w * w; // vw2
  monomial_basis_values(0, 2) = v * v * w; // v2w
  monomial_basis_values(0, 3) = v * v * v; // v3
  monomial_basis_values(0, 4) = u * w * w; // uw2
  monomial_basis_values(0, 5) = u * v * w; // uvw
  monomial_basis_values(0, 6) = u * v * v; // uv2
  monomial_basis_values(0, 7) = u * u * w; // u2w
  monomial_basis_values(0, 8) = u * u * v; // u2v
  monomial_basis_values(0, 9) = u * u * u; // u3

  return monomial_basis_values;
}

Eigen::Matrix<double, 1, 3> CloughTocherPatch::CT_eval(const double &u,
                                                       const double &v,
                                                       const double &w) const {
  int idx = CloughTocherPatch::triangle_ind(u, v, w);

  // std::cout << "subtri_idx: " << idx << std::endl;
  Eigen::Matrix<double, 1, 10> bb_vector =
      CloughTocherPatch::monomial_basis_eval(u, v, w);

  // std::cout << "monomial: " << bb_vector << std::endl;

  Eigen::Matrix<double, 1, 3> val;
  val = bb_vector * m_CT_coeffs[idx];
  return val;
}

std::array<Eigen::Matrix<double, 10, 3>, 3>
CloughTocherPatch::get_coeffs() const {
  return m_CT_coeffs;
}