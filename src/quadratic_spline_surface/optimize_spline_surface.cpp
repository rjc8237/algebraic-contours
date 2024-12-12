// Copyright 2023 Adobe Research. All rights reserved.
// To view a copy of the license, visit LICENSE.md.

#include "optimize_spline_surface.h"

#include "compute_local_twelve_split_hessian.h"
#include "planarH.h"
#include "powell_sabin_local_to_global_indexing.h"

// Build the hessian for the local fitting energy
//
// This is a diagonal matrix with special weights for cones and cone adjacent
// vertices
Eigen::Matrix<double, 12, 12>
build_local_fit_hessian(const std::array<bool, 3> &is_cone,
                        const std::array<bool, 3> &is_cone_adjacent,
                        const OptimizationParameters &optimization_params) {
  Eigen::Matrix<double, 12, 12> H_f;
  H_f.fill(0);

  // Check for cone collapsing vertices
  for (size_t i = 0; i < 3; ++i) {
    size_t vi = generate_local_vertex_position_variable_index(
        i, 0, 1); // local vertex index

    // Weights for cone vertices
    if (is_cone[i]) {
      // Add increased weight to the cone position fit
      spdlog::trace("Weighting cone vertices by {}",
                    optimization_params.cone_position_difference_factor);
      H_f(vi, vi) = optimization_params.cone_position_difference_factor;

      // Add cone gradient fitting term
      spdlog::info("Weighting cone gradients by {}",
                   optimization_params.cone_vertex_gradient_difference_factor);
      size_t g1i = generate_local_vertex_gradient_variable_index(
          i, 0, 0, 1); // local first gradient index
      size_t g2i = generate_local_vertex_gradient_variable_index(
          i, 1, 0, 1); // local second gradient index
      H_f(g1i, g1i) =
          optimization_params.cone_vertex_gradient_difference_factor;
      H_f(g2i, g2i) =
          optimization_params.cone_vertex_gradient_difference_factor;
    }
    // Weights for cone adjacent vertices (which can be collapsed to the cone)
    else if (is_cone_adjacent[i]) {
      // Add increased weight to the cone adjacent position fit
      spdlog::trace(
          "Weighting cone adjacent vertices by {}",
          optimization_params.cone_adjacent_position_difference_factor);
      H_f(vi, vi) =
          optimization_params.cone_adjacent_position_difference_factor;

      // Add cone adjacent vertex gradient fitting term
      spdlog::trace(
          "Weighting cone adjacent gradients by {}",
          optimization_params.cone_adjacent_vertex_gradient_difference_factor);
      size_t g1i = generate_local_vertex_gradient_variable_index(
          i, 0, 0, 1); // local first gradient index
      size_t g2i = generate_local_vertex_gradient_variable_index(
          i, 1, 0, 1); // local second gradient index
      H_f(g1i, g1i) =
          optimization_params.cone_adjacent_vertex_gradient_difference_factor;
      H_f(g2i, g2i) =
          optimization_params.cone_adjacent_vertex_gradient_difference_factor;
    }
    // Default fitting weight is 1.0
    else {
      H_f(vi, vi) = 1.0;
    }
  }

  // Check for edges collapsing to a cone and add weight
  for (size_t i = 0; i < 3; ++i) {
    size_t vj = (i + 1) % 3; // next local vertex index
    size_t vk = (i + 2) % 3; // prev local vertex index
    if ((is_cone_adjacent[vj]) && (is_cone_adjacent[vk])) {
      // Add cone adjacent adjacent edge gradient fit
      spdlog::trace(
          "Weighting cone edge gradients by {}",
          optimization_params.cone_adjacent_edge_gradient_difference_factor);
      size_t gjk = generate_local_edge_gradient_variable_index(
          i, 0, 1); // local first gradient index
      H_f(gjk, gjk) =
          optimization_params.cone_adjacent_edge_gradient_difference_factor;
    }
  }

  return H_f;
}

// Build the hessian for the planar normal constraint term for cone adjacent
// vertices
//
// WARNING: Unlike the other hessians, which are 12x12 matrices assembled per
// x,y,z coordinate and combined into a 36x3 block matrix, this Hessian has
// mixed coordinate terms and is thus directly 36x36
Eigen::Matrix<double, 36, 36> build_planar_constraint_hessian(
    const Eigen::Matrix<double, 3, 2> &uv,
    const std::array<Matrix2x2r, 3> &corner_to_corner_uv_positions,
    const std::array<bool, 3> &reverse_edge_orientations,
    const SpatialVector &normal) {
  // Build planar hessian array for derived derivative quantities
  double planarH_array[36][36];
  planarHfun(normal[0], normal[1], normal[2], planarH_array);

  // Convert planar hessian matrix to an Eigen matrix
  Eigen::Matrix<double, 36, 36> planarH;
  for (size_t i = 0; i < 36; ++i) {
    for (size_t j = 0; j < 36; ++j) {
      planarH(i, j) = planarH_array[i][j];
    }
  }

  // Build C_gl matrix
  Eigen::Matrix<double, 12, 12> C_gl =
      get_C_gl(uv, corner_to_corner_uv_positions, reverse_edge_orientations);

  // Make block diagonal C_gl matrix
  Eigen::Matrix<double, 36, 36> C_gl_diag;
  C_gl_diag.setZero();
  C_gl_diag.block(0, 0, 12, 12) = C_gl;
  C_gl_diag.block(12, 12, 12, 12) = C_gl;
  C_gl_diag.block(24, 24, 12, 12) = C_gl;

  // Build the planar constraint Hessian with indexing so that DoF per
  // coordinate are contiguous
  Eigen::Matrix<double, 36, 36> H_p_permuted =
      0.5 * C_gl_diag.transpose() * planarH * C_gl_diag;

  // Reindex so that coordinates per DoF are contiguous
  Eigen::Matrix<double, 36, 36> H_p;
  H_p.fill(0);
  for (size_t ri = 0; ri < 12; ++ri) {
    for (size_t rj = 0; rj < 3; ++rj) {
      for (size_t ci = 0; ci < 12; ++ci) {
        for (size_t cj = 0; cj < 3; ++cj) {
          H_p(3 * ri + rj, 3 * ci + cj) =
              H_p_permuted(12 * rj + ri, 12 * cj + ci);
        }
      }
    }
  }

  return H_p;
}

// Structure for the energy quadratic Hessian data
struct LocalHessianData {
  Eigen::Matrix<double, 12, 12> H_f; // fitting term hessian
  Eigen::Matrix<double, 12, 12> H_s; // smoothness term hessian
  Eigen::Matrix<double, 36, 36> H_p; // planarity term hessian
  double w_f;                        // fitting term weight
  double w_s;                        // smoothness term weight
  double w_p;                        // planarity term weight
};

// Structure for the energy quadratic local degree of freedom data
struct LocalDOFData {
  Eigen::Matrix<double, 12, 3> r_alpha_0;    // initial local DOF
  Eigen::Matrix<double, 12, 3> r_alpha;      // local DOF
  Eigen::Matrix<double, 36, 1> r_alpha_flat; // flattened local DOF
};

// Assemble the energy quadratic Hessian data
void generate_local_hessian_data(
    const std::array<PlanarPoint, 3> &face_vertex_uv_positions,
    const std::array<Matrix2x2r, 3> &corner_to_corner_uv_positions,
    const std::array<bool, 3> &reverse_edge_orientations,
    const std::array<bool, 3> &is_cone,
    const std::array<bool, 3> &is_cone_adjacent,
    const SpatialVector &face_normal,
    const OptimizationParameters &optimization_params,
    LocalHessianData &local_hessian_data) {
  // Build uv from global positions
  Eigen::Matrix<double, 3, 2> uv;
  uv.row(0) = face_vertex_uv_positions[0];
  uv.row(1) = face_vertex_uv_positions[1];
  uv.row(2) = face_vertex_uv_positions[2];

  // H_s: local smoothness hessian
  local_hessian_data.H_s = build_local_smoothness_hessian(
      uv, corner_to_corner_uv_positions, reverse_edge_orientations);

  // H_f: position fit hessian
  local_hessian_data.H_f =
      build_local_fit_hessian(is_cone, is_cone_adjacent, optimization_params);

  // H_p: planar fitting term
  // DEPRECATED
  if (optimization_params.cone_normal_orthogonality_factor != 0.0) {
    local_hessian_data.H_p =
        build_planar_constraint_hessian(uv, corner_to_corner_uv_positions,
                                        reverse_edge_orientations, face_normal);
  } else {
    local_hessian_data.H_p.setZero();
  }

  // w_s: smoothing weight
  local_hessian_data.w_s =
      optimization_params.parametrized_quadratic_surface_mapping_factor;

  // w_f: fitting weight
  local_hessian_data.w_f = optimization_params.position_difference_factor;

  // w_p: cone planar constraint weight
  local_hessian_data.w_p = optimization_params.cone_normal_orthogonality_factor;
}

// Assemble the local degree of freedom data
void generate_local_dof_data(
    const std::array<SpatialVector, 3> &initial_vertex_positions_T,
    const std::array<SpatialVector, 3> &vertex_positions_T,
    const std::array<Matrix2x3r, 3> &vertex_gradients_T,
    const std::array<SpatialVector, 3> &edge_gradients_T,
    LocalDOFData &local_dof_data) {
  // r_alpha_0: fitting values
  // WARNING: Only fitting to zero implemented for gradients
  // TODO: Implement fitting for creases
  local_dof_data.r_alpha_0.fill(0);
  local_dof_data.r_alpha_0.row(0) = initial_vertex_positions_T[0];
  local_dof_data.r_alpha_0.row(3) = initial_vertex_positions_T[1];
  local_dof_data.r_alpha_0.row(6) = initial_vertex_positions_T[2];
  spdlog::trace("Fit values:\n{}", local_dof_data.r_alpha_0);

  // r_alpha: input values
  local_dof_data.r_alpha.row(0) = vertex_positions_T[0];
  local_dof_data.r_alpha.row(1) = vertex_gradients_T[0].row(0);
  local_dof_data.r_alpha.row(2) = vertex_gradients_T[0].row(1);
  local_dof_data.r_alpha.row(3) = vertex_positions_T[1];
  local_dof_data.r_alpha.row(4) = vertex_gradients_T[1].row(0);
  local_dof_data.r_alpha.row(5) = vertex_gradients_T[1].row(1);
  local_dof_data.r_alpha.row(6) = vertex_positions_T[2];
  local_dof_data.r_alpha.row(7) = vertex_gradients_T[2].row(0);
  local_dof_data.r_alpha.row(8) = vertex_gradients_T[2].row(1);
  local_dof_data.r_alpha.row(9) = edge_gradients_T[0];
  local_dof_data.r_alpha.row(10) = edge_gradients_T[1];
  local_dof_data.r_alpha.row(11) = edge_gradients_T[2];
  spdlog::trace("Input values:\n{}", local_dof_data.r_alpha);

  // Also flatten r_alpha for the normal constraint term
  for (int i = 0; i < 12; i++) {
    for (int j = 0; j < 3; j++) {
      local_dof_data.r_alpha_flat[3 * i + j] = local_dof_data.r_alpha(i, j);
    }
  }
}

// Compute the local twelve split energy quadratic from Hessian and local
// degree of freedom data
void compute_local_twelve_split_energy_quadratic(
    const LocalHessianData &local_hessian_data,
    const LocalDOFData &local_dof_data, double &local_energy,
    TwelveSplitGradient &local_derivatives, TwelveSplitHessian &local_hessian) {
  // Extract local hessian data
  const Eigen::Matrix<double, 12, 12> &H_f = local_hessian_data.H_f;
  const Eigen::Matrix<double, 12, 12> &H_s = local_hessian_data.H_s;
  const Eigen::Matrix<double, 36, 36> &H_p = local_hessian_data.H_p;
  double w_f = local_hessian_data.w_f;
  double w_s = local_hessian_data.w_s;
  double w_p = local_hessian_data.w_p;

  // Extract local degrees of freedom data
  const Eigen::Matrix<double, 12, 3> &r_alpha_0 = local_dof_data.r_alpha_0;
  const Eigen::Matrix<double, 12, 3> &r_alpha = local_dof_data.r_alpha;
  const Eigen::Matrix<double, 36, 1> &r_alpha_flat =
      local_dof_data.r_alpha_flat;

  // full local 12x12 hessian (only smoothness and fitting terms)
  Eigen::Matrix<double, 12, 12> local_hessian_12x12 =
      2 * (w_s * H_s + w_f * H_f);

  // Add smoothness and fitting term blocks to the full local hessian per
  // coordinate
  local_hessian.fill(0);
  for (int i = 0; i < 12; i++) {
    for (int j = 0; j < 12; j++) {
      for (int k = 0; k < 3; k++) {
        local_hessian(3 * i + k, 3 * j + k) = local_hessian_12x12(i, j);
      }
    }
  }

  // Add 36x36 planar constraint term to the local hessian
  local_hessian += 2.0 * w_p * H_p;

  // build per coordinate gradients for smoothness and fit terms
  Eigen::Matrix<double, 12, 3> g_alpha;
  g_alpha = 2 * (w_s * H_s) * r_alpha;
  spdlog::trace("Block gradient after adding smoothness term:\n{}", g_alpha);
  g_alpha += 2 * (w_f * H_f) * (r_alpha - r_alpha_0);
  spdlog::trace("Block gradient after adding fit term:\n{}", g_alpha);

  // Combine per coordinate gradients into the local
  for (int i = 0; i < 12; i++) {
    local_derivatives[3 * i] = g_alpha(i, 0);
    local_derivatives[3 * i + 1] = g_alpha(i, 1);
    local_derivatives[3 * i + 2] = g_alpha(i, 2);
  }

  // Add planar constraint term
  local_derivatives += 2.0 * w_p * H_p * r_alpha_flat;

  // Add smoothness term
  double smoothness_term = 0.0;
  for (int i = 0; i < 3; i++) {
    smoothness_term +=
        (r_alpha.col(i).transpose() * (w_s * H_s) * r_alpha.col(i))(0, 0);
  }
  spdlog::trace("Smoothness term is {}", smoothness_term);

  // Add fit term
  double fit_term = 0.0;
  for (int i = 0; i < 3; i++) {
    Eigen::Matrix<double, 12, 1> r_alpha_diff =
        r_alpha.col(i) - r_alpha_0.col(i);
    fit_term += (r_alpha_diff.transpose() * (w_f * H_f) * r_alpha_diff)(0, 0);
  }
  spdlog::trace("Fit term is {}", fit_term);

  // Add planar fitting term
  double planar_term = 0.0;
  planar_term += r_alpha_flat.transpose() * (w_p * H_p) * r_alpha_flat;
  spdlog::trace("Planar orthogonality term is {}", planar_term);

  // Compute final energy
  local_energy = smoothness_term + fit_term + planar_term;
  spdlog::trace("Final energy is {}", local_energy);
}

// Helper function to cyclically shift an array of three elements
template <typename T> void shift_array(std::array<T, 3> &arr, int shift) {
  std::array<T, 3> arr_copy = arr;
  for (int i = 0; i < 3; ++i) {
    arr[i] = arr_copy[(i + shift) % 3];
  }
}

// Method to cyclically shift the indexing of all energy quadratic data arrays
// for triangle vertex values
void shift_local_energy_quadratic_vertices(
    std::array<SpatialVector, 3> &vertex_positions_T,
    std::array<Matrix2x3r, 3> &vertex_gradients_T,
    std::array<SpatialVector, 3> &edge_gradients_T,
    std::array<SpatialVector, 3> &initial_vertex_positions_T,
    std::array<PlanarPoint, 3> &face_vertex_uv_positions,
    std::array<Matrix2x2r, 3> &corner_to_corner_uv_positions,
    std::array<bool, 3> &reverse_edge_orientations,
    std::array<bool, 3> &is_cone, std::array<bool, 3> &is_cone_adjacent,
    std::array<int, 3> &face_global_vertex_indices,
    std::array<int, 3> &face_global_edge_indices, int shift) {
  shift_array(vertex_positions_T, shift);
  shift_array(vertex_gradients_T, shift);
  shift_array(edge_gradients_T, shift);
  shift_array(initial_vertex_positions_T, shift);
  shift_array(face_vertex_uv_positions, shift);
  shift_array(corner_to_corner_uv_positions, shift);
  shift_array(reverse_edge_orientations, shift);
  shift_array(is_cone, shift);
  shift_array(is_cone_adjacent, shift);
  shift_array(face_global_vertex_indices, shift);
  shift_array(face_global_edge_indices, shift);
}

// Compute the energy system for a twelve-split spline
void compute_twelve_split_energy_quadratic(
    const std::vector<SpatialVector> &vertex_positions,
    const std::vector<Matrix2x3r> &vertex_gradients,
    const std::vector<std::array<SpatialVector, 3>> &edge_gradients,
    const std::vector<int> &global_vertex_indices,
    const std::vector<std::array<int, 3>> &global_edge_indices,
    const std::vector<SpatialVector> &initial_vertex_positions,
    const Eigen::MatrixXd &initial_face_normals, const AffineManifold &manifold,
    const OptimizationParameters &optimization_params, double &energy,
    VectorXr &derivatives, Eigen::SparseMatrix<double> &hessian,
    int num_variable_vertices, int num_variable_edges) {
  int num_independent_variables =
      9 * num_variable_vertices + 3 * num_variable_edges;
  energy = 0;
  derivatives.setZero(num_independent_variables);
  std::vector<Eigen::Triplet<double>> hessian_entries;
  hessian_entries.reserve(36 * num_independent_variables);

  for (AffineManifold::Index face_index = 0; face_index < manifold.num_faces();
       ++face_index) {
    // Get face vertices
    Eigen::MatrixXi const &F = manifold.get_faces();
    int i = F(face_index, 0);
    int j = F(face_index, 1);
    int k = F(face_index, 2);

    // Bundle relevant global variables into per face local vectors
    std::array<SpatialVector, 3> initial_vertex_positions_T, vertex_positions_T,
        edge_gradients_T;
    std::array<Matrix2x3r, 3> vertex_gradients_T;
    build_face_variable_vector(vertex_positions, i, j, k, vertex_positions_T);
    build_face_variable_vector(vertex_gradients, i, j, k, vertex_gradients_T);
    edge_gradients_T = edge_gradients[face_index];
    build_face_variable_vector(initial_vertex_positions, i, j, k,
                               initial_vertex_positions_T);

    // Get the global uv values for the face vertices
    std::array<PlanarPoint, 3> face_vertex_uv_positions;
    manifold.get_face_global_uv(face_index, face_vertex_uv_positions);

    // Get corner uv positions for the given face corners.
    // NOTE: These may differ from the edge difference vectors computed from the
    // global uv by a rotation per vertex due to the local layouts performed at
    // each vertex. Since vertex gradients are defined in terms of these local
    // vertex charts, we must use these directions when computing edge direction
    // gradients from the vertex uv gradients.
    std::array<Matrix2x2r, 3> corner_to_corner_uv_positions;
    manifold.get_face_corner_charts(face_index, corner_to_corner_uv_positions);

    // Get edge orientations
    // NOTE: The edge frame is oriented so that one basis vector points along
    // the edge counterclockwise and the other points perpendicular into the
    // interior of the triangle. If the given face is the bottom face in the
    // edge chart, the sign of the midpoint gradient needs to be reversed.
    std::array<bool, 3> reverse_edge_orientations;
    for (int i = 0; i < 3; ++i) {
      EdgeManifoldChart const &chart = manifold.get_edge_chart(face_index, i);
      reverse_edge_orientations[i] = (chart.top_face_index != face_index);
    }

    // Mark cone vertices
    std::array<bool, 3> is_cone;
    for (int i = 0; i < 3; ++i) {
      int vi = F(face_index, i);
      is_cone[i] = manifold.get_vertex_chart(vi).is_cone;
    }

    // Mark cone adjacent vertices
    std::array<bool, 3> is_cone_adjacent;
    for (int i = 0; i < 3; ++i) {
      int vi = F(face_index, i);
      is_cone_adjacent[i] = manifold.get_vertex_chart(vi).is_cone_adjacent;
    }

    // Get global indices of the local vertex and edge DOFs
    std::array<int, 3> face_global_vertex_indices, face_global_edge_indices;
    build_face_variable_vector(global_vertex_indices, i, j, k,
                               face_global_vertex_indices);
    face_global_edge_indices = global_edge_indices[face_index];

    // Check if an edge is collapsing and make sure any collapsing edges have
    // local vertex indices 0 and 1
    // WARNING: This is a somewhat fragile operation that must occur after all
    // of these arrays are build and before the local to global map is built
    // and is not necessary in the current framework used in the paper but is
    // for some deprecated experimental methods
    bool is_cone_adjacent_face = false;
    for (int i = 0; i < 3; ++i) {
      if (is_cone[(i + 2) % 3]) {
        shift_local_energy_quadratic_vertices(
            vertex_positions_T, vertex_gradients_T, edge_gradients_T,
            initial_vertex_positions_T, face_vertex_uv_positions,
            corner_to_corner_uv_positions, reverse_edge_orientations, is_cone,
            is_cone_adjacent, face_global_vertex_indices,
            face_global_edge_indices, i);
        is_cone_adjacent_face = true;
        break;
      }
    }

    // Get normal for the face
    SpatialVector normal;
    normal.setZero();

    if (is_cone_adjacent_face) {
      normal = initial_face_normals.row(face_index);
      spdlog::trace("Weighting by normal {}", normal.transpose());
    }

    // Get local to global map
    std::vector<int> local_to_global_map;
    generate_twelve_split_local_to_global_map(
        face_global_vertex_indices, face_global_edge_indices,
        num_variable_vertices, local_to_global_map);

    // Compute local hessian data
    LocalHessianData local_hessian_data;
    generate_local_hessian_data(
        face_vertex_uv_positions, corner_to_corner_uv_positions,
        reverse_edge_orientations, is_cone, is_cone_adjacent, normal,
        optimization_params, local_hessian_data);

    // Compute local degree of freedom data
    LocalDOFData local_dof_data;
    generate_local_dof_data(initial_vertex_positions_T, vertex_positions_T,
                            vertex_gradients_T, edge_gradients_T,
                            local_dof_data);

    // Compute the local energy quadratic system for the face
    double local_energy;
    TwelveSplitGradient local_derivatives;
    TwelveSplitHessian local_hessian;
    compute_local_twelve_split_energy_quadratic(
        local_hessian_data, local_dof_data, local_energy, local_derivatives,
        local_hessian);

    // Update energy quadratic with the new face energy
    update_energy_quadratic<TwelveSplitGradient, TwelveSplitHessian>(
        local_energy, local_derivatives, local_hessian, local_to_global_map,
        energy, derivatives, hessian_entries);
  }

  // Set hessian from the triplets
  hessian.resize(num_independent_variables, num_independent_variables);
  hessian.setFromTriplets(hessian_entries.begin(), hessian_entries.end());
}

void convert_full_edge_gradients_to_reduced(
    const std::vector<std::array<Matrix2x3r, 3>> &edge_gradients,
    std::vector<std::array<SpatialVector, 3>> &reduced_edge_gradients) {
  int num_faces = edge_gradients.size();
  reduced_edge_gradients.resize(num_faces);
  for (int i = 0; i < num_faces; ++i) {
    for (int j = 0; j < 3; ++j) {
      reduced_edge_gradients[i][j] = edge_gradients[i][j].row(1);
    }
  }
}

void convert_reduced_edge_gradients_to_full(
    const std::vector<std::array<SpatialVector, 3>> &reduced_edge_gradients,
    const std::vector<std::array<TriangleCornerFunctionData, 3>> &corner_data,
    const AffineManifold &affine_manifold,
    std::vector<std::array<Matrix2x3r, 3>> &edge_gradients) {
  Eigen::MatrixXi const &F = affine_manifold.get_faces();
  int num_faces = reduced_edge_gradients.size();

  // Compute the first gradient and copy the second for each edge
  edge_gradients.resize(num_faces);
  for (int i = 0; i < num_faces; ++i) {
    for (int j = 0; j < 3; ++j) {
      EdgeManifoldChart const &chart = affine_manifold.get_edge_chart(i, j);
      int f_top = chart.top_face_index;
      if (f_top != i)
        continue; // Only process top faces of edge charts to prevent redundancy

      // Get midpoint position and derivative along the edge
      SpatialVector midpoint;
      SpatialVector midpoint_edge_gradient;
      compute_edge_midpoint_with_gradient(corner_data[i][(j + 1) % 3],
                                          corner_data[i][(j + 2) % 3], midpoint,
                                          midpoint_edge_gradient);

      // Copy the gradients
      edge_gradients[i][j].resize(2, 3);
      edge_gradients[i][j].row(0) = midpoint_edge_gradient;
      edge_gradients[i][j].row(1) = reduced_edge_gradients[i][j];

      // If the edge isn't on the boundary, set the other face corner
      // corresponding to it
      if (!chart.is_boundary) {
        int f_bottom = chart.bottom_face_index;
        int v_bottom = chart.bottom_vertex_index;
        int j_bottom = find_face_vertex_index(F.row(f_bottom), v_bottom);
        edge_gradients[f_bottom][j_bottom] = edge_gradients[i][j];
      }
    }
  }
}

void build_twelve_split_spline_energy_system(
    const MatrixXr &initial_V, const MatrixXr &initial_face_normals,
    const AffineManifold &affine_manifold,
    const OptimizationParameters &optimization_params, double &energy,
    VectorXr &derivatives, Eigen::SparseMatrix<double> &hessian,
    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> &hessian_inverse) {
  int num_vertices = initial_V.rows();
  int num_faces = affine_manifold.num_faces();

  // Build halfedge
  std::vector<std::pair<Eigen::Index, Eigen::Index>> he_to_corner =
      affine_manifold.get_he_to_corner();
  Halfedge const &halfedge = affine_manifold.get_halfedge();
  int num_edges = halfedge.num_edges();

  // Assume all vertices and edges are variable
  std::vector<int> fixed_vertices(0);
  std::vector<int> fixed_edges(0);
  std::vector<int> variable_vertices, variable_edges;
  index_vector_complement<int>(fixed_vertices, num_vertices, variable_vertices);
  index_vector_complement<int>(fixed_edges, num_edges, variable_edges);

  // Get variable counts
  int num_variable_vertices = variable_vertices.size();
  int num_variable_edges = variable_edges.size();

  // Initialize variables to optimize
  std::vector<SpatialVector> vertex_positions(num_vertices);
  std::vector<SpatialVector> initial_vertex_positions(num_vertices);
  for (int i = 0; i < num_vertices; ++i) {
    vertex_positions[i] = initial_V.row(i);
    initial_vertex_positions[i] = initial_V.row(i);
  }
  std::vector<Matrix2x3r> vertex_gradients;
  std::vector<std::array<SpatialVector, 3>> edge_gradients;
  generate_zero_vertex_gradients(num_vertices, vertex_gradients);
  generate_zero_edge_gradients(num_faces, edge_gradients);

  // Build vertex variable indices
  std::vector<int> global_vertex_indices;
  build_variable_vertex_indices_map(num_vertices, variable_vertices,
                                    global_vertex_indices);

  // Build edge variable indices
  std::vector<std::array<int, 3>> global_edge_indices;
  build_variable_edge_indices_map(num_faces, variable_edges, halfedge,
                                  he_to_corner, global_edge_indices);

  // Build energy for the affine manifold
  compute_twelve_split_energy_quadratic(
      vertex_positions, vertex_gradients, edge_gradients, global_vertex_indices,
      global_edge_indices, initial_vertex_positions, initial_face_normals,
      affine_manifold, optimization_params, energy, derivatives, hessian,
      num_variable_vertices, num_variable_edges);

  // Build the inverse
  hessian_inverse.compute(hessian);
}

void optimize_twelve_split_spline_surface(
    const MatrixXr &initial_V, const AffineManifold &affine_manifold,
    const Halfedge &halfedge,
    const std::vector<std::pair<Eigen::Index, Eigen::Index>> &he_to_corner,
    const std::vector<int> &variable_vertices,
    const std::vector<int> &variable_edges,
    const Eigen::SparseMatrix<double> &fit_matrix,
    const Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>>
        &hessian_inverse,
    MatrixXr &optimized_V, std::vector<Matrix2x3r> &optimized_vertex_gradients,
    std::vector<std::array<SpatialVector, 3>> &optimized_edge_gradients) {
  // Get variable counts
  int num_vertices = initial_V.rows();
  int num_faces = affine_manifold.num_faces();

  // Initialize variables to optimize
  std::vector<SpatialVector> vertex_positions(num_vertices);
  std::vector<SpatialVector> initial_vertex_positions(num_vertices);
  for (int i = 0; i < num_vertices; ++i) {
    vertex_positions[i] = initial_V.row(i);
    initial_vertex_positions[i] = initial_V.row(i);
  }
  std::vector<Matrix2x3r> vertex_gradients;
  std::vector<std::array<SpatialVector, 3>> edge_gradients;
  generate_zero_vertex_gradients(num_vertices, vertex_gradients);
  generate_zero_edge_gradients(num_faces, edge_gradients);

  // Build variable values gradient as H
  VectorXr initial_variable_values;
  generate_twelve_split_variable_value_vector(
      vertex_positions, vertex_gradients, edge_gradients, variable_vertices,
      variable_edges, halfedge, he_to_corner, initial_variable_values);
  spdlog::trace("Initial variable value vector:\n{}", initial_variable_values);

  // Solve hessian system to get optimized values
  VectorXr right_hand_side = fit_matrix * initial_variable_values;
  VectorXr optimized_variable_values = hessian_inverse.solve(right_hand_side);

  // Update variables
  std::vector<SpatialVector> optimized_vertex_positions = vertex_positions;
  optimized_vertex_gradients = vertex_gradients;
  optimized_edge_gradients = edge_gradients;
  update_position_variables(optimized_variable_values, variable_vertices,
                            optimized_vertex_positions);
  update_vertex_gradient_variables(optimized_variable_values, variable_vertices,
                                   optimized_vertex_gradients);
  update_edge_gradient_variables(optimized_variable_values, variable_vertices,
                                 variable_edges, halfedge, he_to_corner,
                                 optimized_edge_gradients);

  // Copy variable values to constants
  optimized_V.resize(num_vertices, 3);
  for (int i = 0; i < num_vertices; ++i) {
    optimized_V.row(i) = optimized_vertex_positions[i];
  }
}

void generate_optimized_twelve_split_position_data(
    const Eigen::MatrixXd &V, const AffineManifold &affine_manifold,
    const Eigen::SparseMatrix<double> &fit_matrix,
    const Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>>
        &hessian_inverse,
    std::vector<std::array<TriangleCornerFunctionData, 3>> &corner_data,
    std::vector<std::array<TriangleMidpointFunctionData, 3>> &midpoint_data) {
  int num_vertices = V.rows();

  // Build halfedge
  std::vector<std::pair<Eigen::Index, Eigen::Index>> const &he_to_corner =
      affine_manifold.get_he_to_corner();
  Halfedge const &halfedge = affine_manifold.get_halfedge();
  int num_edges = halfedge.num_edges();

  // Assume all vertices and edges are variable
  std::vector<int> fixed_vertices(0);
  std::vector<int> fixed_edges(0);
  std::vector<int> variable_vertices, variable_edges;
  index_vector_complement<int>(fixed_vertices, num_vertices, variable_vertices);
  index_vector_complement<int>(fixed_edges, num_edges, variable_edges);

  // Run optimization
  MatrixXr optimized_V;
  std::vector<Matrix2x3r> optimized_vertex_gradients;
  std::vector<std::array<SpatialVector, 3>> optimized_reduced_edge_gradients;
  optimize_twelve_split_spline_surface(
      V, affine_manifold, halfedge, he_to_corner, variable_vertices,
      variable_edges, fit_matrix, hessian_inverse, optimized_V,
      optimized_vertex_gradients, optimized_reduced_edge_gradients);

  // Build corner position data from the optimized gradients
  generate_affine_manifold_corner_data(optimized_V, affine_manifold,
                                       optimized_vertex_gradients, corner_data);

  // Build the full edge gradients with first gradient determined by the corner
  // position data
  std::vector<std::array<Matrix2x3r, 3>> optimized_edge_gradients;
  convert_reduced_edge_gradients_to_full(optimized_reduced_edge_gradients,
                                         corner_data, affine_manifold,
                                         optimized_edge_gradients);

  // Build midpoint position data from the optimized gradients
  generate_affine_manifold_midpoint_data(
      affine_manifold, optimized_edge_gradients, midpoint_data);
}

void generate_zero_vertex_gradients(int num_vertices,
                                    std::vector<Matrix2x3r> &gradients) {
  // Set the zero gradient for each vertex
  gradients.resize(num_vertices);
  for (int i = 0; i < num_vertices; ++i) {
    gradients[i].setZero(2, 3);
  }
}

void generate_zero_edge_gradients(
    int num_faces, std::vector<std::array<SpatialVector, 3>> &edge_gradients) {
  // Set the zero gradient for each vertex
  edge_gradients.resize(num_faces);
  for (int i = 0; i < num_faces; ++i) {
    for (int j = 0; j < 3; ++j) {
      edge_gradients[i][j].setZero(3);
    }
  }
}
