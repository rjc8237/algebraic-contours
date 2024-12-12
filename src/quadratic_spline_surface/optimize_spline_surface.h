// Copyright 2023 Adobe Research. All rights reserved.
// To view a copy of the license, visit LICENSE.md.

#pragma once

#include "affine_manifold.h"
#include "common.h"
#include "differentiable_variable.h"
#include "halfedge.h"
#include "polynomial_function.h"
#include "position_data.h"
#include <Eigen/CholmodSupport>
#include <Eigen/Sparse>

/// \file optimize_spline_surface.h
///
/// Methods to optimize a spline surface

struct OptimizationParameters {
  // Main optimization weight options
  double position_difference_factor = 1.0;
  double parametrized_quadratic_surface_mapping_factor = 1.0;

  // Weights for cone positions, normals, and gradients
  double cone_position_difference_factor =
      1.0; // fitting weight for cone vertices
  double cone_vertex_gradient_difference_factor =
      1e6; // fitting weight for cone vertex gradients

  double cone_adjacent_position_difference_factor =
      1.0; // fitting weight for vertices collapsed to a cone
  double cone_adjacent_vertex_gradient_difference_factor =
      0.0; // fitting weight for vertex gradients collapsing to a cone
  double cone_adjacent_edge_gradient_difference_factor =
      0.0; // fitting weight for vertex edge gradients collapsing to a cone

  double cone_normal_orthogonality_factor =
      1.0; // wight for encouraging orthogonality with a normal at a cone

  // TODO
  bool compute_final_energy =
      false; // Perform one more energy computation for the final value
  bool flatten_cones = false; // Perform final optimization with fixed vertices
                              // and flatten cone constraints
  int hessian_builder = 1; // 1 for assemble, 0 for autodiff, otherwise assemble
};

/// Build the quadratic energy system for the twelve-split spline with thin
/// plate, fitting, and planarity energies.
///
/// @param[in] initial_V: initial vertex positions
/// @param[in] initial_face_normals: initial vertex normals
/// @param[in] affine_manifold: mesh topology and affine manifold structure
/// @param[in] optimization_params: parameters for the spline optimization
/// @param[out] energy: energy value (i.e., constant term)
/// @param[out] derivatives: energy gradient (i.e., linear term)
/// @param[out] hessian: energy Hessian (i.e., quadratic term)
/// @param[out] hessian_inverse: solver for inverting the Hessian
void build_twelve_split_spline_energy_system(
    const MatrixXr &initial_V, const MatrixXr &initial_face_normals,
    const AffineManifold &affine_manifold,
    const OptimizationParameters &optimization_params, double &energy,
    VectorXr &derivatives, Eigen::SparseMatrix<double> &hessian,
    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> &hessian_inverse);

/// Compute the optimal per triangle position data for given vertex positions.
///
/// @param[in] V: vertex positions
/// @param[in] affine_manifold: mesh topology and affine manifold structure
/// @param[in] fit_matrix: quadratic fit energy Hessian matrix
/// @param[in] hessian_inverse: solver for inverting the energy Hessian
/// @param[out] corner_data: quadratic vertex position and derivative data
/// @param[out] midpoint_data: quadratic edge midpoint derivative data
void generate_optimized_twelve_split_position_data(
    const Eigen::MatrixXd &V, const AffineManifold &affine_manifold,
    const Eigen::SparseMatrix<double> &fit_matrix,
    const Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>>
        &hessian_inverse,
    std::vector<std::array<TriangleCornerFunctionData, 3>> &corner_data,
    std::vector<std::array<TriangleMidpointFunctionData, 3>> &midpoint_data);

/// Generate zero value gradients for a given number of vertices.
///
/// @param[in] num_vertices: number of vertices |V|
/// @param[out] gradients: |V| trivial vertex gradient matrices
void generate_zero_vertex_gradients(int num_vertices,
                                    std::vector<Matrix2x3r> &gradients);

/// Generate zero value gradients for a given number of halfedges.
///
/// @param[in] num_faces: number of faces |F|
/// @param[out] gradients: 3|F| trivial edge gradient matrices
void generate_zero_edge_gradients(
    int num_faces, std::vector<std::array<SpatialVector, 3>> &edge_gradients);

/// Given edge and opposite corner direction gradients at triangle edge
/// midpoints, extract just the opposite corner direction gradient
///
/// @param[in] edge_gradients: edge and corner directed gradients per edge
/// midpoints
/// @param[out] reduced_edge_gradients: opposite corner directed gradients per
/// edge midpoints
void convert_full_edge_gradients_to_reduced(
    const std::vector<std::array<Matrix2x3r, 3>> &edge_gradients,
    std::vector<std::array<SpatialVector, 3>> &reduced_edge_gradients);

/// Given edge direction gradients at triangle edge midpoints, append the
/// gradients in the direction of the opposite triangle corners, which are
/// determined by gradients and position data at the corners.
///
/// @param[in] reduced_edge_gradients: opposite corner directed gradients per
/// edge midpoints
/// @param[in] corner_data: quadratic vertex position and derivative data
/// @param[in] affine_manifold: mesh topology and affine manifold structure
/// @param[out] edge_gradients: edge and corner directed gradients per edge
/// midpoints
void convert_reduced_edge_gradients_to_full(
    const std::vector<std::array<SpatialVector, 3>> &reduced_edge_gradients,
    const std::vector<std::array<TriangleCornerFunctionData, 3>> &corner_data,
    const AffineManifold &affine_manifold,
    std::vector<std::array<Matrix2x3r, 3>> &edge_gradients);
