// Copyright 2023 Adobe Research. All rights reserved.
// To view a copy of the license, visit LICENSE.md.

#pragma once

#include "common.h"
#include "evaluate_surface_normal.h"
#include "optimize_spline_surface.h"
#include "position_data.h"
#include "quadratic_spline_surface_patch.h"
#include <igl/is_vertex_manifold.h>
#include <igl/writeOBJ.h>

/// Parameters for the discretization of a quadratic spline
struct SurfaceDiscretizationParameters {
  /// Number of subdivisions per triangle of the domain
  int num_subdivisions = 2;

  /// If true, compute unit length surface normal vectors
  bool normalize_surface_normals = true;
};

/// A piecewise quadratic surface.
///
/// Supports:
/// - evaluation
/// - patch and subsurface extraction
/// - triangulation
/// - sampling
/// - visualization
/// - (basic) rendering
/// - (de)serialization
class QuadraticSplineSurface {
public:
  // Index type
  typedef size_t PatchIndex;

  /// Constructor for a trivial empty surface
  QuadraticSplineSurface();

  /// Constructor from patches
  ///
  /// @param[in] patches: quadratic surface patches
  QuadraticSplineSurface(std::vector<QuadraticSplineSurfacePatch> &patches);

  /// Get the number of patches in the surface
  ///
  /// @return number of patches
  PatchIndex num_patches() const { return m_patches.size(); }

  /// Get a reference to a spline patch
  ///
  /// @return spline patch
  QuadraticSplineSurfacePatch const &get_patch(PatchIndex patch_index) const {
    return m_patches[patch_index];
  }

  /// Evaluate the surface at a given patch and domain point
  ///
  /// @param[in] patch_index: index of the patch to evaluate
  /// @param[in] domain_point: point in the patch domain to evaluate
  /// @param[out] surface_point: output point on the surface
  void evaluate_patch(const PatchIndex &patch_index,
                      const PlanarPoint &domain_point,
                      SpatialVector &surface_point) const {
    get_patch(patch_index).evaluate(domain_point, surface_point);
  }

  /// Evaluate the surface normal at a given patch and domain point
  ///
  /// @param[in] patch_index: index of the patch to evaluate
  /// @param[in] domain_point: point in the patch domain to evaluate
  /// @param[out] surface_point: output point on the surface
  void evaluate_patch_normal(const PatchIndex &patch_index,
                             const PlanarPoint &domain_point,
                             SpatialVector &surface_normal) const {
    get_patch(patch_index).evaluate_normal(domain_point, surface_normal);
  }

  /// Determine if the surface is empty
  ///
  /// @return true iff the surface is empty
  bool empty() const { return m_patches.empty(); }

  /// Clear the surface
  virtual void clear();

  /// Generate a subsurface with the given patch indices.
  ///
  /// @param[in] patch_indices: indices of the patches to keep
  /// @return subsurface with the given patches
  QuadraticSplineSurface
  subsurface(const std::vector<PatchIndex> &patch_indices) const;

  /// Triangulate a given patch
  ///
  /// @param[in] patch_index: patch to triangulate
  /// @param[in] num_refinements: number of refinements for the triangulation
  /// @param[out] V: vertices of the triangulation
  /// @param[out] F: faces of the triangulation
  /// @param[out] N: vertex normals
  void triangulate_patch(const PatchIndex &patch_index, int num_refinements,
                         Eigen::MatrixXd &V, Eigen::MatrixXi &F,
                         Eigen::MatrixXd &N) const;

  /// Triangulate the surface.
  ///
  /// @param[in] surface_disc_params: discretization parameters
  /// @param[out] V: vertices of the triangulation
  /// @param[out] F: faces of the triangulation
  /// @param[out] N: vertex normals
  void discretize(const SurfaceDiscretizationParameters &surface_disc_params,
                  Eigen::MatrixXd &V, Eigen::MatrixXi &F,
                  Eigen::MatrixXd &N) const;

  /// Triangulate the surface.
  ///
  /// @param[in] surface_disc_params: discretization parameters
  /// @return vertices of the triangulation
  /// @return faces of the triangulation
  /// @return vertex normals
  std::tuple<Eigen::MatrixXd, // V
             Eigen::MatrixXi, // F
             Eigen::MatrixXd  // N
             >
  discretize(const SurfaceDiscretizationParameters &surface_disc_params) const;

  /// Discretize all patch boundaries as polylines
  ///
  /// @param[out] points: list of polyline points
  /// @param[out] polylines: list of lists of polyline edges
  void
  discretize_patch_boundaries(std::vector<SpatialVector> &points,
                              std::vector<std::vector<int>> &polylines) const;

  /// Save the triangulated surface as an obj
  ///
  /// @param[in] filename: filepath to save the obj
  void save_obj(const std::string &filename) const;

  /// Add the surface to the viewer
  ///
  /// @param[in] color: color for the surface in the viewer
  /// @param[in] num_subdivisions: number of subdivisions for the surface
  void add_surface_to_viewer(Eigen::Matrix<double, 3, 1> color = SKY_BLUE,
                             int num_subdivisions = DISCRETIZATION_LEVEL) const;

  /// View the surface
  ///
  /// @param[in] color: color for the surface in the viewer
  /// @param[in] num_subdivisions: number of subdivisions for the surface
  virtual void view(Eigen::Matrix<double, 3, 1> color = SKY_BLUE,
                    int num_subdivisions = DISCRETIZATION_LEVEL) const;

  /// Save a screenshot of the surface in the viewer
  ///
  /// @param[in] filename: file to save the screenshot
  /// @param[in] camera_position: camera position for the screenshot
  /// @param[in] camera_target: camera target for the screenshot
  /// @param[in] use_orthographic: use orthographic perspective if true
  void screenshot(const std::string &filename,
                  SpatialVector camera_position = SpatialVector(0, 0, 2),
                  SpatialVector camera_target = SpatialVector(0, 0, 0),
                  bool use_orthographic = false) const;

  /// Serialize the surface
  ///
  /// @param[in] out: output stream for the surface
  void serialize(std::ostream &out) const;

  /// Deserialize a surface
  ///
  /// @param[in] in: input stream for the surface
  void deserialize(std::istream &in);

  /// Write the surface serialization to file
  ///
  /// @param[in] filename: file path for the serialized surface
  void write_spline(const std::string &filename) const;

  /// Read a surface serialization from file
  ///
  /// @param[in] filename: file path for the serialized surface
  void read_spline(const std::string &filename);

  /// Compute hash tables for the surface
  void compute_patch_hash_tables();

  /// Compute the hash indices of a point in the plane
  ///
  /// @param[in] point: point in the plane
  /// @return pair
  std::pair<int, int> compute_hash_indices(const PlanarPoint &point) const;

  /// Hash table data
  std::vector<int> hash_table[HASH_TABLE_SIZE][HASH_TABLE_SIZE];
  std::vector<std::vector<std::pair<int, int>>> reverse_hash_table;

  /// Hash table parameters
  double patches_bbox_x_min, patches_bbox_x_max, patches_bbox_y_min,
      patches_bbox_y_max;
  double hash_x_interval;
  double hash_y_interval;

protected:
  bool is_valid_patch_index(PatchIndex patch_index) const;

  void compute_patches_bbox();

  std::vector<QuadraticSplineSurfacePatch> m_patches;

  //-------------------- code for c1 meshing --------------------//
public:
  void write_cubic_nodes_to_msh(std::string filename);
  void write_cubic_nodes_to_obj(std::string filename);

  void write_corner_patch_points_to_obj(std::string filename);

  //-------------------------------------------------------------//
};
