// Copyright 2023 Adobe Research. All rights reserved.
// To view a copy of the license, visit LICENSE.md.

#pragma once

#include "common.h"
#include "halfedge.h"

/// \file affine_manifold.h
///
/// Representation of an affine manifold.

/// Local layout manifold chart in R2 of the one ring around a central vertex.
///
/// The one ring is represented as a sequential list of the n faces and n + 1
/// vertices around the central vertex along with the corresponding vertex uv
/// positions. For i = 0,...,n-1, the ith vertex corresponds to to the first
/// vertex in face i ccw from the central vertex.
///
/// For an interior vertex, the one ring begins at an arbitrary face, and the
/// nth vertex is the same as the first. For a boundary vertex, the one ring
/// begins as the boundary face to the right of the vertex and ends at the left
/// boundary face, and the nth vertex is generally different from the 0th.
struct VertexManifoldChart {
  Halfedge::Index vertex_index; // Index of the vertex in the affine manifold
  std::vector<Halfedge::Index>
      vertex_one_ring; // List of manifold vertex indices in the one ring
  std::vector<Halfedge::Index>
      face_one_ring; // List of manifold face indices in the one ring
  MatrixXr
      one_ring_uv_positions; // Local uv coordinates of the one ring vertices
  bool is_boundary = false;  // Mark boundary vertices
  bool is_cone = false;      // Mark cone vertices
  bool is_cone_adjacent = false; // Mark vertices adjacent to a cone
};

/// Local layout manifold chart in R2 of the triangles around an edge.
///
/// An orientation is specified on the chart by a choice of top and bottom face.
///
/// For an interior edge, there are always two adjacent faces. For a boundary
/// edge, there is only one adjacent face. The bottom vertex and face indices
/// are set to an out of range value (e.g., -1), and the bottom vertex is set to
/// the empty vector.
struct EdgeManifoldChart {
  // Face indices
  Halfedge::Index top_face_index;
  Halfedge::Index bottom_face_index;

  // Vertex indices
  Halfedge::Index left_vertex_index;
  Halfedge::Index right_vertex_index;
  Halfedge::Index top_vertex_index;
  Halfedge::Index bottom_vertex_index;

  // Vertex positions
  PlanarPoint left_vertex_uv_position;
  PlanarPoint right_vertex_uv_position;
  PlanarPoint top_vertex_uv_position;
  PlanarPoint bottom_vertex_uv_position;

  // True iff the edge is on the boundary
  bool is_boundary;

  // added for C1 constraints
  // global uv vertex mapping
  int64_t left_global_uv_idx;
  int64_t right_global_uv_idx;
  int64_t top_global_uv_idx;
  int64_t bottom_global_uv_idx;
  int64_t reverse_left_global_uv_idx;
  int64_t reverse_right_global_uv_idx;

  // global uv vertex positions
  PlanarPoint left_global_uv_position;
  PlanarPoint right_global_uv_position;
  PlanarPoint top_global_uv_position;
  PlanarPoint bottom_global_uv_position;
  PlanarPoint reverse_left_global_uv_position;
  PlanarPoint reverse_right_global_uv_position;

  std::array<int64_t, 4> lagrange_nodes;
};

/// Local layout manifold chart in R2 of a triangle.
///
/// This is the same as global uv positions when these are provided.
struct FaceManifoldChart {
  // Face indices
  Halfedge::Index face_index;

  // Vertex positions
  std::array<PlanarPoint, 3> face_uv_positions;

  // Global information
  bool is_boundary = false;      // True iff the edge is on the boundary
  bool is_cone_adjacent = false; // Mark faces adjacent to a cone
  std::array<bool, 3> is_cone_corner = {
      false, false, false}; // Mark individual corners adjacent to a cone

  // added for C1 constraints
  // std::array<int64_t, 3> vertex_indices;
  // std::array<int64_t, 3> global_uv_indices;
  // std::array<PlanarPoint, 3> global_uv_positions;

  std::array<int64_t, 19> lagrange_nodes;
};

/// Representation for an affine manifold, which is a topological manifold F
/// equipped with a discrete metric l that satisfies the triangle inequality.
class AffineManifold {
public:
  typedef int Index;

  /// Default constructor for a trivial manifold
  AffineManifold();

  /// Constructor for a cone manifold from a global parametrization.
  ///
  /// @param[in] F: faces of the cone manifold
  /// @param[in] global_uv: global layout of the manifold
  /// @param[in] F_uv: faces of the global layout
  AffineManifold(const Eigen::MatrixXi &F, const MatrixXr &global_uv,
                 const Eigen::MatrixXi &F_uv);

  /// Get the number of faces in the manifold
  ///
  /// @return number of faces in the manifold
  Index num_faces() const;

  /// Get the number of vertices in the manifold
  ///
  /// @return number of vertices in the manifold
  Index num_vertices() const;

  /// Get faces for the manifold
  ///
  /// @return faces of the manifold
  Eigen::MatrixXi const &get_faces() const;

  /// Get halfedge for the manifold
  ///
  /// @return halfedge of the manifold
  Halfedge const &get_halfedge() const { return m_halfedge; }

  /// Get halfedge to corner map for the manifold
  ///
  /// @return halfedge to corner map of the manifold
  std::vector<std::pair<Eigen::Index, Eigen::Index>> const &
  get_he_to_corner() const {
    return m_he_to_corner;
  }

  /// Get faces for the manifold parametrization
  ///
  /// @return faces of the manifold layout
  Eigen::MatrixXi const &get_F_uv() const;

  /// Get an isometric chart for the vertex with the given index.
  ///
  /// Note that this will be a homeomorphism about the vertex if and only if the
  /// metric is flat there. However, any such chart may be made into a
  /// (nonisometric) homeomorphism about the vertex by composition with a
  /// suitable angle normalization map (r, theta) -> (r, 2 * pi * theta / (2 *
  /// pi - K)), where K is the Gaussian curvature at the vertex.
  ///
  /// @param[in] vertex_index: index of the vertex for the chart
  /// @return chart for the given vertex
  VertexManifoldChart const &get_vertex_chart(Index vertex_index) const;

  /// Get an isometric chart for the edge opposite the corner with the given
  /// face index and vertex index within the face.
  ///
  /// @param[in] face_index: index of a face containing the target edge
  /// @param[in] face_vertex_index: index of the corner opposite the edge in the
  /// face
  /// @return chart for the given edge
  EdgeManifoldChart const &get_edge_chart(Index face_index,
                                          Index face_vertex_index) const;

  /// Get an isometric chart for the given face.
  ///
  /// @param[in] face_index: index of a face
  /// @return chart for the given face
  FaceManifoldChart const &get_face_chart(Index face_index) const;

  /// Get the portions of the isometric vertex charts corresponding to the
  /// corners of a given face.
  ///
  /// In particular, for a face ijk, the local chart layouts are given for:
  ///   [0] vertices j and k in the vertex chart for vertex i
  ///   [1] vertices k and i in the vertex chart for vertex j
  ///   [2] vertices i and j in the vertex chart for vertex k
  ///
  /// @param[in] face_index: index of the face for the chart segments
  /// @param[out] corner_uv_positions: chart uv positions as enumerated above
  void
  get_face_corner_charts(Index face_index,
                         std::array<Matrix2x2r, 3> &corner_uv_positions) const;

  /// Get the portion of the edge charts contained in the interior of the given
  /// face.
  ///
  /// In particular, for a face ijk, the local chart layouts are given for:
  ///   [0] vertices j, k, i in the vertex chart for edge ij
  ///   [1] vertices k, i, j in the vertex chart for edge jk
  ///   [2] vertices i, j, k in the vertex chart for edge ki
  ///
  /// @param[in] face_index: index of the face for the charts
  /// @param[out] face_edge_uv_positions: uv positions contained in the given
  /// face
  void
  get_face_edge_charts(Index face_index,
                       std::array<Matrix3x2r, 3> &face_edge_uv_positions) const;

  /// @brief Get the uv coordinates of the face.
  ///
  /// @param[in] face_index: index of the face for the chart
  /// @param[out] face_edge_uv_positions: global uv positions of the face
  void get_face_global_uv(Index face_index,
                          std::array<PlanarPoint, 3> &face_uv_positions) const;

  /// Compute the curvature curvature at the given vertex.
  ///
  /// Gaussian curvature is used for interior vertices and geodesic curvature
  /// for boundary vertices.
  ///
  /// @param[in] vertex_index: index of the vertex
  /// @return curvature at the given vertex
  double compute_curvature(Index vertex_index) const;

  /// Determine if the vertex is on the boundary
  ///
  /// @param[in] vertex_index: index of the vertex
  /// @return true iff the vertex is on the boundary
  bool is_boundary(Index vertex_index) const;

  /// Determine if the manifold is flat at the given vertex, i.e. has zero
  /// Gaussian curvature or is a boundary vertex.
  ///
  /// @param[in] vertex_index: index of the vertex
  /// @return true iff the manifold is flat at the vertex
  bool is_flat(Index vertex_index) const;

  /// Get list of all flat vertices in the manifold
  ///
  /// @param[out] flat_vertices: list of flat vertices
  void compute_flat_vertices(std::vector<Index> &flat_vertices);

  /// Get list of all cones in the manifold
  ///
  /// @param[out] cones: list of cone vertices
  void compute_cones(std::vector<Index> &cones) const;

  /// Get boolean mask of all cones corners in the manifold
  ///
  /// @param[out] is_cone_corner: true iff corner i, j is a cone
  void
  compute_cones_corners(std::vector<std::array<bool, 3>> &is_cone_corner) const;

  /// Compute a matrix of cone point positions from mesh vertex positions.
  ///
  /// @param[in] V: mesh vertex positions
  /// @param[out] cone_points: cone positions w.r.t. V
  void compute_cone_points(const MatrixXr &V, MatrixXr &cone_points) const;

  /// Return list of all cones in the manifold
  ///
  /// @return list of cone vertices
  std::vector<Index> generate_cones() const;

  /// Get list of all boundary vertices in the manifold
  ///
  /// @param[out] boundary_vertices: list of boundary vertices
  void compute_boundary_vertices(std::vector<Index> &boundary_vertices) const;

  /// Return list of all boundary vertices in the manifold
  ///
  /// @return list of boundary vertices
  std::vector<Index> generate_boundary_vertices() const;

  /// @brief Mark a vertex as adjacent to a cone
  ///
  /// @param[in] vertex_index: vertex to mark
  void mark_cone_adjacent_vertex(Index vertex_index);

  /// @brief Mark a face as adjacent to a cone
  ///
  /// @param[in] face_index: face to mark
  void mark_cone_adjacent_face(Index face_index);

  /// Get global uv coordinates
  ///
  /// @return global uv coordinates, or the empty matrix if they do not exist
  MatrixXr const &get_global_uv() const;

  /// Cut edges adjacent to cones so that a planar layout is possible around
  /// them.
  void cut_cone_edges();

  /// Add the cone manifold and its data to the polyscope viewer with name
  /// 'cone_manifold'
  ///
  /// @param[in] V: mesh vertex positions
  /// @param[in] color: color for the affine manifold in the viewer
  void add_to_viewer(const MatrixXr &V,
                     Eigen::Matrix<double, 3, 1> color = GOLD_YELLOW) const;

  /// View the cone manifold and its data
  ///
  /// @param[in] V: mesh vertex positions
  void view(const MatrixXr &V) const;

  /// Save an image of the cone manifold and its data to file.
  ///
  /// @param[in] filename: file to save the screenshot to
  /// @param[in] V: mesh vertex positions
  /// @param[in] camera_position: camera position for the screenshot
  /// @param[in] camera_target: camera target for the screenshot
  /// @param[in] use_orthographic: use orthographic perspective if true
  void screenshot(const std::string &filename, const MatrixXr &V,
                  SpatialVector camera_position, SpatialVector camera_target,
                  bool use_orthographic) const;
  // Clear all internal data for a trivial cone manifold
  void clear();

public:
  void build_vertex_charts_from_lengths(
      const Eigen::MatrixXi &F, const std::vector<std::vector<double>> &l,
      std::vector<VertexManifoldChart> &vertex_charts) const;

  void build_edge_charts_from_lengths(
      const Eigen::MatrixXi &F, const Halfedge &halfedge,
      const std::vector<std::vector<double>> &l,
      std::vector<EdgeManifoldChart> &edge_charts) const;

  void build_face_charts(const Eigen::MatrixXi &F, const MatrixXr &global_uv,
                         const Eigen::MatrixXi &F_uv,
                         std::vector<FaceManifoldChart> &face_charts) const;

  void build_corner_to_edge_map(
      const std::vector<std::vector<Halfedge::Index>> &corner_to_he,
      const std::vector<Halfedge::Index> &he_to_edge,
      std::vector<std::vector<Halfedge::Index>> &corner_to_edge) const;

  PlanarPoint layout_next_vertex(const PlanarPoint &current_point,
                                 double next_edge_length,
                                 double prev_edge_length) const;

  void layout_one_ring(const Eigen::MatrixXi &F,
                       const std::vector<std::vector<double>> &l,
                       Index vertex_index,
                       const std::vector<Index> &vertex_one_ring,
                       const std::vector<Index> &face_one_ring,
                       MatrixXr &one_ring_uv_positions) const;

  void build_lengths_from_global_uv(const Eigen::MatrixXi &F,
                                    const MatrixXr &global_uv,
                                    std::vector<std::vector<double>> &l) const;

  void align_local_charts(const MatrixXr &uv, const Eigen::MatrixXi &F_uv);

  void mark_cones();

  double compute_corner_uv_length(Index face_index,
                                  Index face_vertex_index) const;

  bool is_valid_affine_manifold() const;

public:
  // Topology information
  // TODO: The faces are duplicated in the halfedge. Our halfedge alway
  // retains the original VF topology, so there is no need to maintain both
  // separately
  Eigen::MatrixXi m_F;
  std::vector<std::vector<Halfedge::Index>> m_corner_to_he;
  std::vector<std::vector<Halfedge::Index>> m_corner_to_edge;
  std::vector<std::pair<Eigen::Index, Eigen::Index>> m_he_to_corner;
  Halfedge m_halfedge;

  // Global metric information
  std::vector<std::vector<double>> m_l;
  MatrixXr m_global_uv;
  Eigen::MatrixXi m_F_uv;

  // Local metric information
  std::vector<VertexManifoldChart> m_vertex_charts;
  std::vector<EdgeManifoldChart> m_edge_charts;
  std::vector<FaceManifoldChart> m_face_charts;

public:
  // code added for C1 constraints

  // edge vectors
  void get_u_ij(Eigen::SparseMatrix<double> &u_ij_u,
                Eigen::SparseMatrix<double> &u_ij_v) const;

  // reindexing matrices
  void get_P_G2F(Eigen::SparseMatrix<int64_t> &P_G2F) const;
  void get_P_G2E(Eigen::SparseMatrix<int64_t> &P_G2E) const;
  void get_P_d_E(Eigen::SparseMatrix<int64_t> &P_d_E) const;
  void get_P_d_M(Eigen::SparseMatrix<int64_t> &P_d_M) const;

  void compute_face_edge_lagrange_node_indices();
  void compute_edge_global_uv_mappings();
  void compute_face_global_uv_mappings();
  void generate_lagrange_nodes();

  // per face control points indices
  // b0 b1 b2 b01 b10 b12 b21 b20 b02 b01^c b12^c b20^c b0c bc0 b1c bc1 b2c
  // bc2 bc
  std::vector<std::pair<int64_t, PlanarPoint>>
      m_lagrange_nodes; // (face_idx, barycentric coord)
  std::map<int64_t, int64_t> v_to_lagrange_node_map; // vidx to node idx

  // edge with v_idx to two nodes idx on edge
  std::map<std::pair<int64_t, int64_t>, std::array<int64_t, 4>>
      m_boundary_edge_to_node_map;
};

/// Representation for an affine manifold with a global parametrization, which
/// yields a flat metric and thus an affine manifold structure.
class ParametricAffineManifold : public AffineManifold {
public:
  ParametricAffineManifold();

  /// Constructor for a parametric affine manifold from a global
  /// parametrization.
  ///
  /// @param[in] F: faces of the affine manifold
  /// @param[in] global_uv: affine global layout of the manifold
  ParametricAffineManifold(const Eigen::MatrixXi &F, const MatrixXr &global_uv);

  /// Get global uv coordinates for a given vertex.
  ///
  /// @param[in] vertex_index: index of the vertex for the uv position
  /// @param[out] uv_coords: global uv position for the given vertex
  void get_vertex_global_uv(Index vertex_index, PlanarPoint &uv_coords) const;

public:
  bool is_valid_parametric_affine_manifold() const;
};

/// Generate an affine manifold with the cone faces removed but cone adjaceny
/// information retained
void remove_cones(const Eigen::MatrixXd &V,
                  const AffineManifold &affine_manifold,
                  Eigen::MatrixXd &pruned_V,
                  AffineManifold &pruned_affine_manifold,
                  std::vector<AffineManifold::Index> &cones,
                  std::vector<AffineManifold::Index> &removed_faces);