// Copyright 2023 Adobe Research. All rights reserved.
// To view a copy of the license, visit LICENSE.md.

#include "affine_manifold.h"

#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include "vertex_circulator.h"

#include <map>

// *************
// Cone Manifold
// *************

// **************
// Public Methods
// **************

AffineManifold::AffineManifold() { clear(); }

AffineManifold::AffineManifold(const Eigen::MatrixXi &F,
                               const MatrixXr &global_uv,
                               const Eigen::MatrixXi &F_uv)
    : m_F(F), m_global_uv(global_uv), m_F_uv(F_uv) {
  // Check the input
#if CHECK_VALIDITY
  if (!is_manifold(F)) {
    spdlog::error("Input mesh is not manifold");
    clear();
    return;
  }
  if (!is_manifold(F_uv)) {
    spdlog::error("Input mesh is not manifold");
    clear();
    return;
  }
  if (F_uv.rows() != F.rows()) {
    spdlog::error("Input mesh and uv mesh have different sizes");
    clear();
    return;
  }
#endif

  // Build halfedge
  m_halfedge = Halfedge(F, m_corner_to_he, m_he_to_corner);
  std::vector<Halfedge::Index> he_to_edge =
      m_halfedge.get_halfedge_to_edge_map();
  build_corner_to_edge_map(m_corner_to_he, he_to_edge, m_corner_to_edge);

  // Build edge lengths and charts from the global uv
  build_lengths_from_global_uv(F_uv, global_uv, m_l);
  build_vertex_charts_from_lengths(F, m_l, m_vertex_charts);
  build_edge_charts_from_lengths(F, m_halfedge, m_l, m_edge_charts);
  build_face_charts(F, global_uv, F_uv, m_face_charts);

  // Align charts with the input parametrization
  align_local_charts(global_uv, F_uv);

  // Mark vertices, edges, and faces adjacent to cones
  mark_cones();

  // Check validity
  if (!is_valid_affine_manifold()) {
    spdlog::error("Could not build a cone manifold");
    clear();
  }
}

AffineManifold::Index AffineManifold::num_faces() const { return m_F.rows(); }

AffineManifold::Index AffineManifold::num_vertices() const {
  return m_vertex_charts.size();
}

Eigen::MatrixXi const &AffineManifold::get_faces() const { return m_F; }

Eigen::MatrixXi const &AffineManifold::get_F_uv() const { return m_F_uv; }

VertexManifoldChart const &
AffineManifold::get_vertex_chart(AffineManifold::Index vertex_index) const {
  return m_vertex_charts[vertex_index];
}

EdgeManifoldChart const &
AffineManifold::get_edge_chart(Index face_index,
                               Index face_vertex_index) const {
  Index edge_index = m_corner_to_edge[face_index][face_vertex_index];
  return m_edge_charts[edge_index];
}

FaceManifoldChart const &
AffineManifold::get_face_chart(Index face_index) const {
  return m_face_charts[face_index];
}

void AffineManifold::get_face_corner_charts(
    AffineManifold::Index face_index,
    std::array<Matrix2x2r, 3> &corner_uv_positions) const {
  for (Eigen::Index i = 0; i < 3; ++i) {
    // Get the chart for vertex i in the given face
    int vertex_index = m_F(face_index, i);
    VertexManifoldChart const &chart = m_vertex_charts[vertex_index];

    // Iterate over the one ring of face vertex i
    for (size_t j = 0; j < chart.face_one_ring.size(); ++j) {
      // Skip faces until the input face is found
      if (chart.face_one_ring[j] != face_index)
        continue;

      // Get the uv coordinates of the other two face vertices in the chart of
      // face vertex i
      int first_edge = j;
      int second_edge = j + 1;
      corner_uv_positions[i].row(0) =
          chart.one_ring_uv_positions.row(first_edge);
      corner_uv_positions[i].row(1) =
          chart.one_ring_uv_positions.row(second_edge);
      break;
    }
  }
}

void AffineManifold::get_face_edge_charts(
    AffineManifold::Index face_index,
    std::array<Matrix3x2r, 3> &face_edge_uv_positions) const {
  // Iterate over edges
  for (size_t i = 0; i < 3; ++i) {
    // Get the chart for edge jk opposite vertex i in the given face
    EdgeManifoldChart const &chart = get_edge_chart(face_index, i);

    // Break into cases depending on if the face is the top or bottom face in
    // the chart
    if (chart.top_face_index == face_index) {
      face_edge_uv_positions[i].row(0) = chart.right_vertex_uv_position;
      face_edge_uv_positions[i].row(1) = chart.top_vertex_uv_position;
      face_edge_uv_positions[i].row(2) = chart.left_vertex_uv_position;
    } else if (chart.bottom_face_index == face_index) {
      face_edge_uv_positions[i].row(0) = chart.left_vertex_uv_position;
      face_edge_uv_positions[i].row(1) = chart.bottom_vertex_uv_position;
      face_edge_uv_positions[i].row(2) = chart.right_vertex_uv_position;
    } else {
      spdlog::error("Face {} not found in the given edge chart", face_index);
    }
  }
}

void AffineManifold::get_face_global_uv(
    AffineManifold::Index face_index,
    std::array<PlanarPoint, 3> &face_uv_positions) const {
  face_uv_positions = get_face_chart(face_index).face_uv_positions;
}

double
AffineManifold::compute_curvature(AffineManifold::Index vertex_index) const {
  VertexManifoldChart const &chart = m_vertex_charts[vertex_index];

  // Get zero uv coordinate (location of the central vertex)
  PlanarPoint zero;
  zero.setZero(2);

  // Compute cone angle
  double cone_angle = 0.0;
  for (size_t j = 0; j < chart.face_one_ring.size(); ++j) {
    cone_angle +=
        angle_from_positions<2>(zero, chart.one_ring_uv_positions.row(j),
                                chart.one_ring_uv_positions.row(j + 1));
  }

  // Compute geodesic curvature for boundary vertices and Gaussian curvature for
  // interior vertices
  if (chart.is_boundary) {
    return (M_PI)-cone_angle;
  } else {
    return (2 * M_PI) - cone_angle;
  }
}

bool AffineManifold::is_boundary(AffineManifold::Index vertex_index) const {
  return get_vertex_chart(vertex_index).is_boundary;
}

bool AffineManifold::is_flat(AffineManifold::Index vertex_index) const {
  // All vertices with zero curvature are flat
  if (float_equal_zero(compute_curvature(vertex_index), 1e-5))
    return true;

  // All vertices on the boundary are flat
  if (is_boundary(vertex_index))
    return true;

  return false;
}

void AffineManifold::compute_flat_vertices(
    std::vector<AffineManifold::Index> &flat_vertices) {
  flat_vertices.clear();
  flat_vertices.reserve(num_vertices());

  for (Index vertex_index = 0; vertex_index < num_vertices(); ++vertex_index) {
    if (is_flat(vertex_index)) {
      flat_vertices.push_back(vertex_index);
    }
  }
}

void AffineManifold::compute_cones(
    std::vector<AffineManifold::Index> &cones) const {
  cones.clear();
  cones.reserve(num_vertices());

  for (Index vertex_index = 0; vertex_index < num_vertices(); ++vertex_index) {
    if (!is_flat(vertex_index)) {
      spdlog::debug("Getting cone {} of curvature {}", vertex_index,
                    compute_curvature(vertex_index));
      cones.push_back(vertex_index);
    }
  }
}

void AffineManifold::compute_cones_corners(
    std::vector<std::array<bool, 3>> &is_cone_corner) const {
  is_cone_corner.resize(num_faces());
  for (Index fi = 0; fi < num_faces(); ++fi) {
    for (size_t k = 0; k < 3; ++k) {
      is_cone_corner[fi][k] = get_face_chart(fi).is_cone_corner[k];
    }
  }
}

void AffineManifold::compute_cone_points(const MatrixXr &V,
                                         MatrixXr &cone_points) const {
  // Compute the cone indices
  std::vector<Index> cones;
  compute_cones(cones);

  // Build the cone points from the vertex set
  Index num_cones = cones.size();
  cone_points.resize(num_cones, V.cols());
  for (Index i = 0; i < num_cones; ++i) {
    Index ci = cones[i];
    cone_points.row(i) = V.row(ci);
  }
}

std::vector<AffineManifold::Index> AffineManifold::generate_cones() const {
  std::vector<Index> cones;
  compute_cones(cones);
  return cones;
}

void AffineManifold::compute_boundary_vertices(
    std::vector<AffineManifold::Index> &boundary_vertices) const {
  boundary_vertices.clear();
  boundary_vertices.reserve(num_vertices());

  for (Index vertex_index = 0; vertex_index < num_vertices(); ++vertex_index) {
    if (is_boundary(vertex_index)) {
      boundary_vertices.push_back(vertex_index);
    }
  }
}

std::vector<AffineManifold::Index>
AffineManifold::generate_boundary_vertices() const {
  std::vector<Index> boundary_vertices;
  compute_boundary_vertices(boundary_vertices);
  return boundary_vertices;
}

void AffineManifold::mark_cone_adjacent_vertex(
    AffineManifold::Index vertex_index) {
  m_vertex_charts[vertex_index].is_cone_adjacent = true;
}

void AffineManifold::mark_cone_adjacent_face(AffineManifold::Index face_index) {
  m_face_charts[face_index].is_cone_adjacent = true;
}

void AffineManifold::cut_cone_edges() {
  Eigen::MatrixXi const &F = get_faces();

  // Get cone vertices
  std::vector<Index> cones;
  compute_cones(cones);

  // For each cone, choose an edge and make it a boundary edge
  for (size_t i = 0; i < cones.size(); ++i) {
    // Get cone vertex chart
    Index vi = cones[i];
    VertexManifoldChart const &vertex_chart = get_vertex_chart(vi);

    // Get edge chart adjacent to the cone edge
    Index face_index = vertex_chart.face_one_ring[0];
    Eigen::Index face_vertex_index =
        find_face_vertex_index(F.row(face_index), vi);
    EdgeManifoldChart const &edge_chart =
        get_edge_chart(face_index, face_vertex_index);

    // Mark edge and endpoints as boundaries
    Index edge_index = m_corner_to_edge[face_index][face_vertex_index];
    Index v0 = edge_chart.left_vertex_index;
    Index v1 = edge_chart.right_vertex_index;
    m_edge_charts[edge_index].is_boundary = true;
    m_vertex_charts[v0].is_boundary = true;
    m_vertex_charts[v1].is_boundary = true;
  }
}

MatrixXr const &AffineManifold::get_global_uv() const { return m_global_uv; }

void AffineManifold::add_to_viewer(const MatrixXr &V,
                                   Eigen::Matrix<double, 3, 1> color) const {
  polyscope::init();

  // Add manifold
  Eigen::MatrixXi const F = get_faces();
  polyscope::registerSurfaceMesh("cone_manifold", V, F);
  polyscope::getSurfaceMesh("cone_manifold")
      ->setEdgeWidth(1)
      ->setSurfaceColor(glm::vec3(color[0], color[1], color[2]));

  // Add cone points
  MatrixXr cone_points;
  compute_cone_points(V, cone_points);
  polyscope::registerPointCloud("cones", cone_points);
  polyscope::getPointCloud("cones")->setPointColor(glm::vec3(0.5, 0.0, 0.0));
}

void AffineManifold::view(const MatrixXr &V) const {
  add_to_viewer(V);
  polyscope::show();
}

void AffineManifold::screenshot(const std::string &filename, const MatrixXr &V,
                                SpatialVector camera_position,
                                SpatialVector camera_target,
                                bool use_orthographic) const {
  // Add the contour network to the surface
  add_to_viewer(V);

  // Build the cameras for the viewer
  glm::vec3 glm_camera_position = {camera_position[0], camera_position[1],
                                   camera_position[2]};
  glm::vec3 glm_camera_target = {camera_target[0], camera_target[1],
                                 camera_target[2]};

  // Set up the cameras
  polyscope::view::lookAt(glm_camera_position, glm_camera_target);
  if (use_orthographic) {
    polyscope::view::projectionMode = polyscope::ProjectionMode::Orthographic;
  }

  // Take the screenshot
  polyscope::screenshot(filename);
  SPDLOG_INFO("Screenshot saved to {}", filename);
  polyscope::removeAllStructures();
}

void AffineManifold::clear() {
  m_F.setZero(0, 0);
  m_corner_to_he.clear();
  m_corner_to_edge.clear();
  m_he_to_corner.clear();
  m_halfedge.clear();

  m_l.clear();
  m_global_uv.setZero(0, 0);
  m_F_uv.setZero(0, 0);

  m_vertex_charts.clear();
  m_edge_charts.clear();
  m_face_charts.clear();
}

// *****************
// Protected Methods
// *****************

// Build isometric charts for a surface with a flat metric
void AffineManifold::build_vertex_charts_from_lengths(
    const Eigen::MatrixXi &F, const std::vector<std::vector<double>> &l,
    std::vector<VertexManifoldChart> &vertex_charts) const {
  Index num_vertices = F.maxCoeff() + 1;

  // Build vertex circulator
  VertexCirculator vertex_circulator(F);

  // Iterate over vertices
  vertex_charts.resize(num_vertices);
  for (Index vertex_index = 0; vertex_index < num_vertices; ++vertex_index) {
    // Record the given vertex index in the chart
    vertex_charts[vertex_index].vertex_index = vertex_index;

    // Build one ring in the original surface for the vertex chart
    vertex_circulator.get_one_ring(vertex_index,
                                   vertex_charts[vertex_index].vertex_one_ring,
                                   vertex_charts[vertex_index].face_one_ring);

    // Layout the vertices according to the given flat metric
    layout_one_ring(F, l, vertex_index,
                    vertex_charts[vertex_index].vertex_one_ring,
                    vertex_charts[vertex_index].face_one_ring,
                    vertex_charts[vertex_index].one_ring_uv_positions);

    // A vertex is on the boundary iff the vertex one ring is not a closed loop
    Index v0 = vertex_charts[vertex_index].vertex_one_ring.front();
    Index vn = vertex_charts[vertex_index].vertex_one_ring.back();
    vertex_charts[vertex_index].is_boundary = (v0 != vn);

    // By default, assume not cone adjacent (Changed later)
    // FIXME This is dangerous
    vertex_charts[vertex_index].is_cone_adjacent = false;
  }
}

void AffineManifold::build_edge_charts_from_lengths(
    const Eigen::MatrixXi &F, const Halfedge &halfedge,
    const std::vector<std::vector<double>> &l,
    std::vector<EdgeManifoldChart> &edge_charts) const {
  // Build edge charts
  Index num_edges = halfedge.num_edges();
  edge_charts.resize(num_edges);
  for (Index edge_index = 0; edge_index < num_edges; ++edge_index) {
    // Get relevant halfedges for the face
    Halfedge::Index he_top = halfedge.edge_to_first_halfedge(edge_index);
    Halfedge::Index he_bottom = halfedge.edge_to_second_halfedge(edge_index);
    Halfedge::Index he_top_next = halfedge.next_halfedge(he_top);
    Halfedge::Index he_bottom_next = halfedge.next_halfedge(he_bottom);

    // Get indices for the faces and vertices around the edge
    EdgeManifoldChart chart;
    chart.top_face_index = halfedge.halfedge_to_face(he_top);
    chart.bottom_face_index = halfedge.halfedge_to_face(he_bottom);
    chart.left_vertex_index = halfedge.halfedge_to_tail_vertex(he_top);
    chart.right_vertex_index = halfedge.halfedge_to_head_vertex(he_top);
    chart.top_vertex_index = halfedge.halfedge_to_head_vertex(he_top_next);
    chart.bottom_vertex_index =
        halfedge.halfedge_to_head_vertex(he_bottom_next);
    chart.is_boundary = halfedge.is_boundary_edge(edge_index);

    // Get lengths of the edges of the top triangle
    int j_top = find_face_vertex_index(F.row(chart.top_face_index),
                                       chart.left_vertex_index);
    assert(j_top >= 0);
    double lij = l[chart.top_face_index][(j_top + 2) % 3];
    double ljk = l[chart.top_face_index][(j_top + 0) % 3];
    double lki = l[chart.top_face_index][(j_top + 1) % 3];

    // Layout vertices starting at vi. Note that here, unlike in the vertex
    // chart case, we start from axis aligned edge with length 1
    chart.left_vertex_uv_position = PlanarPoint(0, 0);
    chart.right_vertex_uv_position = PlanarPoint(1.0, 0);
    assert(lij > 0);
    chart.top_vertex_uv_position = layout_next_vertex(
        chart.right_vertex_uv_position, ljk / lij, lki / lij);

    // Get center of the target edge for a later shift
    PlanarPoint center = 0.5 * chart.right_vertex_uv_position;

    // If the edge is not on the boundary, build the bottom triangle
    if (!chart.is_boundary) {
      // Get lengths of the edges of the bottom triangle if it
      int j_bottom = find_face_vertex_index(F.row(chart.bottom_face_index),
                                            chart.left_vertex_index);
      assert(j_bottom >= 0);
      double lil = l[chart.bottom_face_index][(j_bottom + 2) % 3];
      double llj = l[chart.bottom_face_index][(j_bottom + 0) % 3];
      assert(float_equal(lij, l[chart.bottom_face_index][(j_bottom + 1) % 3]));

      // Construct the last vertex counterclockwise and then reflect it
      PlanarPoint uvl_reflected = layout_next_vertex(
          chart.right_vertex_uv_position, llj / lij, lil / lij);
      chart.bottom_vertex_uv_position = reflect_across_x_axis(uvl_reflected);

      // Shift all vertices so the midpoint is at the origin
      chart.left_vertex_uv_position -= center;
      chart.right_vertex_uv_position -= center;
      chart.top_vertex_uv_position -= center;
      chart.bottom_vertex_uv_position -= center;
    } else {
      // Shift all constructed vertices so the midpoint is at the origin
      chart.left_vertex_uv_position -= center;
      chart.right_vertex_uv_position -= center;
      chart.top_vertex_uv_position -= center;

      // Set the bottom uv position to the zero vector
      chart.bottom_vertex_uv_position.setZero();
    }

    // Set chart
    edge_charts[edge_index] = chart;
  }
}

void AffineManifold::build_face_charts(
    const Eigen::MatrixXi &F, const MatrixXr &global_uv,
    const Eigen::MatrixXi &F_uv,
    std::vector<FaceManifoldChart> &face_charts) const {
  Index num_faces = F.rows();
  face_charts.resize(num_faces);
  for (Index face_index = 0; face_index < num_faces; ++face_index) {
    face_charts[face_index].face_index = face_index;
    for (Eigen::Index face_vertex_index = 0; face_vertex_index < 3;
         ++face_vertex_index) {
      Index uv_vertex_index = F_uv(face_index, face_vertex_index);
      face_charts[face_index].face_uv_positions[face_vertex_index] =
          global_uv.row(uv_vertex_index);
    }
  }
}

// Compose corner to halfedge and halfedge to edge maps
void AffineManifold::build_corner_to_edge_map(
    const std::vector<std::vector<Halfedge::Index>> &corner_to_he,
    const std::vector<Halfedge::Index> &he_to_edge,
    std::vector<std::vector<Halfedge::Index>> &corner_to_edge) const {
  Index num_faces = corner_to_he.size();
  corner_to_edge.resize(num_faces);

  for (Index face_index = 0; face_index < num_faces; ++face_index) {
    corner_to_edge[face_index].resize(3);
    for (Index face_vertex_index = 0; face_vertex_index < 3;
         face_vertex_index++) {
      Index he_index = corner_to_he[face_index][face_vertex_index];
      corner_to_edge[face_index][face_vertex_index] = he_to_edge[he_index];
    }
  }
}

// Layout the next vertex in a triangle with given lengths and the current point
// position
PlanarPoint AffineManifold::layout_next_vertex(const PlanarPoint &current_point,
                                               double next_edge_length,
                                               double prev_edge_length) const {
  // Get the current point and its rotation
  PlanarPoint p0 = current_point;
  PlanarPoint p0_perp(-p0[1], p0[0]);

  // Get ratios of edge lengths
  double current_edge_length = current_point.norm();
  double l1 = next_edge_length / current_edge_length;
  double l2 = prev_edge_length / current_edge_length;

  // Compute parallel and perpendicular components of the next point
  // construction
  double a = 0.5 * (1 + l2 * l2 - l1 * l1);
  double b = 2 * area_from_length(1.0, l1, l2);

  // Build the next point
  PlanarPoint next_point = a * p0 + b * p0_perp;
  assert(!vector_contains_nan(next_point));
  assert(float_equal(next_point.norm(), prev_edge_length));
  return next_point;
}

// Layout the one ring around the given vertex from the lengths
void AffineManifold::layout_one_ring(const Eigen::MatrixXi &F,
                                     const std::vector<std::vector<double>> &l,
                                     int vertex_index,
                                     const std::vector<int> &vertex_one_ring,
                                     const std::vector<int> &face_one_ring,
                                     MatrixXr &one_ring_uv_positions) const {
  one_ring_uv_positions.resize(vertex_one_ring.size(), 2);
  if (spdlog::get_level() != spdlog::level::info) {
    SPDLOG_TRACE("Building layout for one ring: {}",
                 formatted_vector(vertex_one_ring));
  }

  // Initialize first vertex to position (l0, 0), where l0 is the length of the
  // edge
  int f0 = face_one_ring[0];
  int j0 = find_face_vertex_index(F.row(f0), vertex_index);
  double l0 = l[f0][(j0 + 2) % 3];
  one_ring_uv_positions.row(0) = PlanarPoint(l0, 0);

  // Layout remaining vertices
  for (size_t i = 0; i < face_one_ring.size(); ++i) {
    // Get current face and the index of the vertex in it
    int f = face_one_ring[i];
    int j = find_face_vertex_index(F.row(f), vertex_index);
    if (spdlog::get_level() != spdlog::level::info) {
      SPDLOG_TRACE("Laying out vertex for face {}", F.row(f));
      SPDLOG_TRACE("Face lengths are {}", formatted_vector(l[f]));
    }

    // Get the lengths of the triangle edges
    double next_edge_length = l[f][j];
    double prev_edge_length = l[f][(j + 1) % 3];
    assert(float_equal(l[f][(j + 2) % 3], one_ring_uv_positions.row(i).norm()));

    // Layout the next vertex
    one_ring_uv_positions.row(i + 1) = layout_next_vertex(
        one_ring_uv_positions.row(i), next_edge_length, prev_edge_length);
    if (spdlog::get_level() != spdlog::level::info) {
      SPDLOG_TRACE("Next vertex is {}", one_ring_uv_positions.row(i + 1));
    }
    assert(float_equal(next_edge_length, (one_ring_uv_positions.row(i + 1) -
                                          one_ring_uv_positions.row(i))
                                             .norm()));
    assert(
        float_equal(prev_edge_length, one_ring_uv_positions.row(i + 1).norm()));
  }
  spdlog::trace("Final layout:\n{}", one_ring_uv_positions);

  assert(!matrix_contains_nan(one_ring_uv_positions));
}

// Build a corner-indexed metric for a surface with a global parametrization
void AffineManifold::build_lengths_from_global_uv(
    const Eigen::MatrixXi &F, const MatrixXr &global_uv,
    std::vector<std::vector<double>> &l) const {
  Index num_faces = F.rows();
  Index face_size = F.cols();
  assert(face_size == 3);
  l.resize(num_faces);

  // Iterate over faces
  for (Index i = 0; i < num_faces; ++i) {
    l[i].resize(face_size);

    // Iterate over vertices in face i
    for (Index j = 0; j < face_size; ++j) {
      // Get the length of the edge opposite face vertex j
      PlanarPoint prev_uv = global_uv.row(F(i, (j + 2) % face_size));
      PlanarPoint next_uv = global_uv.row(F(i, (j + 1) % face_size));
      PlanarPoint edge_vector = prev_uv - next_uv;
      l[i][j] = edge_vector.norm();
    }
  }
}

// Align local uv charts with the global parametrization
void AffineManifold::align_local_charts(const MatrixXr &uv,
                                        const Eigen::MatrixXi &F_uv) {
  // Rotate and scale local layouts to align with the global layout
  for (Index vertex_index = 0; vertex_index < num_vertices(); ++vertex_index) {
    // Get the (transposed) similarity map that maps [1, 0]^T to the first local
    // uv edge
    MatrixXr local_layout =
        get_vertex_chart(vertex_index).one_ring_uv_positions;
    PlanarPoint local_edge = local_layout.row(0);
    MatrixXr local_similarity_map(2, 2);
    local_similarity_map << local_edge[0], local_edge[1], -local_edge[1],
        local_edge[0];

    // Get the global uv values corresponding the edge of the face
    Index edge_face_index = get_vertex_chart(vertex_index).face_one_ring[0];
    Index edge_face_vertex_index =
        find_face_vertex_index(m_F.row(edge_face_index), vertex_index);
    Index uv_vertex_index = F_uv(edge_face_index, edge_face_vertex_index);
    Index uv_edge_vertex_index =
        F_uv(edge_face_index, (edge_face_vertex_index + 1) % 3);

    // Get (transposed) similarity map that maps [1, 0]^T to the first global uv
    // edge
    PlanarPoint global_edge =
        uv.row(uv_edge_vertex_index) - uv.row(uv_vertex_index);
    MatrixXr global_similarity_map(2, 2);
    global_similarity_map << global_edge[0], global_edge[1], -global_edge[1],
        global_edge[0];

    // Apply composite similarity maps to the local uv positions
    MatrixXr similarity_map =
        global_similarity_map * local_similarity_map.inverse();
    m_vertex_charts[vertex_index].one_ring_uv_positions =
        m_vertex_charts[vertex_index].one_ring_uv_positions * similarity_map;
  }

  // Check validity after direct member variable manipulation
#if CHECK_VALIDITY
  assert(is_valid_affine_manifold());
#endif
}

// Mark cones and surrounding elements in the vertex and face charts
void AffineManifold::mark_cones() {
  Eigen::MatrixXi const &F = get_faces();
  std::vector<Index> cones;
  compute_cones(cones);

  for (size_t i = 0; i < cones.size(); ++i) {
    Index ci = cones[i];
    m_vertex_charts[ci].is_cone = true;
    VertexManifoldChart const &chart = get_vertex_chart(ci);
    spdlog::debug("Marking cone at {}", ci);

    // Mark vertices adjacent to cones
    SPDLOG_DEBUG("Marking cone adjacent vertices at {}",
                 formatted_vector(chart.vertex_one_ring, ", "));
    for (size_t j = 0; j < chart.vertex_one_ring.size(); ++j) {
      Index vj = chart.vertex_one_ring[j];
      m_vertex_charts[vj].is_cone_adjacent = true;
    }

    // Mark faces adjacent to cones
    SPDLOG_DEBUG("Marking cone adjacent faces at {}",
                 formatted_vector(chart.face_one_ring, ", "));
    for (size_t j = 0; j < chart.face_one_ring.size(); ++j) {
      Index fj = chart.face_one_ring[j];
      m_face_charts[fj].is_cone_adjacent = true;

      // Mark individual corners
      for (Index k = 0; k < 3; ++k) {
        Index vk = F(fj, k);
        if (m_vertex_charts[vk].is_cone) {
          m_face_charts[fj].is_cone_corner[k] = true;
        }
      }
    }
  }
}

double AffineManifold::compute_corner_uv_length(
    AffineManifold::Index face_index,
    AffineManifold::Index face_vertex_index) const {
  Index vn = m_F_uv(face_index, (face_vertex_index + 1) % 3);
  Index vp = m_F_uv(face_index, (face_vertex_index + 2) % 3);
  PlanarPoint next_uv = m_global_uv.row(vn);
  PlanarPoint prev_uv = m_global_uv.row(vp);
  PlanarPoint edge_vector = next_uv - prev_uv;
  return edge_vector.norm();
}

// Check that the manifold is valid and self-consistent
bool AffineManifold::is_valid_affine_manifold() const {
  // Threshold for length comparisons
  double length_threshold = 1e-6;

  // Zero uv coordinate
  PlanarPoint zero;
  zero.setZero(2);

  // Face containment helper lambda
  auto contains_vertex = [](const Eigen::VectorXi &face, Index index) {
    for (Eigen::Index i = 0; i < face.size(); ++i) {
      if (face[i] == index)
        return true;
    }

    return false;
  };

  // Edge length check helper lambda
  auto edge_has_length = [&](const PlanarPoint &v0, const PlanarPoint &v1,
                             double length) {
    PlanarPoint edge = v1 - v0;
    return float_equal(edge.norm(), length, length_threshold);
  };

  // Check that the sizes of the member variables are consistent
  if (static_cast<Index>(m_F.rows()) != static_cast<Index>(m_l.size()))
    return false;

  // Check that the global metric is consistent
  for (Index fi = 0; fi < num_faces(); ++fi) {
    for (Index j = 0; j < 3; ++j) {

      // Check the length of the edge is the same as the uv length
      double edge_length = m_l[fi][j];
      double edge_uv_length = compute_corner_uv_length(fi, j);
      if (!float_equal(edge_length, edge_uv_length, length_threshold)) {
        spdlog::error(
            "Inconsistent edge length {} and uv length {} for corner {}, {}",
            edge_length, edge_uv_length, fi, j);
        return false;
      }

      // Get opposite halfedge and corner if it exists
      Halfedge::Index he = m_corner_to_he[fi][j];
      if (m_halfedge.is_boundary_halfedge(he))
        continue;
      Halfedge::Index he_opp = m_halfedge.opposite_halfedge(he);
      Index fi_opp = m_he_to_corner[he_opp].first;
      Index j_opp = m_he_to_corner[he_opp].second;

      // Check uvs are the same for the opposite corners
      double opposite_edge_uv_length = compute_corner_uv_length(fi_opp, j_opp);
      if (!float_equal(edge_length, opposite_edge_uv_length,
                       length_threshold)) {
        spdlog::error(
            "Inconsistent opposite uv length for corners {}, {} and {}, {}", fi,
            j, fi_opp, j_opp);
        return false;
      }
    }
  }

  // Check that each vertex chart is valid
  for (Index vertex_index = 0; vertex_index < num_vertices(); ++vertex_index) {
    auto const &chart = m_vertex_charts[vertex_index];

    // Check basic chart indexing and size validity
    if (chart.vertex_index != vertex_index)
      return false;
    if (chart.vertex_one_ring.size() != (chart.face_one_ring.size() + 1))
      return false;
    if (static_cast<Index>(chart.one_ring_uv_positions.rows()) !=
        static_cast<Index>(chart.vertex_one_ring.size()))
      return false;

    // Check that each one ring face contains the cental vertex, the vertex with
    // the same index in the vertex one ring, and the vertex with one larger
    // index
    for (size_t i = 0; i < chart.face_one_ring.size(); ++i) {
      Index face_index = chart.face_one_ring[i];
      Index face_vertex_index =
          find_face_vertex_index(m_F.row(face_index), vertex_index);
      Index vi = chart.vertex_one_ring[i];
      Index vj = chart.vertex_one_ring[i + 1];

      // Check that the one ring indexing is valid
      if (face_index > m_F.rows())
        return false;
      if (!contains_vertex(m_F.row(face_index), vertex_index))
        return false;
      if (!contains_vertex(m_F.row(face_index), vi))
        return false;
      if (!contains_vertex(m_F.row(face_index), vj))
        return false;

      // Check that each local uv length is compatible with the given metric
      if (spdlog::get_level() != spdlog::level::info) {
        spdlog::trace("chart uv positions: {}", chart.one_ring_uv_positions);
        SPDLOG_TRACE("Face lengths: {}", formatted_vector(m_l[face_index]));
      }
      if (!edge_has_length(zero, chart.one_ring_uv_positions.row(i),
                           m_l[face_index][(face_vertex_index + 2) % 3])) {
        spdlog::error(
            "uv position {} in chart {} does not have expected norm {}",
            chart.one_ring_uv_positions.row(i), vertex_index,
            m_l[face_index][(face_vertex_index + 2) % 3]);
        return false;
      }
      if (!edge_has_length(chart.one_ring_uv_positions.row(i + 1),
                           chart.one_ring_uv_positions.row(i),
                           m_l[face_index][(face_vertex_index + 0) % 3])) {
        spdlog::error(
            "uv positions {} and {} in chart {} do not have expected length {}",
            chart.one_ring_uv_positions.row(i + 1),
            chart.one_ring_uv_positions.row(i), vertex_index,
            m_l[face_index][(face_vertex_index + 0) % 3]);
        return false;
      }
      if (!edge_has_length(zero, chart.one_ring_uv_positions.row(i + 1),
                           m_l[face_index][(face_vertex_index + 1) % 3])) {
        spdlog::error(
            "uv position {} in chart {} does not have expected norm {}",
            chart.one_ring_uv_positions.row(i + 1), vertex_index,
            m_l[face_index][(face_vertex_index + 1) % 3]);
        return false;
      }
    }
  }

  // Return true if no issues found
  return true;
}

// **************************
// Parametric Affine Manifold
// **************************

// **************
// Public Methods
// **************

ParametricAffineManifold::ParametricAffineManifold() {
  assert(is_valid_parametric_affine_manifold());
}

ParametricAffineManifold::ParametricAffineManifold(const Eigen::MatrixXi &F,
                                                   const MatrixXr &global_uv)
    : AffineManifold(F, global_uv, F) {
  assert(is_valid_parametric_affine_manifold());
}

void ParametricAffineManifold::get_vertex_global_uv(
    AffineManifold::Index vertex_index, PlanarPoint &uv_coords) const {
  uv_coords = m_global_uv.row(vertex_index);
}

// ***************
// Private Methods
// ***************

bool ParametricAffineManifold::is_valid_parametric_affine_manifold() const {
  if (m_F_uv != m_F)
    return false;

  // Check layout around each vertex is compatible with the global uv
  for (Index vertex_index = 0; vertex_index < num_vertices(); ++vertex_index) {
    auto const &chart = m_vertex_charts[vertex_index];
    for (size_t i = 0; i < chart.vertex_one_ring.size(); ++i) {
      Index vi = chart.vertex_one_ring[i];
      PlanarPoint local_uv_difference = chart.one_ring_uv_positions.row(i);
      PlanarPoint global_uv_difference =
          m_global_uv.row(vi) - m_global_uv.row(vertex_index);
      if (!vector_equal(global_uv_difference, local_uv_difference)) {
        spdlog::error("Global uv coordinates {} and {} do not have expected "
                      "difference {}",
                      m_global_uv.row(vi), m_global_uv.row(vertex_index),
                      local_uv_difference);
        return false;
      }
    }
  }

  // Return true if no issues found
  return true;
}

void remove_cones(const Eigen::MatrixXd &V,
                  const AffineManifold &affine_manifold,
                  Eigen::MatrixXd &pruned_V,
                  AffineManifold &pruned_affine_manifold,
                  std::vector<AffineManifold::Index> &cones,
                  std::vector<AffineManifold::Index> &removed_faces) {
  // Compute the cones
  affine_manifold.compute_cones(cones);
  spdlog::debug("Removing cones at {}", formatted_vector(cones, ", "));

  // Create boolean arrays of cone adjacent vertices
  std::vector<bool> is_cone_adjacent_vertex(affine_manifold.num_vertices());
  for (AffineManifold::Index vi = 0; vi < affine_manifold.num_vertices();
       ++vi) {
    is_cone_adjacent_vertex[vi] =
        affine_manifold.get_vertex_chart(vi).is_cone_adjacent;
  }

  // Create boolean arrays of cone adjacent faces
  std::vector<bool> is_cone_adjacent_face(affine_manifold.num_faces());
  for (AffineManifold::Index fi = 0; fi < affine_manifold.num_faces(); ++fi) {
    is_cone_adjacent_face[fi] =
        affine_manifold.get_face_chart(fi).is_cone_adjacent;
  }

  // Remove faces from VF meshes
  Eigen::MatrixXi const &F_orig = affine_manifold.get_faces();
  MatrixXr const &global_uv_orig = affine_manifold.get_global_uv();
  Eigen::MatrixXi const &F_uv_orig = affine_manifold.get_F_uv();
  Eigen::MatrixXi F;
  MatrixXr global_uv;
  Eigen::MatrixXi F_uv;
  remove_mesh_vertices(global_uv_orig, F_uv_orig, cones, global_uv, F_uv,
                       removed_faces);
  remove_mesh_faces(V, F_orig, removed_faces, pruned_V, F);

  // Remove faces from the cone adjacent arrays
  std::vector<bool> is_cone_adjacent_face_reindexed;
  std::vector<bool> is_cone_adjacent_vertex_reindexed;
  remove_vector_values<bool>(removed_faces, is_cone_adjacent_face,
                             is_cone_adjacent_face_reindexed);
  remove_vector_values<bool>(cones, is_cone_adjacent_vertex,
                             is_cone_adjacent_vertex_reindexed);

  // Make new affine manifold with cones removed
  pruned_affine_manifold = AffineManifold(F, global_uv, F_uv);

  // Mark cone adjacent faces
  for (size_t fi = 0; fi < is_cone_adjacent_face_reindexed.size(); ++fi) {
    if (is_cone_adjacent_face_reindexed[fi]) {
      pruned_affine_manifold.mark_cone_adjacent_face(fi);
    }
  }

  // Mark cone adjacent vertices
  for (size_t vi = 0; vi < is_cone_adjacent_vertex_reindexed.size(); ++vi) {
    if (is_cone_adjacent_vertex_reindexed[vi]) {
      pruned_affine_manifold.mark_cone_adjacent_vertex(vi);
    }
  }
}

// code added for C1 constraints

void AffineManifold::compute_edge_global_uv_mappings() {
  const auto &F = get_faces();
  const auto &F_uv = get_F_uv();
  const auto &uvs = get_global_uv();

  for (auto &e : m_edge_charts) {
    const int64_t top_f_idx = e.top_face_index;
    const int64_t bottom_f_idx = e.bottom_face_index;

    // top face
    const auto &f_top = F.row(top_f_idx);
    const auto &f_top_uv = F_uv.row(top_f_idx);

    std::array<int64_t, 3> top_local_idx = {{-1, -1, -1}}; // left, right, top
    for (int i = 0; i < 3; ++i) {
      if (f_top[i] == e.left_vertex_index) {
        top_local_idx[0] = i;
      }
      if (f_top[i] == e.right_vertex_index) {
        top_local_idx[1] = i;
      }
      if (f_top[i] == e.top_vertex_index) {
        top_local_idx[2] = i;
      }
    }

    for (int i = 0; i < 3; ++i) {
      assert(top_local_idx[i] > -1);
    }

    e.left_global_uv_idx = f_top_uv[top_local_idx[0]];
    e.right_global_uv_idx = f_top_uv[top_local_idx[1]];
    e.top_global_uv_idx = f_top_uv[top_local_idx[2]];

    e.left_global_uv_position = uvs.row(e.left_global_uv_idx);
    e.right_global_uv_position = uvs.row(e.right_global_uv_idx);
    e.top_global_uv_position = uvs.row(e.top_global_uv_idx);

    // bottom face
    const auto &f_bottom = F.row(bottom_f_idx);
    const auto &f_bottom_uv = F_uv.row(bottom_f_idx);

    std::array<int64_t, 3> bottom_local_idx = {
        {-1, -1, -1}}; // right (reverse left), left (reverse right), bottom
    for (int i = 0; i < 3; ++i) {
      if (f_bottom[i] == e.right_vertex_index) {
        bottom_local_idx[0] = i;
      }
      if (f_bottom[i] == e.left_vertex_index) {
        bottom_local_idx[1] = i;
      }
      if (f_bottom[i] == e.bottom_vertex_index) {
        bottom_local_idx[2] = i;
      }
    }

    for (int i = 0; i < 3; ++i) {
      assert(bottom_local_idx[i] > -1);
    }

    e.reverse_left_global_uv_idx = f_bottom_uv[bottom_local_idx[0]];
    e.reverse_right_global_uv_idx = f_bottom_uv[bottom_local_idx[1]];
    e.bottom_global_uv_idx = f_bottom_uv[bottom_local_idx[2]];

    e.reverse_left_global_uv_position = uvs.row(e.reverse_left_global_uv_idx);
    e.reverse_right_global_uv_position = uvs.row(e.reverse_right_global_uv_idx);
    e.bottom_global_uv_position = uvs.row(e.bottom_global_uv_idx);
  }
}

// redundant
void AffineManifold::compute_face_global_uv_mappings() { return; }

void AffineManifold::generate_lagrange_nodes() {
  // b0 b1 b2 b01 b10 b12 b21 b20 b02 b01^c b12^c b20^c b0c bc0 b1c bc1 b2c
  // bc2 bc

  const std::array<PlanarPoint, 19> CT_nodes = {{
      PlanarPoint(1., 0.),           // b0
      PlanarPoint(0., 1.),           // b1
      PlanarPoint(0., 0.),           // b2
      PlanarPoint(2. / 3., 1. / 3.), // b01
      PlanarPoint(1. / 3., 2. / 3.), // b10
      PlanarPoint(0., 2. / 3.),      // b12
      PlanarPoint(0., 1. / 3.),      // b21
      PlanarPoint(1. / 3., 0.),      // b20
      PlanarPoint(2. / 3., 0.),      // b02
      PlanarPoint(4. / 9., 4. / 9.), // b01^c
      PlanarPoint(1. / 9., 4. / 9.), // b12^c
      PlanarPoint(4. / 9., 1. / 9.), // b20^c
      PlanarPoint(7. / 9., 1. / 9.), // b0c
      PlanarPoint(5. / 9., 2. / 9.), // bc0
      PlanarPoint(1. / 9., 7. / 9.), // b1c
      PlanarPoint(2. / 9., 5. / 9.), // bc1
      PlanarPoint(1. / 9., 1. / 9.), // b2c
      PlanarPoint(2. / 9., 2. / 9.), // bc2
      PlanarPoint(1. / 3., 1. / 3.), // bc
  }};

  // std::map<int64_t, int64_t> v_to_lagrange_node_map;
  const auto &F = get_faces();
  // const auto &F_uv = get_F_uv();

  // compute cornor vertices
  for (size_t i = 0; i < m_face_charts.size(); ++i) {
    const auto &f_idx = m_face_charts[i].face_index;
    const auto &f = F.row(f_idx);

    for (int k = 0; k < 3; ++k) {
      if (v_to_lagrange_node_map.find(f[k]) != v_to_lagrange_node_map.end()) {
        // vertex already computed
        continue;
      } else {
        v_to_lagrange_node_map[f[k]] = m_lagrange_nodes.size();
        m_lagrange_nodes.emplace_back(i, CT_nodes[k]);
      }
    }
  }

  // compute nodes on edges and faces
  // std::map<std::pair<int64_t, int64_t>, std::array<int64_t, 4>>
  //     m_boundary_edge_to_node_map;

  for (size_t i = 0; i < m_face_charts.size(); ++i) {
    const auto &f_idx = m_face_charts[i].face_index;
    const auto &f = F.row(f_idx);

    std::array<int64_t, 19> l_vids = {-1}; // node indices for face

    // b0 b1 b2
    for (int k = 0; k < 3; ++k) {
      l_vids[k] = v_to_lagrange_node_map[f[k]];
    }

    // b01 b10 b12 b21 b20 b02
    for (int k = 0; k < 3; ++k) {
      if (m_boundary_edge_to_node_map.find(std::make_pair(
              f[(k + 1) % 3], f[k])) != m_boundary_edge_to_node_map.end()) {
        // if already computed its reverse edge
        const auto &vs =
            m_boundary_edge_to_node_map[std::make_pair(f[(k + 1) % 3], f[k])];
        l_vids[3 + k * 2 + 0] = vs[1];
        l_vids[3 + k * 2 + 1] = vs[0];
      } else {
        // create new nodes
        l_vids[3 + k * 2 + 0] = m_lagrange_nodes.size();
        m_lagrange_nodes.emplace_back(i, CT_nodes[3 + k * 2 + 0]);
        l_vids[3 + k * 2 + 1] = m_lagrange_nodes.size();
        m_lagrange_nodes.emplace_back(i, CT_nodes[3 + k * 2 + 1]);
        // add to edge-node map
        m_boundary_edge_to_node_map[std::make_pair(f[k], f[(k + 1) % 3])] = {
            {l_vids[3 + k * 2 + 0], l_vids[3 + k * 2 + 1]}};
      }
    }

    // b01^c b12^c b20^c b0c bc0 b1c bc1 b2c bc2 bc
    for (int k = 9; k < 19; ++k) {
      l_vids[k] = m_lagrange_nodes.size();
      m_lagrange_nodes.emplace_back(i, CT_nodes[k]);
    }

    for (int k = 0; k < 19; ++k) {
      assert(l_vids[k] > -1);
    }

    // can avoid copy, but fine now
    m_face_charts[i].lagrange_nodes = l_vids;

    // assign lagrange node to edge chart
    // TODO:
    // maybe outside this loop
  }

  assert(m_lagrange_nodes.size() == 10 * m_face_charts.size() +
                                        2 * m_edge_charts.size() +
                                        m_vertex_charts.size());
}

/*
get constraint matrices
*/

void AffineManifold::get_u_ij(Eigen::SparseMatrix<double> &u_ij_u,
                              Eigen::SparseMatrix<double> &u_ij_v) const {
  const int64_t N_L = m_lagrange_nodes.size();
  assert(size_t(N_L) == 10 * m_face_charts.size() + 2 * m_edge_charts.size() +
                            m_vertex_charts.size());
  u_ij_u.resize(N_L, N_L);
  u_ij_v.resize(N_L, N_L);

  std::vector<Eigen::Triplet<double>> triplets_u;
  std::vector<Eigen::Triplet<double>> triplets_v;
  for (const auto &e : m_edge_charts) {
    const auto ev0 = e.left_vertex_index;
    const auto ev1 = e.right_vertex_index;
    const auto ev0_pos = e.left_vertex_uv_position;
    const auto ev1_pos = e.right_vertex_uv_position;

    triplets_u.emplace_back(ev0, ev1, (ev1_pos - ev0_pos)[0]);
    triplets_u.emplace_back(ev1, ev0, (ev0_pos - ev1_pos)[0]);
    triplets_v.emplace_back(ev0, ev1, (ev1_pos - ev0_pos)[1]);
    triplets_v.emplace_back(ev1, ev0, (ev0_pos - ev1_pos)[1]);
  }

  u_ij_u.setFromTriplets(triplets_u.begin(), triplets_u.end());
  u_ij_v.setFromTriplets(triplets_v.begin(), triplets_v.end());
}
