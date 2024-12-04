#include "quadratic_spline_surface.h"

void QuadraticSplineSurface::write_cubic_nodes_to_msh(std::string filename) {
  std::ofstream file(filename);

  /*
  $MeshFormat
    4.1 0 8     MSH4.1, ASCII
    $EndMeshFormat
  */

  file << "$MeshFormat\n"
       << "4.1 0 8\n"
       << "$EndMeshFormat\n";

  // msh 10 node 3rd-order triangle nodes
  const std::array<PlanarPoint, 10> normalized_vs = {
      {PlanarPoint(0, 0), PlanarPoint(1., 0), PlanarPoint(0, 1.),
       PlanarPoint(1. / 3., 0), PlanarPoint(2. / 3., 0),
       PlanarPoint(2. / 3., 1. / 3.), PlanarPoint(1. / 3., 2. / 3.),
       PlanarPoint(0, 2. / 3.), PlanarPoint(0, 1. / 3.),
       PlanarPoint(1. / 3., 1. / 3.)}};

  file << "$Nodes\n";

  const size_t node_size = m_patches.size() * 10;
  file << "1 " << node_size << " 1 " << node_size << "\n";
  file << "2 1 0 " << node_size << "\n";

  for (size_t i = 1; i <= node_size; ++i) {
    file << i << "\n";
  }

  for (const auto &patch : m_patches) {
    for (const auto &normalized_v : normalized_vs) {
      PlanarPoint nv = normalized_v;
      auto domain_v = patch.denormalize_domain_point(nv);
      SpatialVector surface_point;
      patch.evaluate(domain_v, surface_point);
      file << surface_point(0, 0) << " " << surface_point(0, 1) << " "
           << surface_point(0, 2) << "\n";
    }
  }

  file << "$EndNodes\n";

  // write elements
  const size_t element_size = m_patches.size();

  file << "$Elements\n";
  file << "1 " << element_size << " 1 " << element_size << "\n";
  file << "2 1 21 " << element_size << "\n";
  for (size_t i = 0; i < element_size; ++i) {
    file << i + 1 << " " << i * 10 + 1 << " " << i * 10 + 2 << " " << i * 10 + 3
         << " " << i * 10 + 4 << " " << i * 10 + 5 << " " << i * 10 + 6 << " "
         << i * 10 + 7 << " " << i * 10 + 8 << " " << i * 10 + 9 << " "
         << i * 10 + 10 << "\n";
  }

  file << "$EndElements\n";
}

void QuadraticSplineSurface::write_cubic_nodes_to_obj(std::string filename) {
  std::ofstream file(filename);

  // msh 10 node 3rd-order triangle nodes
  const std::array<PlanarPoint, 10> normalized_vs = {
      {PlanarPoint(0, 0), PlanarPoint(1., 0), PlanarPoint(0, 1.),
       PlanarPoint(1. / 3., 0), PlanarPoint(2. / 3., 0),
       PlanarPoint(2. / 3., 1. / 3.), PlanarPoint(1. / 3., 2. / 3.),
       PlanarPoint(0, 2. / 3.), PlanarPoint(0, 1. / 3.),
       PlanarPoint(1. / 3., 1. / 3.)}};

  for (const auto &patch : m_patches) {
    for (const auto &normalized_v : normalized_vs) {
      PlanarPoint nv = normalized_v;
      auto domain_v = patch.denormalize_domain_point(nv);
      SpatialVector surface_point;
      patch.evaluate(domain_v, surface_point);
      file << "v " << surface_point(0, 0) << " " << surface_point(0, 1) << " "
           << surface_point(0, 2) << "\n";
    }
  }

  // write elements
  const size_t element_size = m_patches.size();

  for (size_t i = 0; i < element_size; ++i) {
    file << "f " << " " << i * 10 + 1 << " " << i * 10 + 2 << " " << i * 10 + 3
         << "\n";
  }
}

void QuadraticSplineSurface::write_corner_patch_points_to_obj(
    std::string filename) {
  std::ofstream file(filename);

  const std::array<PlanarPoint, 3> normalized_vs = {
      {PlanarPoint(0, 0), PlanarPoint(1, 0), PlanarPoint(0, 1)}};

  for (const auto &patch : m_patches) {
    for (const auto &normalized_v : normalized_vs) {
      PlanarPoint nv = normalized_v;
      auto domain_v = patch.denormalize_domain_point(nv);
      SpatialVector surface_point;
      patch.evaluate(domain_v, surface_point);
      file << "v " << surface_point(0, 0) << " " << surface_point(0, 1) << " "
           << surface_point(0, 2) << std::endl;
    }
  }
  for (size_t i = 0; i < m_patches.size(); ++i) {
    file << "f " << i * 3 + 1 << " " << i * 3 + 2 << " " << i * 3 + 3
         << std::endl;
  }

  file.close();
}