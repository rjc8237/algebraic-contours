#include "clough_tocher_surface.hpp"

#include <fstream>
#include <igl/edges.h>
#include <igl/per_vertex_normals.h>

#include "clough_tocher_constraint_matrices.hpp"

CloughTocherSurface::CloughTocherSurface() {}

CloughTocherSurface::CloughTocherSurface(
    const Eigen::MatrixXd &V, const AffineManifold &affine_manifold,
    const OptimizationParameters &optimization_params,
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

  // compute patches
  assert(m_corner_data.size() == m_midpoint_data.size());
  for (size_t i = 0; i < m_corner_data.size(); ++i) {
    m_patches.push_back(
        CloughTocherPatch(m_corner_data[i], m_midpoint_data[i]));
  }
}

Eigen::Matrix<double, 1, 3>
CloughTocherSurface::evaluate_patch(const PatchIndex &patch_index,
                                    const double &u, const double &v,
                                    const double &w) {
  return m_patches[patch_index].CT_eval(u, v, w);
}

void CloughTocherSurface::generate_face_normals(
    const Eigen::MatrixXd &V, const AffineManifold &affine_manifold,
    Eigen::MatrixXd &N) {
  Eigen::MatrixXi const &F = affine_manifold.get_faces();

  // Compute the cones of the affine manifold
  std::vector<AffineManifold::Index> cones;
  affine_manifold.compute_cones(cones);

  std::cout << "#Cone: " << cones.size() << std::endl;

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

void CloughTocherSurface::write_cubic_surface_to_msh_no_conn(
    std::string filename) {
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
  const std::array<PlanarPoint, 10> tri_0_bcs = {
      {PlanarPoint(0, 0), PlanarPoint(1., 0), PlanarPoint(1. / 3., 1. / 3.),
       PlanarPoint(1. / 3., 0), PlanarPoint(2. / 3., 0),
       PlanarPoint(7. / 9., 1. / 9.), PlanarPoint(5. / 9., 2. / 9.),
       PlanarPoint(2. / 9., 2. / 9.), PlanarPoint(1. / 9., 1. / 9.),
       PlanarPoint(4. / 9., 1. / 9.)}};

  const std::array<PlanarPoint, 10> tri_1_bcs = {
      {PlanarPoint(1, 0), PlanarPoint(0, 1), PlanarPoint(1. / 3., 1. / 3.),
       PlanarPoint(2. / 3., 1. / 3.), PlanarPoint(1. / 3., 2. / 3.),
       PlanarPoint(1. / 9., 7. / 9.), PlanarPoint(2. / 9., 5. / 9.),
       PlanarPoint(5. / 9., 2. / 9.), PlanarPoint(7. / 9., 1. / 9.),
       PlanarPoint(4. / 9., 4. / 9.)}};

  const std::array<PlanarPoint, 10> tri_2_bcs = {
      {PlanarPoint(0, 1), PlanarPoint(0, 0), PlanarPoint(1. / 3., 1. / 3.),
       PlanarPoint(0, 2. / 3.), PlanarPoint(0, 1. / 3.),
       PlanarPoint(1. / 9., 1. / 9.), PlanarPoint(2. / 9., 2. / 9.),
       PlanarPoint(2. / 9., 5. / 9.), PlanarPoint(1. / 9., 7. / 9.),
       PlanarPoint(1. / 9., 4. / 9.)}};

  file << "$Nodes\n";

  const size_t node_size = m_patches.size() * 30;
  file << "1 " << node_size << " 1 " << node_size << "\n";
  file << "2 1 0 " << node_size << "\n";

  for (size_t i = 1; i <= node_size; ++i) {
    file << i << "\n";
  }

  for (const auto &patch : m_patches) {
    for (const auto &bc : tri_0_bcs) {
      auto z = patch.CT_eval(bc[0], bc[1], 1 - bc[0] - bc[1]);
      file << z(0, 0) << " " << z(0, 1) << " " << z(0, 2) << "\n";
    }
    for (const auto &bc : tri_1_bcs) {
      auto z = patch.CT_eval(bc[0], bc[1], 1 - bc[0] - bc[1]);
      file << z(0, 0) << " " << z(0, 1) << " " << z(0, 2) << "\n";
    }
    for (const auto &bc : tri_2_bcs) {
      auto z = patch.CT_eval(bc[0], bc[1], 1 - bc[0] - bc[1]);
      file << z(0, 0) << " " << z(0, 1) << " " << z(0, 2) << "\n";
    }
  }

  file << "$EndNodes\n";

  // write elements
  const size_t element_size = m_patches.size() * 3;

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

void CloughTocherSurface::write_coeffs_to_obj(std::string filename) {
  std::ofstream file(filename);

  for (const auto &patch : m_patches) {
    std::array<Eigen::Matrix<double, 10, 3>, 3> coeffs = patch.get_coeffs();
    for (int i = 0; i < 1; ++i) {
      for (int j = 0; j < 10; ++j) {
        file << "v " << coeffs[i](j, 0) << " " << coeffs[i](j, 1) << " "
             << coeffs[i](j, 2) << "\n";
      }
    }
  }

  file << "f 1 1 1\n";
}

void CloughTocherSurface::sample_to_obj(std::string filename, int sample_size) {
  std::ofstream file(filename);

  for (const auto &patch : m_patches) {
    for (int i = 0; i <= sample_size; ++i) {
      for (int j = 0; j <= sample_size - i; ++j) {
        double u = 1. / sample_size * i;
        double v = 1. / sample_size * j;
        double w = 1 - u - v;

        // std::cout << "u: " << u << " v: " << v << " w: " << w << std::endl;
        // std::cout << "w^3: " << w * w * w << std::endl;
        auto z = patch.CT_eval(u, v, w);

        // std::cout << z << std::endl;

        file << "v " << z[0] << " " << z[1] << " " << z[2] << '\n';
      }
    }
  }

  file << "f 1 1 1\n";
}

void CloughTocherSurface::write_cubic_surface_to_msh_with_conn(
    std::string filename) {
  std::ofstream file(filename + ".msh");

  /*
    $MeshFormat
      4.1 0 8     MSH4.1, ASCII
      $EndMeshFormat
    */

  file << "$MeshFormat\n"
       << "4.1 0 8\n"
       << "$EndMeshFormat\n";

  /*
  subtri0: 0 4 5 1 12 13 3 11 10 16
  subtri1: 1 6 7 2 14 15 3 13 12 17
  subtri2: 2 8 9 0 10 11 3 15 14 18
  */

  const std::array<PlanarPoint, 19> CT_nodes = {{
      PlanarPoint(0., 0.),           PlanarPoint(1., 0.),
      PlanarPoint(0., 1.),           PlanarPoint(1. / 3., 1. / 3.),
      PlanarPoint(1. / 3., 0.),      PlanarPoint(2. / 3., 0.),
      PlanarPoint(2. / 3., 1. / 3.), PlanarPoint(1. / 3., 2. / 3.),
      PlanarPoint(0., 2. / 3.),      PlanarPoint(0., 1. / 3.),
      PlanarPoint(1. / 9., 1. / 9.), PlanarPoint(2. / 9., 2. / 9.),
      PlanarPoint(7. / 9., 1. / 9.), PlanarPoint(5. / 9., 2. / 9.),
      PlanarPoint(1. / 9., 7. / 9.), PlanarPoint(2. / 9., 5. / 9.),
      PlanarPoint(4. / 9., 1. / 9.), PlanarPoint(4. / 9., 4. / 9.),
      PlanarPoint(1. / 9., 4. / 9.),
  }};

  // std::vector<Eigen::Vector3d> vertices;
  std::map<std::pair<int64_t, int64_t>, std::array<int64_t, 4>>
      boundary_edge_to_v_map;
  std::vector<std::array<int64_t, 10>> faces;
  std::map<int64_t, int64_t> v_to_v_map;
  std::vector<Eigen::Vector3d> vertices;
  const auto &F = m_affine_manifold.get_faces();

  Eigen::MatrixXd edges;
  igl::edges(F, edges);

  Eigen::MatrixXd edges_uv;
  igl::edges(m_affine_manifold.get_F_uv(), edges_uv);

  // checks
  std::cout << "#3d face edges: " << edges.rows() << std::endl;
  std::cout << "#uv face edges: " << edges_uv.rows() << std::endl;
  std::cout << "#edge charts: " << m_affine_manifold.m_edge_charts.size()
            << std::endl;
  for (const auto &e : m_affine_manifold.m_edge_charts) {
    if (e.is_boundary) {
      std::cout << "boundary" << std::endl;
    }
  }

  for (const auto &v : m_affine_manifold.m_vertex_charts) {
    if (v.is_cone) {
      std::cout << "cone" << std::endl;
    }
  }

  // compute corner vertices first
  for (size_t p_idx = 0; p_idx < m_patches.size(); p_idx++) {
    const auto &patch = m_patches[p_idx];
    // idk why but this is  2 0 1 not 0 1 2, maybe because of the half edge data
    // structure
    std::array<int64_t, 3> Fv = {{F(p_idx, 2), F(p_idx, 0), F(p_idx, 1)}};
    for (int i = 0; i < 3; ++i) {
      if (v_to_v_map.find(Fv[i]) != v_to_v_map.end()) {
        // vertex already computed
        continue;
      } else {
        auto z = patch.CT_eval(CT_nodes[i][0], CT_nodes[i][1],
                               1 - CT_nodes[i][0] - CT_nodes[i][1]);
        v_to_v_map[Fv[i]] = vertices.size();
        vertices.push_back(z);
      }
    }
  }

  assert(size_t(F.rows()) == m_patches.size());

  for (size_t p_idx = 0; p_idx < m_patches.size(); p_idx++) {
    std::array<int64_t, 19> l_vids = {-1};
    // idk why but this is  2 0 1 not 0 1 2, maybe because of the half edge data
    // structure
    std::array<int64_t, 3> Fv = {{F(p_idx, 2), F(p_idx, 0), F(p_idx, 1)}};
    const auto &patch = m_patches[p_idx];

    // node 0 - 2
    for (int i = 0; i < 3; ++i) {
      l_vids[i] = v_to_v_map[Fv[i]];
    }

    // node 3
    auto zz = patch.CT_eval(CT_nodes[3][0], CT_nodes[3][1],
                            1 - CT_nodes[3][0] - CT_nodes[3][1]);
    l_vids[3] = vertices.size();
    vertices.push_back(zz);

    // node 4 5 6 7 8 9
    for (int i = 0; i < 3; ++i) {
      if (boundary_edge_to_v_map.find(std::make_pair(Fv[(i + 1) % 3], Fv[i])) !=
          boundary_edge_to_v_map.end()) {
        // this edge is processed in some other patch
        const auto &vs =
            boundary_edge_to_v_map[std::make_pair(Fv[(i + 1) % 3], Fv[i])];
        l_vids[4 + i * 2 + 0] = vs[1];
        l_vids[4 + i * 2 + 1] = vs[0];
      } else {
        // eval new vertices
        auto z0 = patch.CT_eval(
            CT_nodes[4 + i * 2 + 0][0], CT_nodes[4 + i * 2 + 0][1],
            1 - CT_nodes[4 + i * 2 + 0][0] - CT_nodes[4 + i * 2 + 0][1]);
        auto z1 = patch.CT_eval(
            CT_nodes[4 + i * 2 + 1][0], CT_nodes[4 + i * 2 + 1][1],
            1 - CT_nodes[4 + i * 2 + 1][0] - CT_nodes[4 + i * 2 + 1][1]);
        l_vids[4 + i * 2 + 0] = vertices.size();
        vertices.push_back(z0);
        l_vids[4 + i * 2 + 1] = vertices.size();
        vertices.push_back(z1);

        boundary_edge_to_v_map[std::make_pair(Fv[i], Fv[(i + 1) % 3])] = {
            {l_vids[4 + i * 2 + 0], l_vids[4 + i * 2 + 0 + 1]}};
      }
    }

    // node 10 - 18
    for (int i = 10; i < 19; ++i) {
      auto z = patch.CT_eval(CT_nodes[i][0], CT_nodes[i][1],
                             1 - CT_nodes[i][0] - CT_nodes[i][1]);
      l_vids[i] = vertices.size();
      vertices.push_back(z);
    }

    /*
    subtri0: 0 1 3 4 5 12 13 11 10 16
    subtri1: 1 2 3 6 7 14 15 13 12 17
    subtri2: 2 0 3 8 9 10 11 15 14 18
    */
    faces.push_back(
        {{l_vids[0] + 1, l_vids[1] + 1, l_vids[3] + 1, l_vids[4] + 1,
          l_vids[5] + 1, l_vids[12] + 1, l_vids[13] + 1, l_vids[11] + 1,
          l_vids[10] + 1, l_vids[16] + 1}});
    faces.push_back(
        {{l_vids[1] + 1, l_vids[2] + 1, l_vids[3] + 1, l_vids[6] + 1,
          l_vids[7] + 1, l_vids[14] + 1, l_vids[15] + 1, l_vids[13] + 1,
          l_vids[12] + 1, l_vids[17] + 1}});
    faces.push_back(
        {{l_vids[2] + 1, l_vids[0] + 1, l_vids[3] + 1, l_vids[8] + 1,
          l_vids[9] + 1, l_vids[10] + 1, l_vids[11] + 1, l_vids[15] + 1,
          l_vids[14] + 1, l_vids[18] + 1}});

    // // debug code
    // for (size_t i = 0; i < l_vids.size(); ++i) {
    //   std::cout << l_vids[i] << ": " << CT_nodes[i] << std::endl;
    // }
    // std::cout << std::endl;
    // if (p_idx == 2)
    //   break;
  }

  file << "$Nodes\n";

  const size_t node_size = vertices.size();
  file << "1 " << node_size << " 1 " << node_size << "\n";
  file << "2 1 0 " << node_size << "\n";

  for (size_t i = 1; i <= node_size; ++i) {
    file << i << "\n";
  }

  for (size_t i = 0; i < node_size; ++i) {
    file << vertices[i][0] << " " << vertices[i][1] << " " << vertices[i][2]
         << "\n";
  }

  file << "$EndNodes\n";

  // write elements
  // assert(m_patches.size() * 3 == faces.size());
  const size_t element_size = faces.size();

  file << "$Elements\n";
  file << "1 " << element_size << " 1 " << element_size << "\n";
  file << "2 1 21 " << element_size << "\n";
  for (size_t i = 0; i < element_size; ++i) {
    file << i + 1 << " ";
    for (int j = 0; j < 10; ++j) {
      file << faces[i][j] << " ";
    }
    file << "\n";
  }

  file << "$EndElements\n";

  // mark cones
  // const auto &cone_indices = m_affine_manifold.generate_cones();

  // file << "$NodeData\n";
  // file << "1\n";                       // num string tags
  // file << "\"Cone\"\n";                // string tag
  // file << "1\n";                       // num real tags
  // file << "0.0\n";                     // time step starts
  // file << "3\n";                       // three integer tags
  // file << "0\n";                       // time step
  // file << "1\n";                       // num field
  // file << cone_indices.size() << "\n"; // num associated nodal values
  // for (const auto &idx : cone_indices) {
  //   file << v_to_v_map[idx] + 1 << " 1.0\n";
  // }
  // file << "$EndNodeData\n";

  std::ofstream v_map_file(filename + "_input_v_to_output_v_map.txt");
  for (const auto &pair : v_to_v_map) {
    v_map_file << pair.first << " " << pair.second << std::endl;
  }
}

void CloughTocherSurface::
    write_cubic_surface_to_msh_with_conn_from_lagrange_nodes(
        std::string filename) {
  std::ofstream file(filename + ".msh");

  /*
    $MeshFormat
      4.1 0 8     MSH4.1, ASCII
      $EndMeshFormat
    */

  file << "$MeshFormat\n"
       << "4.1 0 8\n"
       << "$EndMeshFormat\n";

  /*
  subtri0: 0 1 18 3 4 14 15 13 12 9
  b0 b1 bc b01 b10 b1c bc1 bc0 b0c b01^c
  subtri1: 1 2 18 5 6 16 17 15 14 10
  b1 b2 bc b12 b21 b2c bc2 bc1 b1c b12^c
  subtri2: 2 0 18 7 8 12 13 17 16 11
  b2 b0 bc b20 b02 b0c bc0 bc2 b2c b20^c
  */

  // m_affine_manifold.generate_lagrange_nodes();

  const auto &lagrange_nodes = m_affine_manifold.m_lagrange_nodes;
  // evaluate vertices
  std::vector<Eigen::Vector3d> vertices;
  for (size_t i = 0; i < lagrange_nodes.size(); ++i) {
    const auto patch_idx = lagrange_nodes[i].first;
    const auto bc = lagrange_nodes[i].second;
    auto z = m_patches[patch_idx].CT_eval(bc[0], bc[1], 1. - bc[0] - bc[1]);
    vertices.push_back(z);
  }

  // build faces
  std::vector<std::array<int64_t, 10>> faces;
  for (const auto &f_chart : m_affine_manifold.m_face_charts) {
    const auto &l_nodes = f_chart.lagrange_nodes;
    faces.push_back(
        {{l_nodes[0], l_nodes[1], l_nodes[18], l_nodes[3], l_nodes[4],
          l_nodes[14], l_nodes[15], l_nodes[13], l_nodes[12], l_nodes[9]}});
    faces.push_back(
        {{l_nodes[1], l_nodes[2], l_nodes[18], l_nodes[5], l_nodes[6],
          l_nodes[16], l_nodes[17], l_nodes[15], l_nodes[14], l_nodes[10]}});
    faces.push_back(
        {{l_nodes[2], l_nodes[0], l_nodes[18], l_nodes[7], l_nodes[8],
          l_nodes[12], l_nodes[13], l_nodes[17], l_nodes[16], l_nodes[11]}});
  }

  file << "$Nodes\n";

  const size_t node_size = vertices.size();
  file << "1 " << node_size << " 1 " << node_size << "\n";
  file << "2 1 0 " << node_size << "\n";

  for (size_t i = 1; i <= node_size; ++i) {
    file << i << "\n";
  }

  for (size_t i = 0; i < node_size; ++i) {
    file << vertices[i][0] << " " << vertices[i][1] << " " << vertices[i][2]
         << "\n";
  }

  file << "$EndNodes\n";

  // write elements
  // assert(m_patches.size() * 3 == faces.size());
  const size_t element_size = faces.size();

  file << "$Elements\n";
  file << "1 " << element_size << " 1 " << element_size << "\n";
  file << "2 1 21 " << element_size << "\n";
  for (size_t i = 0; i < element_size; ++i) {
    file << i + 1 << " ";
    for (int j = 0; j < 10; ++j) {
      file << faces[i][j] + 1 << " ";
    }
    file << "\n";
  }

  file << "$EndElements\n";

  // mark cones
  // const auto &cone_indices = m_affine_manifold.generate_cones();

  // file << "$NodeData\n";
  // file << "1\n";                       // num string tags
  // file << "\"Cone\"\n";                // string tag
  // file << "1\n";                       // num real tags
  // file << "0.0\n";                     // time step starts
  // file << "3\n";                       // three integer tags
  // file << "0\n";                       // time step
  // file << "1\n";                       // num field
  // file << cone_indices.size() << "\n"; // num associated nodal values
  // for (const auto &idx : cone_indices) {
  //   file << v_to_v_map[idx] + 1 << " 1.0\n";
  // }
  // file << "$EndNodeData\n";

  std::ofstream v_map_file(filename + "_input_v_to_output_v_map.txt");
  for (const auto &pair : m_affine_manifold.v_to_lagrange_node_map) {
    v_map_file << pair.first << " " << pair.second << std::endl;
  }
}

void CloughTocherSurface::P_G2F(Eigen::SparseMatrix<double> &m) {
  const auto N_L = m_affine_manifold.m_lagrange_nodes.size();
  const auto F_cnt = m_affine_manifold.m_face_charts.size();

  m.resize(19 * F_cnt, N_L);

  std::vector<Eigen::Triplet<double>> triplets;

  const auto &face_charts = m_affine_manifold.m_face_charts;
  for (size_t i = 0; i < face_charts.size(); ++i) {
    for (int j = 0; j < 19; ++j) {
      triplets.emplace_back(i * 19 + j, face_charts[i].lagrange_nodes[j], 1);
    }
  }

  m.setFromTriplets(triplets.begin(), triplets.end());
}

void CloughTocherSurface::C_L_int(Eigen::Matrix<double, 7, 19> &m) {
  Eigen::Matrix<double, 12, 12> L_L2d_ind = L_L2d_ind_m();
  Eigen::Matrix<double, 7, 12> L_d2L_dep = L_d2L_dep_m();

  Eigen::Matrix<double, 7, 12> neg_L_dot_L = -L_d2L_dep * L_L2d_ind;
  m.block<7, 12>(0, 0) = neg_L_dot_L;
  m.block<7, 7>(0, 12) = Eigen::MatrixXd::Identity(7, 7);
}

void CloughTocherSurface::C_F_int(Eigen::SparseMatrix<double> &m) {
  const auto N_L = m_affine_manifold.m_lagrange_nodes.size();
  const auto F_cnt = m_affine_manifold.m_face_charts.size();

  Eigen::SparseMatrix<double> p_g2f;
  P_G2F(p_g2f);
  Eigen::Matrix<double, 7, 19> c_l_int;
  C_L_int(c_l_int);

  Eigen::SparseMatrix<double> C_diag;
  C_diag.resize(7 * F_cnt, 19 * F_cnt);

  for (size_t i = 0; i < F_cnt; ++i) {
    for (int j = 0; j < 7; ++j) {
      for (int k = 0; k < 19; ++k) {
        C_diag.insert(i * 7 + j, i * 19 + k) = c_l_int(j, k);
      }
    }
  }

  m.resize(7 * F_cnt, N_L);
  std::cout << 7 * F_cnt << " " << N_L << std::endl;

  m = C_diag * p_g2f;
  std::cout << m.rows() << " " << m.cols() << std::endl;
}
