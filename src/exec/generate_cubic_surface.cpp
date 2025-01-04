#include "apply_transformation.h"
#include "common.h"
#include "compute_boundaries.h"
#include "contour_network.h"
#include "generate_transformation.h"
#include "globals.cpp"
#include "twelve_split_spline.h"
#include <CLI/CLI.hpp>
#include <fstream>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>

#include "clough_tocher_surface.hpp"

int main(int argc, char *argv[]) {
  // Build maps from strings to enums
  std::map<std::string, spdlog::level::level_enum> log_level_map{
      {"trace", spdlog::level::trace},       {"debug", spdlog::level::debug},
      {"info", spdlog::level::info},         {"warn", spdlog::level::warn},
      {"critical", spdlog::level::critical}, {"off", spdlog::level::off},
  };

  // Get command line arguments
  CLI::App app{"Generate Clough-Tocher cubic surface mesh."};
  std::string input_filename = "";
  std::string output_dir = "./";
  spdlog::level::level_enum log_level = spdlog::level::off;
  Eigen::Matrix<double, 3, 1> color = SKY_BLUE;
  int num_subdivisions = DISCRETIZATION_LEVEL;
  OptimizationParameters optimization_params;
  double weight = optimization_params.position_difference_factor;
  app.add_option("-i,--input", input_filename, "Mesh filepath")
      ->check(CLI::ExistingFile)
      ->required();
  app.add_option("--log_level", log_level, "Level of logging")
      ->transform(CLI::CheckedTransformer(log_level_map, CLI::ignore_case));
  app.add_option("--num_subdivisions", num_subdivisions,
                 "Number of subdivisions")
      ->check(CLI::PositiveNumber);
  app.add_option("-w,--weight", weight,
                 "Fitting weight for the quadratic surface approximation")
      ->check(CLI::PositiveNumber);
  CLI11_PARSE(app, argc, argv);

  // Set logger level
  spdlog::set_level(log_level);

  // Set optimization parameters
  optimization_params.position_difference_factor = weight;

  // Get input mesh
  Eigen::MatrixXd V, uv, N;
  Eigen::MatrixXi F, FT, FN;
  igl::readOBJ(input_filename, V, uv, N, F, FT, FN);

  // Generate quadratic spline
  spdlog::info("Computing spline surface");
  //   std::vector<std::vector<int>> face_to_patch_indices;
  //   std::vector<int> patch_to_face_indices;
  Eigen::SparseMatrix<double> fit_matrix;
  Eigen::SparseMatrix<double> energy_hessian;
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>>
      energy_hessian_inverse;
  AffineManifold affine_manifold(F, uv, FT);

  CloughTocherSurface ct_surface(V, affine_manifold, optimization_params,
                                 fit_matrix, energy_hessian,
                                 energy_hessian_inverse);

  ct_surface.m_affine_manifold.generate_lagrange_nodes();

  ct_surface.write_coeffs_to_obj("test_cubic_points.obj");

  ct_surface.sample_to_obj("test_sample_cubic_points.obj", 25);

  ct_surface.write_cubic_surface_to_msh_no_conn("test_cubic_sphere.msh");
  //   ct_surface.write_cubic_surface_to_msh_with_conn("duck_with_conn");

  ct_surface.write_cubic_surface_to_msh_with_conn_from_lagrange_nodes(
      "icosphere_from_lagrange_nodes");

  Eigen::SparseMatrix<double> c_f_int;
  ct_surface.C_F_int(c_f_int);
  Eigen::SparseMatrix<double> C_E_end;
  ct_surface.C_E_end(C_E_end);
  Eigen::SparseMatrix<double> C_E_mid;
  ct_surface.C_E_mid(C_E_mid);

  std::ofstream file("interior_constraint_matrix.txt");
  file << std::setprecision(16) << c_f_int;
  std::ofstream file_2("edge_endpoint_constraint_matrix.txt");
  file_2 << std::setprecision(16) << c_f_int;
  std::ofstream file_3("edge_midpoint_constraint_matrix.txt");
  file_3 << std::setprecision(16) << c_f_int;

  file.close();
  file_2.close();
  file_3.close();

  //   std::ofstream file_4("interior_constraint_matrix_triplets.txt");
  //   const auto trip1 = c_f_int.to_triplets();
  //   std::ofstream file_5("edge_endpoint_constraint_matrix_triplets.txt");
  //   std::ofstream file_6("edge_midpoint_constraint_matrix_triplets.txt");

  //   file_4.close();
  //   file_5.close();
  //   file_6.close();

  return 0;
}
