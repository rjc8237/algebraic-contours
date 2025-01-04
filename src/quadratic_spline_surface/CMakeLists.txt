add_library(QuadraticSplineSurfaceLib
  compute_local_twelve_split_hessian.cpp
  optimize_spline_surface.cpp
  position_data.cpp
  powell_sabin_local_to_global_indexing.cpp
  quadratic_spline_surface.cpp
  quadratic_spline_surface_patch.cpp
  twelve_split_spline.cpp
  quadratic_spline_surface_msh_writer.cpp
)
target_include_directories(QuadraticSplineSurfaceLib PUBLIC .)
target_link_libraries(QuadraticSplineSurfaceLib PUBLIC
  QuadraticContoursCoreLib
)
target_compile_definitions(QuadraticSplineSurfaceLib PUBLIC
  SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG
  # SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_OFF
)

target_compile_options(QuadraticSplineSurfaceLib PRIVATE
  -Wall -Wpedantic -Wextra -Werror
)