add_executable(generate_algebraic_contours generate_algebraic_contours.cpp)
target_link_libraries(generate_algebraic_contours PRIVATE
  QuadraticContoursLib
  CLI11::CLI11
)
target_compile_options(generate_algebraic_contours PRIVATE
  -Wall -Wpedantic -Wextra -Werror
)

add_executable(view_quadratic_spline view_quadratic_spline.cpp)
target_link_libraries(view_quadratic_spline PRIVATE
  QuadraticContoursLib
  CLI11::CLI11
)
target_compile_options(view_quadratic_spline PRIVATE
  -Wall -Wpedantic -Wextra -Werror
)


add_executable(generate_cubic_surface generate_cubic_surface.cpp)
target_link_libraries(generate_cubic_surface PRIVATE
  QuadraticContoursLib
  CloughTocherPatchLib
  CLI11::CLI11
)
target_compile_options(generate_cubic_surface PRIVATE
  -Wall -Wpedantic -Wextra -Werror
)
