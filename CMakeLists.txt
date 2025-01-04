cmake_minimum_required(VERSION 3.25)
project(AlgebraicContours VERSION 0.0.1)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

# Fix thread issue
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
endif()

# Get required packages
list(PREPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Set compile definitions
add_compile_definitions(EIGEN_STACK_ALLOCATION_LIMIT=0)
add_compile_definitions(CHECK_VALIDITY)

# Fetch libigl
include(libigl)

# Fetch other dependencies
include(polyscope)
include(eigen)
include(spdlog)
include(suitesparse)
include(cli11)
add_subdirectory(${PROJECT_SOURCE_DIR}/ext/svg)

# Install executables to bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Build main library
add_subdirectory(src/core)
add_subdirectory(src/quadratic_spline_surface)
add_subdirectory(src/contour_network)
add_subdirectory(src/clough_tocher_patches)
target_include_directories(QuadraticContoursCoreLib PUBLIC ${PROJECT_SOURCE_DIR}/ext )
add_library(QuadraticContoursLib INTERFACE)
target_link_libraries(QuadraticContoursLib INTERFACE
  QuadraticContoursCoreLib
  QuadraticSplineSurfaceLib
  ContourNetworkLib
  CloughTocherPatchLib
)

# Build utility library
add_subdirectory(src/util)

# Optionally build executables
option(BUILD_EXECUTABLES  "Build algebraic contour executables"  ON)
if (BUILD_EXECUTABLES)
  add_subdirectory(src/exec)
endif()

# Optionally build experiments
option(BUILD_FIGURE_EXECUTABLES "Build executables to generate paper figures"  ON)
if (BUILD_FIGURE_EXECUTABLES)
  message("Building figure executables")
  add_subdirectory(src/figures)
endif()

# Optionally build tests
option(BUILD_TESTS  "Build contour tests"  ON)
if (BUILD_TESTS)
  message("Building tests")
  add_subdirectory(src/tests)
endif()

