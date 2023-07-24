cmake_minimum_required(VERSION 3.25.2)
project(cell_abm)

set(CMAKE_CXX_STANDARD 14)
enable_testing()

# Ensuring GoogleTest is Present
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

# Putting together the main executable & building relevant libraries:
add_subdirectory(src)

# Testing individual libraries:
add_subdirectory(tests)