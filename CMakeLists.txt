cmake_minimum_required(VERSION 3.24.3)
project(cell_abm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_WARN_DEPRECATED FALSE)

add_definitions("-DBOOST_ALLOW_DEPRECATED_HEADERS")
add_definitions("-DBOOST_BIND_GLOBAL_PLACEHOLDERS")
enable_testing()

set(CMAKE_CXX_CLANG_TIDY
  clang-tidy;
  -format-style='file';
  -header-filter=${CMAKE_CURRENT_SOURCE_DIR};
)

add_compile_options(
  -Wno-deprecated-declarations
)

# Ensuring GoogleTest is Present
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

# Ensuring Boost is present:
set(Boost_INCLUDE_DIR /usr/local/Cellar/boost/1.83.0/include)
find_package(Boost COMPONENTS regex system filesystem program_options REQUIRED)
message(STATUS ${Boost_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIR})

# Putting together the main executable & building relevant libraries:
add_subdirectory(src)

# Testing individual libraries:
add_subdirectory(tests)