cmake_minimum_required(VERSION 3.24.3)
project(cell_abm)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_WARN_DEPRECATED FALSE)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
message(STATUS ${CMAKE_BUILD_TYPE})

add_definitions("-DBOOST_ALLOW_DEPRECATED_HEADERS")
add_definitions("-DBOOST_BIND_GLOBAL_PLACEHOLDERS")
enable_testing()

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# Command line flags to force use of non-apple clang:
# cmake -B . -S .. -DCMAKE_C_COMPILER=/opt/homebrew/bin/gcc-13 -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-13

message(STATUS "${CMAKE_CXX_COMPILER_ID}")

# Enable Clang Tidy only if using Clang
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  message(STATUS "Using Clang compiler!")
else()
  message(STATUS "Not using Clang!")
endif()

add_compile_options(
  -Wno-deprecated-declarations
)

# Linking against OpenMP if present:
find_package(OpenMP)

# Ensuring GoogleTest is present:
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

# Ensuring Boost is present:
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /usr/local/boost_1_84_0/stage/lib)
# ^ location of GCC compiled libraries on my local machine (for OpenMP access on Mac)

find_package(Boost 1.81.0 REQUIRED COMPONENTS program_options filesystem)
include_directories(${Boost_INCLUDE_DIR})
message(STATUS ${Boost_INCLUDE_DIR})
message(STATUS ${Boost_LIBRARY_DIRS})

# Putting together the main executable & building relevant libraries:
add_subdirectory(src)

# Testing individual libraries:
add_subdirectory(tests)