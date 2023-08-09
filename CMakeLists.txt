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

# Ensuring Boost is present:
find_package(Boost COMPONENTS regex system filesystem REQUIRED)
message(STATUS ${Boost_INCLUDE_DIRS})
set(Boost_INCLUDE_DIRS /usr/local/Cellar/boost/1.82.0_1/include)
message(STATUS ${Boost_INCLUDE_DIRS})

include_directories(${Boost_INCLUDE_DIRS})

# Putting together the main executable & building relevant libraries:
add_subdirectory(src)

# Testing individual libraries:
add_subdirectory(tests)