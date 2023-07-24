add_library(agents agents.cpp)

add_executable(main main.cpp)

target_link_libraries(main PRIVATE agents)
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})