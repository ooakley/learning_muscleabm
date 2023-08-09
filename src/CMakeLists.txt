add_library(agents agents.cpp agents.h)

add_library(ecm ecm.cpp ecm.h)

add_library(world world.cpp world.h)

add_executable(main main.cpp)

target_link_libraries(main
    PRIVATE
    agents
    ecm
    world
    ${Boost_LIBRARIES}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
)

target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})