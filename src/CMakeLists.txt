# Building nodes of dependency tree:
add_library(agents agents.cpp agents.h)
add_library(ecm ecm.cpp ecm.h)

# Building world, and link to agents and ecm to ensure transitivity:
add_library(world world.cpp world.h)
target_link_libraries(world
    PRIVATE
    agents
    ecm
)

# Building main executable:
add_executable(main main.cpp)
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(main
    PRIVATE
    world
    agents
    ecm
    ${Boost_LIBRARIES}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    Boost::program_options
)

# Linking to OpenMP (for multiprocessing) if available:
if(OpenMP_CXX_FOUND)
    target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)
endif()