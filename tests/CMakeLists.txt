# Behavioural tests of the model, one file per behaviour:
add_executable(
  main_test
  collision_test.cpp
  matrix_test.cpp
  movement_test.cpp
  stick_slip_test.cpp
)
target_link_libraries(
  main_test
  GTest::gtest_main
  world
  agents
  ecm
  collision
)
target_include_directories(main_test PRIVATE ../src)

include(GoogleTest)
gtest_discover_tests(main_test)
