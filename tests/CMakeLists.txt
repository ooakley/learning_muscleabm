add_executable(
  main_test
  main.cpp
  agents_test.cpp
)
target_link_libraries(
  main_test
  GTest::gtest_main
  agents
)
target_include_directories(main_test PRIVATE ../src)

include(GoogleTest)
gtest_discover_tests(main_test)