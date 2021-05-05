cmake_minimum_required(VERSION 3.19)
project(PointSetsFitting)

add_library(PointSetsFitting STATIC psf.cpp psf.h)
target_include_directories(PointSetsFitting INTERFACE .)
target_compile_features(PointSetsFitting PRIVATE cxx_std_17)

# Test the point sets fitting library with gtest
option(TESTPSFLIB "Test library" OFF)
IF(${TESTPSFLIB})
    MESSAGE(STATUS "Test library activated")
    find_package(GTest REQUIRED)
    include_directories(${GTEST_INCLUDE_DIRS})
    add_executable( runPsfLibTests test/main.cpp test/test_psf.cpp)
    target_link_libraries( runPsfLibTests ${GTEST_LIBRARIES} PointSetsFitting )
ENDIF()