cmake_minimum_required(VERSION 3.15.4)
project(PointSetsFitting)

find_package(Eigen3 REQUIRED)

add_library(PointSetsFitting STATIC psf.cpp psf.h)
target_include_directories(PointSetsFitting INTERFACE .)
target_link_libraries(PointSetsFitting Eigen3::Eigen )
target_compile_features(PointSetsFitting PUBLIC cxx_std_17)


add_executable(psfcmd app_psf_cmd.cpp)
target_link_libraries(psfcmd PointSetsFitting )
target_compile_features(psfcmd PUBLIC cxx_std_17)


# Test the point sets fitting library with gtest
option(TESTPSFLIB "Test library" OFF)
IF(${TESTPSFLIB})
    MESSAGE(STATUS "Test library activated")

    find_package(GTest REQUIRED)
    include_directories(${GTEST_INCLUDE_DIRS})
    add_executable( runPsfLibTests test/main.cpp test/test_psf.cpp)
    target_link_libraries( runPsfLibTests ${GTEST_LIBRARIES} PointSetsFitting pthread)
ENDIF()
