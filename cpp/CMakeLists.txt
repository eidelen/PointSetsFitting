cmake_minimum_required(VERSION 3.13.0)
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
    
    # Download and build gtest
    include(FetchContent)
    FetchContent_Declare(
        googletest
        # Specify the commit you depend on and update it regularly.
        URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable( runPsfLibTests test/main.cpp test/test_psf.cpp)
    target_link_libraries( runPsfLibTests PointSetsFitting gtest_main)
ENDIF()
