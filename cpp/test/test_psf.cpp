#include <gtest/gtest.h>
#include <tuple>
#include "psf.h"

TEST(Points, PosVec2Mat)
{
    std::vector<std::tuple<double,double,double>> input = {{0.0, 1.0, 2.0}, {3.0, 4.0, 5.0}};
    Eigen::MatrixXd output(4, 2);
    output << 0.0, 3.0, 1.0, 4.0, 2.0, 5.0, 1.0, 1.0;

    auto toEigen = positionVector2EigenMatrix(input);

    std::cout << output << std::endl << std::endl << toEigen << std::endl;
    ASSERT_TRUE(output.isApprox(toEigen));
}

TEST(Points, Center)
{
    Eigen::MatrixXd input(4, 2);
    input <<  1.0,  -1.0,
              2.0,  -2.0,
              0.0,   2.0,
              1.0,   1.0;

    Eigen::MatrixXd output(4, 1);
    output <<  0.0, 0.0, 1.0, 1.0;

    auto center = computeCenterOfPoints(input);

    ASSERT_TRUE(output.isApprox(center));
}

TEST(Points, CenterTrivial)
{
    Eigen::MatrixXd input(4, 100);
    input.fill(3.14);
    input.row(1).fill(6.0);
    input.row(3).fill(1.0);

    Eigen::MatrixXd output(4, 1);
    output <<  3.14, 6.0, 3.14, 1.0;

    auto center = computeCenterOfPoints(input);

    ASSERT_TRUE(output.isApprox(center));
}