#include <gtest/gtest.h>
#include <tuple>
#include "psf.h"

TEST(Points, PosVec2Mat)
{
    std::vector<std::tuple<double,double,double>> input = {{0.0, 1.0, 2.0}, {3.0, 4.0, 5.0}};
    Eigen::MatrixXd output(4, 2);
    output << 0.0, 3.0, 1.0, 4.0, 2.0, 5.0, 1.0, 1.0;

    auto toEigen = vectorOfPositions2EigenMatrix(input);

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

TEST(Points, MoveCenter)
{
    Eigen::MatrixXd input(4, 100);
    input.fill(3.14);
    input.row(1).fill(6.0);
    input.row(3).fill(1.0);

    Eigen::MatrixXd outputCenterPoints(4, 100);
    outputCenterPoints.fill(0.0);
    outputCenterPoints.row(3).fill(1.0);

    Eigen::MatrixXd outputTranslation(4, 1);
    outputTranslation <<  -3.14, -6.0, -3.14, 1.0;

    auto [resCentered, resTranslation] = centerPoints(input);

    ASSERT_TRUE(outputCenterPoints.isApprox(resCentered));
    ASSERT_TRUE(outputTranslation.isApprox(resTranslation));
}