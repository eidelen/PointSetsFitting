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
    outputTranslation <<  3.14, 6.0, 3.14, 1.0;

    auto [resCentered, resTranslation] = centerPoints(input);

    ASSERT_TRUE(outputTranslation.isApprox(resTranslation));
    ASSERT_TRUE(outputCenterPoints.isApprox(resCentered));

}

TEST(Points, Validate)
{
    Eigen::MatrixXd inp0(3,2);
    inp0 << 1, 2,
            3, 4,
            5, 6;

    Eigen::MatrixXd out0(4,2);
    out0 << 1, 2,
            3, 4,
            5, 6,
            1, 1;

    Eigen::MatrixXd res = validateMatrixOfPoints(inp0);
    ASSERT_TRUE(out0.isApprox(res));

    Eigen::MatrixXd inp1(4,2);
    inp1 << 1, 2,
            3, 4,
            5, 6,
            1, 1;

    Eigen::MatrixXd res1 = validateMatrixOfPoints(inp1);
    ASSERT_TRUE(inp1.isApprox(res1));
}

TEST(Fitting, IvalidInput)
{
    Eigen::MatrixXd inpA0(2,3);
    ASSERT_ANY_THROW(pointSetsFitting(inpA0, inpA0));

    Eigen::MatrixXd inpA1(3,3);
    Eigen::MatrixXd inpB1(3,4);
    ASSERT_ANY_THROW(pointSetsFitting(inpA1, inpB1));

    Eigen::MatrixXd inpA2(3,2);
    ASSERT_ANY_THROW(pointSetsFitting(inpA2, inpA2));
}

TEST(Fitting, Translation)
{
    Eigen::MatrixXd setA(3,4);
    setA << 0, 1, 0, 0,
           0, 0, 1, 0,
           0, 0, 0, 2;

    Eigen::MatrixXd setB(3,4);
    setB << 10, 11, 10, 10,
            2, 2, 3, 2,
            5, 5, 5, 7;

    Eigen::MatrixXd t = Eigen::MatrixXd::Identity(4,4);
    t.block(0,3,3,1) << 10, 2, 5;

    auto[transformation, error] = pointSetsFitting(setA, setB);
    ASSERT_TRUE(t.isApprox(transformation));
    ASSERT_NEAR(error, 0.0, 0.0001);
}
