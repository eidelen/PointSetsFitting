#include <gtest/gtest.h>
#include <tuple>
#include <Eigen/Geometry>
#include "psf.h"

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> generatePointSetsAndTransformation(size_t nbrPoints)
{
    // create random point set A
    Eigen::MatrixXd setA = Eigen::MatrixXd::Random(4,nbrPoints)*10.0;
    setA.row(3).fill(1.0);

    // create random transformation
    Eigen::MatrixXd translation = Eigen::MatrixXd::Random(3,1)*10.0;
    Eigen::MatrixXd rotation = Eigen::MatrixXd::Random(3,1);
    Eigen::Affine3d t;
    t = Eigen::Translation3d(translation);
    t.rotate(Eigen::AngleAxisd(rotation(0,0), Eigen::Vector3d::UnitZ())
             * Eigen::AngleAxisd(rotation(1,0), Eigen::Vector3d::UnitY())
             * Eigen::AngleAxisd(rotation(2,0), Eigen::Vector3d::UnitZ()));

    // error free transformation -> set B
    Eigen::MatrixXd setB = t.matrix() * setA;

    return {setA, setB, t.matrix()};
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> generatePointSetsAndTransformationWithNoise(size_t nbrPoints, double noise)
{
    auto[setA, setB, t] = generatePointSetsAndTransformation(nbrPoints);

    // introduce noise to set B
    Eigen::MatrixXd setBNoise = setB.topRows(3) + Eigen::MatrixXd::Random(3,nbrPoints) * noise;

    return {setA, setBNoise, t.matrix()};
}

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

TEST(Error, FittingError)
{
    Eigen::MatrixXd setA(4,4);
    setA << 0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 2,
            1, 1, 1, 1;

    Eigen::MatrixXd setB(4,4);
    setB << 0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, -2,
            1, 1, 1, 1;

    double fittingError = 1.0;
    double resFittingError = computeFittingError(setA, setB, Eigen::MatrixXd::Identity(4,4));
    ASSERT_NEAR(fittingError, resFittingError, 0.0001);
}

TEST(Fitting, RandomTransformations)
{
    for(size_t n : {3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 80, 160, 500})
    {
        std::cout << "Transformation for n = " << n << std::endl;
        for(size_t run = 0; run < 1000; run++)
        {
            auto[setA, setB, t] = generatePointSetsAndTransformation(n);
            auto[transformation, error] = pointSetsFitting(setA, setB);
            ASSERT_TRUE(t.matrix().isApprox(transformation));
            ASSERT_NEAR(error, 0.0, 0.0001);
            ASSERT_NEAR(t.matrix().determinant(), 1.0, 0.0001);
        }
    }
}

TEST(Fitting, RandomTransformationsNoise)
{
    for(size_t n : {3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 80, 160, 500})
    {
        std::cout << "Transformation with noise for n = " << n << std::endl;
        for(size_t run = 0; run < 1000; run++)
        {
            auto[setA, setB, t] = generatePointSetsAndTransformation(n);

            // introduce some noise to set B
            Eigen::MatrixXd setBNoise = setB.topRows(3) + Eigen::MatrixXd::Random(3,n)*0.01;

            auto[transformation, error] = pointSetsFitting(setA, setBNoise);

            // check rotation
            ASSERT_TRUE(t.matrix().block(0,0,3,3).isApprox(transformation.block(0,0,3,3), 0.1));
            // check translation
            ASSERT_TRUE(t.matrix().block(0,3,3,1).isApprox(transformation.block(0,3,3,1), 0.2));
            ASSERT_LT(0.0, error);
            ASSERT_NEAR(t.matrix().determinant(), 1.0, 0.0001);
        }
    }
}

TEST(Fitting, RandomTransformationsIncreaseNoise)
{
    double errorBefore = 0.0;
    std::vector<double> noiseLevels(20);
    std::generate(noiseLevels.begin(), noiseLevels.end(), [] () {
        static double ns = 0.0;
        return (ns += 0.2);
    });

    for(double noise : noiseLevels)
    {
        auto[setA, setB, t] = generatePointSetsAndTransformationWithNoise(500, noise);
        auto[transformation, error] = pointSetsFitting(setA, setB);

        std::cout << "noise " << noise << ": error " << error  << std::endl;

        ASSERT_LE(errorBefore, error);
        ASSERT_NEAR(t.matrix().determinant(), 1.0, 0.0001);
        errorBefore = error;
    }
}
