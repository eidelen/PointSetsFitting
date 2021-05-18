#include "psf.h"

#include <exception>
#include <vector>
#include <numeric>
#include <iostream>

std::pair<Eigen::MatrixXd, double> pointSetsFitting(const Eigen::MatrixXd& setA, const Eigen::MatrixXd& setB)
{
    // check input: same number points in each set, at least 3 points
    if( setA.cols() != setB.cols() || setA.cols() < 3 || setA.rows() < 3 || setB.rows() < 3 )
        throw std::invalid_argument("Invalid input dimensions");

    Eigen::MatrixXd setAVal = validateMatrixOfPoints(setA);
    Eigen::MatrixXd setBVal = validateMatrixOfPoints(setB);

    auto[setACentered, transA] = centerPoints(setAVal);
    auto[setBCentered, transB] = centerPoints(setBVal);

    Eigen::MatrixXd lsqRotation = computeLsqRotation(setACentered, setBCentered);

    // Assemble transformation
    Eigen::MatrixXd rigidTransformation = Eigen::MatrixXd::Identity(4,4);
    rigidTransformation.block(0,0,3,3) = lsqRotation;
    rigidTransformation.block(0,3,3,1) = transB.topRows(3) - (lsqRotation * transA.topRows(3));

    return {rigidTransformation, computeFittingError(setAVal, setBVal, rigidTransformation)};
}

Eigen::MatrixXd computeLsqRotation(const Eigen::MatrixXd& setCenteredA, const Eigen::MatrixXd& setCenteredB)
{
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(3, 3);
    for(size_t i = 0; i < setCenteredA.cols(); i++)
    {
        h = h + (setCenteredA.block(0,i,3,1) * setCenteredB.block(0,i,3,1).transpose());
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(h, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd lsqRotation = svd.matrixV() * svd.matrixU().transpose();

    // fix rotation if necessary -> reflections
    if( lsqRotation.determinant() < 0 )
    {
        Eigen::MatrixXd correctedV = svd.matrixV();
        correctedV.col(2) = correctedV.col(2) * (-1.0);
        lsqRotation = correctedV * svd.matrixU().transpose();
    }

    return lsqRotation;
}

Eigen::MatrixXd vectorOfPositions2EigenMatrix(const std::vector<std::tuple<double,double,double>>& input )
{
    Eigen::MatrixXd ret(4, input.size());
    for(size_t i = 0; i < input.size(); i++)
    {
        auto [x, y, z] = input[i];
        ret.col(i) << x, y, z, 1.0;
    }

    return ret;
}

Eigen::MatrixXd computeCenterOfPoints( const Eigen::MatrixXd& input )
{
    return input.rowwise().sum() * 1.0 / input.cols();
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> centerPoints(const Eigen::MatrixXd &input)
{
    Eigen::MatrixXd centerPos = computeCenterOfPoints(input);
    centerPos(3,0) = 1.0;
    Eigen::MatrixXd toCenterTransform = Eigen::MatrixXd::Identity(4, 4);
    toCenterTransform.block(0,3,3,1) = (-centerPos).topRows(3);
    return {toCenterTransform * input, centerPos};
}

Eigen::MatrixXd validateMatrixOfPoints(const Eigen::MatrixXd& input)
{
    Eigen::MatrixXd validated(4, input.cols());
    validated.topRows(3) = input.topRows(3);
    validated.row(3).fill(1.0);
    return validated;
}

double computeFittingError(const Eigen::MatrixXd& setA, const Eigen::MatrixXd& setB, const Eigen::MatrixXd& transformation)
{
    Eigen::MatrixXd diffSetB = ((transformation * setA) - setB).topRows(3);
    return diffSetB.colwise().norm().mean();
}

std::tuple<Eigen::MatrixXd, double, std::vector<size_t>> pointSetsCorrespondence(const Eigen::MatrixXd& setA, const Eigen::MatrixXd& setB)
{
    size_t nPoints = setA.cols();

    std::vector<size_t> corrsp(nPoints);
    std::iota(corrsp.begin(), corrsp.end(), 0);

    std::tuple<Eigen::MatrixXd, double, std::vector<size_t>> bestSolution = {Eigen::MatrixXd::Identity(4,4),
                                                                             std::numeric_limits<double>::max(), {}};

    // permute over all correspondence permutations of set B and
    // choose the one with lowest error.
    do
    {
        // assemble set B
        std::vector<size_t> newCorrsp;
        Eigen::MatrixXd newSetB(3, nPoints);
        for(size_t q = 0; q < nPoints; q++)
        {
            newCorrsp.push_back(corrsp[q]);
            newSetB.col(q) = setB.block(0, corrsp[q], 3, 1);
        }

        auto[trans, error] = pointSetsFitting(setA, newSetB);

        if( error < std::get<1>(bestSolution) )
            bestSolution = {trans, error, newCorrsp};
    }
    while ( std::next_permutation(corrsp.begin(), corrsp.end()) );

    return bestSolution;
}
