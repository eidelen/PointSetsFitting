#include "psf.h"

#include <exception>
#include <vector>

std::pair<Eigen::MatrixXd, double> pointSetsFitting(const Eigen::MatrixXd& setA, const Eigen::MatrixXd& setB)
{
    // check input: same number points in each set, at least 3 points
    if( setA.cols() != setB.cols() || setA.cols() < 3 || setA.rows() < 3 || setB.rows() < 3 )
        throw "Invalid input";

    Eigen::MatrixXd setAVal = validateMatrixOfPoints(setA);
    Eigen::MatrixXd setBVal = validateMatrixOfPoints(setB);

    auto[setACentered, transA] = centerPoints(setAVal);
    auto[setBCentered, transB] = centerPoints(setBVal);

    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(3, 3);
    for(size_t i = 0; i < setACentered.cols(); i++)
    {
        h = h + (setACentered.block(0,i,3,1) * setBCentered.block(0,i,3,1).transpose());
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(h, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd lsqRotation = svd.matrixV() * svd.matrixU().transpose();

    // fix rotation if necessary -> reflections
    if( lsqRotation.determinant() < 0 )
    {
        Eigen::MatrixXd correctedV = svd.matrixV();
        correctedV.row(2) = correctedV.row(2) * (-1.0);
        lsqRotation = correctedV * svd.matrixU().transpose();
    }

    // Assemble transformation
    Eigen::MatrixXd rigidTransformation = Eigen::MatrixXd::Identity(4,4);
    rigidTransformation.block(0,0,3,3) = lsqRotation;
    rigidTransformation.block(0,3,3,1) = transB.topRows(3) - (lsqRotation * transA.topRows(3));

    return {rigidTransformation, computeFittingError(setAVal, setBVal, rigidTransformation)};
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
