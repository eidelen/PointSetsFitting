#include "psf.h"

#include <iostream>
#include <vector>

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
    Eigen::MatrixXd centerTranslation = -computeCenterOfPoints(input);
    centerTranslation(3,0) = 1.0;
    Eigen::MatrixXd toCenterTransform = Eigen::MatrixXd::Identity(4, 4);
    toCenterTransform.col(3) = centerTranslation;
    return {toCenterTransform * input, centerTranslation};
}

