#include "psf.h"

#include <iostream>
#include <vector>

Eigen::MatrixXd positionVector2EigenMatrix( const std::vector<std::tuple<double,double,double>>& input )
{
    Eigen::MatrixXd ret(4, input.size());
    for(size_t i = 0; i < input.size(); i++)
    {
        auto [x, y, z] = input[i];
        ret(0, i) = x;
        ret(1, i) = y;
        ret(2, i) = z;
        ret(3, i) = 1.0;
    }

    return ret;
}

Eigen::MatrixXd computeCenterOfPoints( const Eigen::MatrixXd& input )
{
    return input.rowwise().sum() * 1.0 / input.cols();
}

