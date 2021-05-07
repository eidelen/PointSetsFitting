#ifndef CPP_PSF_H
#define CPP_PSF_H


#include <Eigen/Dense>
#include <tuple>

/**
 * Creates a 4xn Eigen matrix from an input vector of 3d-positions.
 * @param input Vector of 3d-positions
 * @return 4xn Eigen matrix
 */
Eigen::MatrixXd vectorOfPositions2EigenMatrix(const std::vector<std::tuple<double,double,double>>& input );

/**
 * Computes the center of multiple points.
 * @param input Point in form of a 4xn matrix.
 * @return Center in form a 4 x 1 matrix / vector
 */
Eigen::MatrixXd computeCenterOfPoints( const Eigen::MatrixXd& input );

/**
 * Translates points such that their center is at the coordinate
 * system's origin.
 * @param input 4xn matrix of input points
 * @return 4xn matrix of centered points and 4x1 translation vector
 */
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> centerPoints( const Eigen::MatrixXd& input );

#endif //CPP_PSF_H
