#ifndef CPP_PSF_H
#define CPP_PSF_H

#include <Eigen/Dense>
#include <tuple>

/**
 * Computes the rigid transformation between two sets
 * of corresponding points.
 * @param setA Point set A (4xn or 3xn matrix)
 * @param setB Point set B (4xn or 3xn matrix)
 * @return Pair of rigid transformation and fitting error
 */
std::pair<Eigen::MatrixXd, double> pointSetsFitting(const Eigen::MatrixXd& setA, const Eigen::MatrixXd& setB);

/**
 * Finds the point-to-point correspondences between two
 * point sets which results in the lowest rigid
 * transformation error.
 * @param setA Point set A (4xn or 3xn matrix)
 * @param setB Point set B (4xn or 3xn matrix)
 * @return Rigid transformation, transformation error, correspondences
 */
std::tuple<Eigen::MatrixXd, double, std::vector<size_t>> pointSetsCorrespondence(const Eigen::MatrixXd& setA,
                                                                                                    const Eigen::MatrixXd& setB);

/**
 * Compute the rotation matrix between two centered point sets.
 * @param setA Centered point set A
 * @param setB Centered point set B
 * @return Rotation matrix R, such that A = R * B
 */
Eigen::MatrixXd computeLsqRotation(const Eigen::MatrixXd& setA, const Eigen::MatrixXd& setB);

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

/**
 * Returns a 4xn homogenous matrix of points
 * @param input Points
 * @return 4xn homogenous matrix of points
 */
Eigen::MatrixXd validateMatrixOfPoints(const Eigen::MatrixXd& input);

/**
 * Computes the fitting error.
 * @param setA Point set A as 4xn matrix
 * @param setB Point set B as 4xn matrix
 * @param transformation 4x4 rigid transformation
 * @return Average fitting error
 */
double computeFittingError(const Eigen::MatrixXd& setA, const Eigen::MatrixXd& setB, const Eigen::MatrixXd& transformation);

#endif //CPP_PSF_H
