"""
This is a python implementation of the algorithm "Least-Squares Fitting of
two 3d Point Sets". This method can be used to compute the rigid transformation
between two sets of corresponding 3D vectors.
"""

from typing import Tuple
import numpy as np

def point_sets_fitting(set_a: np.ndarray, set_b: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Computes the rigid transformation between two sets of
    corresponding points.
    :param set_a: Point set A
    :param set_b: Point set B
    :return: (Rigid transformation, fitting error)
    """

    # check input data
    if 4 < set_a.shape[0] < 3 or 4 < set_b.shape[0] < 3: # data dimension
        raise Exception('Wrong vector dimension')
    if set_a.shape[1] < 3:
        raise Exception('Too few point correspondences')
    if set_a.shape[1] != set_b.shape[1]:
        raise Exception('Mismatching point set sizes')

    # If homogeneous coordinate input, adapt
    if set_a.shape[0] == 4:
        set_a = set_a[:-1, :]
    if set_b.shape[0] == 4:
        set_b = set_b[:-1, :]


    # move point sets to center
    centered_a, translation_a = move_point_set_to_center(set_a)
    centered_b, translation_b = move_point_set_to_center(set_b)

    # compute rotation
    h_3x3 = np.zeros((3, 3))
    for pnt_a, pnt_b in zip(centered_a.transpose(), centered_b.transpose()):
        pnt_a = pnt_a[np.newaxis].transpose() # create 2d from 1d representation
        pnt_b = (pnt_b[np.newaxis])
        h_3x3 = h_3x3 + (pnt_a @ pnt_b)

    u_orth, _, vh_orth = np.linalg.svd(h_3x3)

    lsq_rotation = vh_orth.transpose() @ u_orth.transpose()

    # fix rotation if necessary -> reflections
    det = np.linalg.det(lsq_rotation)
    if  det < 0:
        vh_orth[2, :] = vh_orth[2, :] * (-1)
        lsq_rotation = vh_orth.transpose() @ u_orth.transpose()
        det = np.linalg.det(lsq_rotation)

    # if rotation matrix determinant is different from +1, there
    # is a problem with this very solution.
    if not __isclose(det, 1.0):
        raise Exception('Invalid rotation matrix determinant: %.6f' % (det))

    # compose a 4x4 matrix of rotation and translation
    rigid_transformation = np.eye(4, 4)
    # set rotation matrix
    rigid_transformation[0:3, 0:3] = lsq_rotation
    # compute and set translation vector
    rigid_transformation[0:3, 3] = translation_b - (lsq_rotation @ translation_a)

    return rigid_transformation, compute_fitting_error(set_a, set_b, rigid_transformation)


# from https://stackoverflow.com/questions/5595425/what-is-the-best-way-to-compare-floats-for-almost-equality-in-python
def __isclose(a_val: float, b_val: float, rel_tol=1e-09, abs_tol=0.0) -> bool:
    return abs(a_val - b_val) <= max(rel_tol * max(abs(a_val), abs(b_val)), abs_tol)


def compute_point_set_center(point_set: np.ndarray) -> np.ndarray:
    """
    Computes the center of a point set.
    :param point_set: Point set
    :return: Center position
    """
    center = point_set.sum(axis=1) * (1.0 / point_set.shape[1])
    return center


def move_point_set_to_center(point_set: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Move point set to its center.
    :param point_set: Point set
    :return: Centered point set.
    """

    center = compute_point_set_center(point_set)

    # subtract center position from each position in point_set
    centered_set = point_set - center[np.newaxis].transpose()

    return centered_set, center


def compute_fitting_error(set_a: np.ndarray, set_b: np.ndarray,
                          transformation: np.ndarray) -> float:
    """
    Computes the fitting error.
    :param set_a: Point set a
    :param set_b: Point set b
    :param transformation: Rigid transformation matrix
    :return: Transformation error
    """

    set_a = to_homogeneous_repr(set_a)
    set_b = to_homogeneous_repr(set_b)

    diff_set_b = transformation @ set_a - set_b

    accum_norm = 0
    for dif_vec in diff_set_b.transpose():
        accum_norm += np.linalg.norm(dif_vec)

    return accum_norm / diff_set_b.shape[1]


def to_homogeneous_repr(points: np.ndarray) -> np.ndarray:
    """Adds the homogeneous 4th line"""
    pnt_h = np.ones((points.shape[0]+1, points.shape[1]))
    pnt_h[0:3, :] = points
    return pnt_h
