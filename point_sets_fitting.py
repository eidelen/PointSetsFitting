"""
This is a python implementation of the algorithm "Least-Squares Fitting of
two 3d Point Sets". This method can be used to compute the rigid transformation
between two sets of corresponding 3D vectors.
"""

import numpy as np

def point_sets_fitting(set_a, set_b):
    """
    Computes the rigid transformation between two sets of
    corresponding points.
    :param set_a: Point set A
    :param set_b: Point set B
    :return: (Rigid transformation, fitting error)
    """

    # check and transform parameters

    # list of points -> [ax, ay, az], [bx, by, bz], ...
    if isinstance(set_a, list):
        set_a = np.asmatrix(set_a).transpose()
    if isinstance(set_b, list):
        set_b = np.asmatrix(set_b).transpose()

    # matrix or numpy array
    set_a = np.asmatrix(set_a)
    set_b = np.asmatrix(set_b)

    n_points = set_a.shape[1]

    # check input data
    if set_a.shape[0] < 3: # data dimension
        raise Exception('Wrong vector dimension')
    if n_points < 3:
        raise Exception('Too few point correspondences')
    if set_a.shape[1] != set_b.shape[1]:
        raise Exception('Mismatching point set sizes')


    # move point sets to center
    centered_a, translation_a = move_point_set_to_center(set_a)
    centered_b, translation_b = move_point_set_to_center(set_b)

    # compute rotation
    h_3x3 = np.asmatrix(np.zeros((3, 3)))
    for i in range(n_points):
        h_3x3 = h_3x3 + centered_a[:, i] * centered_b[:, i].transpose()

    u_orth, _, vh_orth = np.linalg.svd(h_3x3)

    lsq_rotation = vh_orth.transpose() * u_orth.transpose()

    # fix rotation if necessary -> reflections
    det = np.linalg.det(lsq_rotation)
    if  det < 0:
        vh_orth[2, :] = vh_orth[2, :] * (-1)
        lsq_rotation = vh_orth.transpose() * u_orth.transpose()
        det = np.linalg.det(lsq_rotation)

    # if rotation matrix determinant is different from +1, there is a problem with this very matrix.
    assert __isclose(det, 1.0)

    # compute translation
    trans = translation_b - (lsq_rotation * translation_a)

    # compose a 4x4 matrix of rotation and translation
    rigid_transformation = np.asmatrix(np.eye(4, 4))
    rigid_transformation[0:3, 0:3] = lsq_rotation
    rigid_transformation[0:3, 3] = trans

    return rigid_transformation, compute_fitting_error(set_a, set_b, rigid_transformation)


# from https://stackoverflow.com/questions/5595425/what-is-the-best-way-to-compare-floats-for-almost-equality-in-python
def __isclose(a_val, b_val, rel_tol=1e-09, abs_tol=0.0):
    return abs(a_val - b_val) <= max(rel_tol * max(abs(a_val), abs(b_val)), abs_tol)


def compute_point_set_center(point_set):
    """
    Computes the center of a point set.
    :param point_set: Point set
    :return: Center position
    """
    assert isinstance(point_set, np.matrix)

    return np.asmatrix(point_set.sum(axis=1) * 1.0 / point_set.shape[1])


def move_point_set_to_center(point_set):
    """
    Move point set to its center.
    :param point_set: Point set
    :return: Centered point set.
    """
    assert isinstance(point_set, np.matrix)

    center = compute_point_set_center(point_set)
    centered_set = np.asmatrix(np.zeros(point_set.shape))

    for i in range(point_set.shape[1]):
        centered_set[:, i] = point_set[:, i] - center

    return centered_set, center


def compute_fitting_error(set_a, set_b, transformation):
    """
    Computes the fitting error.
    :param set_a: Point set a
    :param set_b: Point set b
    :param transformation: Rigid transformation matrix
    :return: Transformation error
    """
    assert isinstance(set_a, np.matrix)
    assert isinstance(set_b, np.matrix)
    assert isinstance(transformation, np.matrix)

    set_a = to_homogeneous_repr(set_a)
    set_b = to_homogeneous_repr(set_b)

    diff_set_b = transformation * set_a - set_b

    nbr_of_points = diff_set_b.shape[1]
    accum_norm = 0
    for i in range(nbr_of_points):
        accum_norm += np.linalg.norm(diff_set_b[:, i])

    return accum_norm / nbr_of_points


def to_homogeneous_repr(points):
    """Adds the homogeneous 4th line"""
    assert isinstance(points, np.matrix)

    pnt_h = np.asmatrix(np.ones((points.shape[0]+1, points.shape[1])))
    pnt_h[0:3, :] = points

    return pnt_h
