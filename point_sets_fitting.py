import numpy as np


def pointSetFitting( setA, setB ):


    # check and transform parameters

    # list of points -> [ax, ay, az], [bx, by, bz], ...
    if isinstance(setA, list):
        setA = np.asmatrix(setA).transpose();
    if isinstance(setB, list):
        setB = np.asmatrix(setB).transpose();

    # matrix or numpy array
    setA = np.asmatrix(setA);
    setB = np.asmatrix(setB);

    dataDim = setA.shape[0]
    nbrPoints = setA.shape[1]

    # check input data
    if dataDim < 3:
        raise Exception('Wrong vector dimension')
    if nbrPoints < 3:
        raise Exception('Too few point correspondences')
    if setA.shape[1] != setB.shape[1]:
        raise Exception('Mismatching point set sizes')


    # move point sets to center
    cSetA, transA = movePointSetToCenter(setA)
    cSetB, transB = movePointSetToCenter(setB)

    # compute rotation
    h = np.asmatrix(np.zeros((3,3)))
    for i in xrange(nbrPoints):
        pA = cSetA[:,i]
        pB = cSetB[:,i]

        h = h + pA * pB.transpose()

    u, s, vh = np.linalg.svd(h)

    rotation = vh.transpose() * u.transpose()

    # fix rotation if necessary -> reflections
    det = np.linalg.det(rotation)
    if  det < 0 :
        vh[2,:] = vh[2,:] * (-1)
        rotation = vh.transpose() * u.transpose();
        det = np.linalg.det(rotation)

    # if rotation matrix determinant is different from +1, there is a problem with this very matrix.
    assert isclose(det, 1.0)

    # compute translation
    trans = transB - (rotation * transA)

    # compose a 4x4 matrix of rotation and translation
    rigidTransformation = np.asmatrix(np.eye(4,4))
    rigidTransformation[0:3, 0:3] = rotation
    rigidTransformation[0:3,3] = trans

    return rigidTransformation, fittingError(setA, setB, rigidTransformation)


# from https://stackoverflow.com/questions/5595425/what-is-the-best-way-to-compare-floats-for-almost-equality-in-python
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def getPointSetCenter(aPointSet):
    assert isinstance(aPointSet, np.matrix)

    return np.asmatrix(aPointSet.sum(axis=1) * 1.0/aPointSet.shape[1])



def movePointSetToCenter(aPointSet):
    assert isinstance(aPointSet, np.matrix)

    nbrOfPoints = aPointSet.shape[1]
    center = getPointSetCenter(aPointSet)
    mvSet = np.asmatrix(np.zeros( aPointSet.shape ))

    for i in xrange(nbrOfPoints):
        mvSet[:,i] = aPointSet[:,i] - center

    return mvSet, center


def fittingError(setA, setB, transformation):
    assert isinstance(setA, np.matrix)
    assert isinstance(setB, np.matrix)
    assert isinstance(transformation, np.matrix)

    setAH = toHomogeneous(setA)
    setBH = toHomogeneous(setB)

    diffSetB = transformation * setAH - setBH

    nbrOfPoints = diffSetB.shape[1]
    accumNorm = 0
    for i in xrange(nbrOfPoints):
        accumNorm += np.linalg.norm(diffSetB[:, i])

    return accumNorm / nbrOfPoints


def toHomogeneous(points):
    assert isinstance(points, np.matrix)

    pntH = np.asmatrix(np.ones( (points.shape[0]+1, points.shape[1]) ))
    pntH[0:3,:] = points

    return pntH

