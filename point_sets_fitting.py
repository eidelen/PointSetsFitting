import numpy as np


def pointSetFitting( setA, setB ):
    assert isinstance(setA, np.matrix)
    assert isinstance(setB, np.matrix)

    nbrPoints = setA.shape[1]


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
    if np.linalg.det(rotation) < 0 :
        print("Reflection occured")
        vhMod = np.asmatrix(np.zeros(vh.shape))
        vhMod[:,3] = vh[:,3] * (-1)
        rotation = vhMod.transpose() * u.transpose;


    # compute translation
    trans = transB - (rotation * transA)


    # compose a 4x4 matrix of rotation and translation
    rigidTransformation = np.asmatrix(np.eye(4,4))
    rigidTransformation[0:3, 0:3] = rotation
    rigidTransformation[0:3,3] = trans

    #todo: compute transformation error
    return rigidTransformation, 0



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

