import numpy as np


def pointSetFitting( setA, setB ):
    return


def getPointSetCenter(aPointSet):
    assert isinstance(aPointSet, np.matrix)

    return aPointSet.sum(axis=1) * 1.0/aPointSet.shape[1]


def movePointSetToCenter(aPointSet):
    assert isinstance(aPointSet, np.matrix)

    center = getPointSetCenter(aPointSet)
    for i in xrange( aPointSet.shape[1] ):
        aPointSet[:,i] = aPointSet[:,i] - center

    return aPointSet

