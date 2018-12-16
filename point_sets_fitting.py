import numpy as np


def pointSetFitting( setA, setB ):
    return


def getPointSetCenter(aPointSet):
    assert isinstance(aPointSet, np.matrix)

    return aPointSet.sum(axis=1) * 1.0/aPointSet.shape[1]
