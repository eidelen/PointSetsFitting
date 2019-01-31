import unittest
import numpy as np
from transforms3d.euler import euler2mat
from transforms3d.affines import compose
import point_sets_fitting as psf


class PsfTester(unittest.TestCase):

    def test_toHomogeneous(self):

        setIn = np.mat('0 0 0; 1 0 0; 0 1 0; 0 0 2').transpose()
        outExpected = np.mat('0 0 0 1; 1 0 0 1; 0 1 0 1; 0 0 2 1').transpose()

        out = psf.toHomogeneous(setIn)

        np.testing.assert_array_almost_equal(outExpected, out, decimal=5)


    def test_center(self):

        testInput = np.mat('0  0  0; 0  4  0 ; 8  0  0; 0  0  12').transpose()
        centerIs = np.mat('2; 1; 3')
        centerComputed = psf.getPointSetCenter(testInput)
        self.assertTrue(np.array_equal(centerIs,centerComputed))


    def test_move_to_center(self):

        testInput = np.mat('0  0  0; 0  4  0 ; 8  0  0; 0  0  12').transpose()
        movedAre = np.mat('-2  -1  -3; -2  3  -3 ; 6  -1  -3; -2  -1  9').transpose()
        centerIs = np.mat('2; 1; 3')

        movedComputed, centerComputed = psf.movePointSetToCenter(testInput)

        self.assertTrue(np.array_equal(movedAre,movedComputed))
        self.assertTrue(np.array_equal(centerIs, centerComputed))


    def test_fitting_identity(self):

        setA = np.mat('0 0 0; 1 0 0; 0 1 0; 0 0 2').transpose()
        setB = np.mat('0 0 0; 1 0 0; 0 1 0; 0 0 2').transpose()

        expectedError = 0;
        expectedTransformation = np.asmatrix(np.eye(4,4))

        transformation, err = psf.pointSetFitting(setA, setB)

        np.testing.assert_array_almost_equal(expectedTransformation, transformation, decimal=5)
        self.assertAlmostEqual(expectedError,err,delta = 0.0001)


    def test_fitting_identity_arrinput(self):

        setA = np.array([[0, 0, 0] ,[1, 0, 0] ,[0, 1, 0] ,[0, 0 ,2]]).transpose()
        setB = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 2]]).transpose()

        expectedError = 0;
        expectedTransformation = np.asmatrix(np.eye(4, 4))

        transformation, err = psf.pointSetFitting(setA, setB)

        np.testing.assert_array_almost_equal(expectedTransformation, transformation, decimal=5)
        self.assertAlmostEqual(expectedError, err, delta=0.0001)


    def test_fitting_identity_list(self):

        listA = [[0, 0, 0] ,[1, 0, 0] ,[0, 1, 0] ,[0, 0 ,2]]
        listB = [[0, 0, 0] ,[1, 0, 0] ,[0, 1, 0] ,[0, 0 ,2]]

        expectedError = 0;
        expectedTransformation = np.asmatrix(np.eye(4, 4))

        transformation, err = psf.pointSetFitting(listA, listB)

        np.testing.assert_array_almost_equal(expectedTransformation, transformation, decimal=5)
        self.assertAlmostEqual(expectedError, err, delta=0.0001)


    def test_fitting_differen_set_lengths(self):
        with self.assertRaises(Exception):
            listA = [[0, 0, 0] ,[1, 0, 0] ,[0, 1, 0] ,[0, 0 ,2]]
            listB = [[0, 0, 0] ,[1, 0, 0] ,[0, 1, 0]]
            psf.pointSetFitting(listA, listB)


    def test_fitting_too_few_points(self):
        with self.assertRaises(Exception):
            listA = [[0, 0, 0], [1, 0, 0]]
            listB = [[0, 0, 0], [1, 0, 0]]

            psf.pointSetFitting(listA, listB)


    def test_fitting_vector_dimension(self):
        with self.assertRaises(Exception):
            listA = [[0, 0], [1, 0], [0, 1]]
            listB = [[0, 0], [1, 0], [0, 1]]

            psf.pointSetFitting(listA, listB)


    def test_fitting_translation(self):

        setA = np.mat('0 0 0; 1 0 0; 0 1 0; 0 0 2').transpose()
        expectedTransformation = compose([10 ,2 ,5], np.eye(3,3), np.ones(3))
        setB = expectedTransformation * psf.toHomogeneous(setA)

        expectedError = 0;

        transformation, err = psf.pointSetFitting(setA, setB[0:3,:])

        np.testing.assert_array_almost_equal(expectedTransformation, transformation, decimal=5)
        self.assertAlmostEqual(expectedError,err,delta = 0.0001)


    def test_fitting_rotation(self):

        setA = np.mat('0 0 0; 1 0 0; 0 1 0; 0 0 2').transpose()

        expectedTransformation = compose(np.zeros(3),euler2mat(0.1, 0.2, 0.3),np.ones(3))
        setB = expectedTransformation * psf.toHomogeneous(setA)

        expectedError = 0;

        transformation, err = psf.pointSetFitting(setA, setB[0:3, :])

        np.testing.assert_array_almost_equal(expectedTransformation, transformation, decimal=5)
        self.assertAlmostEqual(expectedError, err, delta=0.0001)


    def test_fitting_noisefree_3P(self):

        for k in xrange(1000):
            setA = np.asmatrix(np.random.rand(3,3))
            trans = np.random.rand(1, 3)
            expTr = compose(trans[0, :], euler2mat(np.random.rand(), np.random.rand(), np.random.rand()), np.ones(3))
            setB = expTr * psf.toHomogeneous(setA)

            expError = 0;

            transformation, err = psf.pointSetFitting(setA, setB[0:3, :])

            np.testing.assert_array_almost_equal(expTr, transformation, decimal=5)
            self.assertAlmostEqual(expError, err, delta=0.0001)


    def test_fitting_noisefree_10P(self):

        for k in xrange(1000):
            setA = np.asmatrix(np.random.rand(3,10))
            trans = np.random.rand(1, 3)
            expTr = compose(trans[0, :], euler2mat(np.random.rand(), np.random.rand(), np.random.rand()), np.ones(3))
            setB = expTr * psf.toHomogeneous(setA)

            expError = 0;

            transformation, err = psf.pointSetFitting(setA, setB[0:3, :])

            np.testing.assert_array_almost_equal(expTr, transformation, decimal=5)
            self.assertAlmostEqual(expError, err, delta=0.0001)


    def test_point_set_error(self):
        setA = np.mat('0 0 0; 1 0 0; 0 1 0; 0 0 2').transpose()
        setB = np.mat('0 0 0; 1 0 0; 0 1 0; 0 0 -2').transpose()
        tf = np.asmatrix( np.eye(4,4) )
        expectedError = 1;

        err = psf.fittingError(setA, setB, tf)

        self.assertAlmostEqual(expectedError, err, delta=0.0001)


if __name__ == '__main__':
    unittest.main()
