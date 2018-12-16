import unittest
import numpy as np
import point_sets_fitting as psf


class PsfTester(unittest.TestCase):

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


    def test_fitting_translation(self):

        setA = np.mat('0 0 0 1; 1 0 0 1; 0 1 0 1; 0 0 2 1').transpose()
        expectedTransformation = np.asmatrix(np.eye(4, 4))
        expectedTransformation[0,3] = 10
        expectedTransformation[1, 3] = 2
        setB = expectedTransformation * setA

        expectedError = 0;

        transformation, err = psf.pointSetFitting(setA[0:3,:], setB[0:3,:])

        np.testing.assert_array_almost_equal(expectedTransformation, transformation, decimal=5)
        self.assertAlmostEqual(expectedError,err,delta = 0.0001)


if __name__ == '__main__':
    unittest.main()
