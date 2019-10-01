import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath)

import unittest
import numpy as np
from transforms3d.euler import euler2mat
from transforms3d.affines import compose
import point_sets_fitting as psf


class PsfTester(unittest.TestCase):

    def test_toHomogeneous(self):

        set_in = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 2]]).transpose()
        set_out_expected = np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 2, 1]]).transpose()
        set_out = psf.to_homogeneous_repr(set_in)
        np.testing.assert_array_almost_equal(set_out_expected, set_out, decimal=5)


    def test_center(self):

        test_input = np.array([[0, 0, 0], [0, 4, 0], [8, 0, 0], [0, 0, 12]]).transpose()
        center_is = np.array([2, 1, 3])
        center_computed = psf.compute_point_set_center(test_input)
        self.assertTrue(np.array_equal(center_is,center_computed))


    def test_move_to_center(self):

        test_input = np.array([[0, 0, 0], [0, 4, 0], [8, 0, 0], [0, 0, 12]]).transpose()
        moved_is = np.array([[-2, -1, -3], [-2, 3, -3], [6, -1, -3], [-2, -1, 9]]).transpose()
        center_is = np.array([2, 1, 3])

        moved_computed, center_computed = psf.move_point_set_to_center(test_input)

        self.assertTrue(np.array_equal(moved_is, moved_computed))
        self.assertTrue(np.array_equal(center_is, center_computed))


    def test_fitting_identity(self):

        set_a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 2]]).transpose()
        set_b = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 2]]).transpose()

        expected_error = 0;
        expected_transformation = np.eye(4,4)

        transformation, err = psf.point_sets_fitting(set_a, set_b)

        np.testing.assert_array_almost_equal(expected_transformation, transformation, decimal=5)
        self.assertAlmostEqual(expected_error, err, delta = 0.0001)


    def test_fitting_differen_set_lengths(self):
        with self.assertRaises(Exception):
            listA = [[0, 0, 0] ,[1, 0, 0] ,[0, 1, 0] ,[0, 0 ,2]]
            listB = [[0, 0, 0] ,[1, 0, 0] ,[0, 1, 0]]
            psf.point_sets_fitting(listA, listB)


    def test_fitting_too_few_points(self):
        with self.assertRaises(Exception):
            listA = [[0, 0, 0], [1, 0, 0]]
            listB = [[0, 0, 0], [1, 0, 0]]
            psf.point_sets_fitting(listA, listB)


    def test_fitting_vector_dimension(self):
        with self.assertRaises(Exception):
            listA = [[0, 0], [1, 0], [0, 1]]
            listB = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
            psf.point_sets_fitting(listA, listB)

        with self.assertRaises(Exception):
            listA = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
            listB = [[0, 0], [1, 0], [0, 1]]
            psf.point_sets_fitting(listA, listB)


    def test_fitting_translation(self):

        setA = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 2]]).transpose()
        expectedTransformation = compose([10 ,2 ,5], np.eye(3,3), np.ones(3))
        setB = expectedTransformation @ psf.to_homogeneous_repr(setA)

        expectedError = 0;

        transformation, err = psf.point_sets_fitting(setA, setB[0:3, :])

        np.testing.assert_array_almost_equal(expectedTransformation, transformation, decimal=5)
        self.assertAlmostEqual(expectedError,err,delta = 0.0001)


    def test_fitting_rotation(self):
        setA = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 2]]).transpose()

        expectedTransformation = compose(np.zeros(3),euler2mat(0.1, 0.2, 0.3),np.ones(3))
        setB = expectedTransformation @ psf.to_homogeneous_repr(setA)

        expectedError = 0;

        transformation, err = psf.point_sets_fitting(setA, setB[0:3, :])

        np.testing.assert_array_almost_equal(expectedTransformation, transformation, decimal=5)
        self.assertAlmostEqual(expectedError, err, delta=0.0001)


    def test_fitting_noisefree_3P(self):

        for k in range(1000):
            setA = np.random.rand(3,3)
            trans = np.random.rand(1, 3)
            expTr = compose(trans[0, :], euler2mat(np.random.rand(), np.random.rand(), np.random.rand()), np.ones(3))
            setB = expTr @ psf.to_homogeneous_repr(setA)

            expError = 0;

            transformation, err = psf.point_sets_fitting(setA, setB[0:3, :])

            np.testing.assert_array_almost_equal(expTr, transformation, decimal=5)
            self.assertAlmostEqual(expError, err, delta=0.0001)


    def test_fitting_noisefree_10P(self):

        for k in range(1000):
            setA = np.random.rand(3,10)
            trans = np.random.rand(1, 3)
            expTr = compose(trans[0, :], euler2mat(np.random.rand(), np.random.rand(), np.random.rand()), np.ones(3))
            setB = expTr @ psf.to_homogeneous_repr(setA)

            expError = 0;

            transformation, err = psf.point_sets_fitting(setA, setB[0:3, :])

            np.testing.assert_array_almost_equal(expTr, transformation, decimal=5)
            self.assertAlmostEqual(expError, err, delta=0.0001)


    def test_point_set_error(self):
        setA = np.array([[0, 0, 0],[1, 0, 0], [0, 1, 0], [0, 0, 2]]).transpose()
        setB = np.array([[0, 0, 0],[1, 0, 0], [0, 1, 0], [0, 0, -2]]).transpose()
        tf = np.eye(4,4)
        expectedError = 1;

        err = psf.compute_fitting_error(setA, setB, tf)

        self.assertAlmostEqual(expectedError, err, delta=0.0001)


if __name__ == '__main__':
    unittest.main()
