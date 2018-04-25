import unittest
from function import *
import numpy as np


class TestFunction(unittest.TestCase):

    def setUp(self):
        self.fonction = Function()
        self.x = 2.3
        self.out_x = 2.3
        self.derivate_x = 1
        self.vector = np.arange(3, dtype=np.float64)
        self.out_vector = np.arange(3, dtype=np.float64)
        self.derivate_vector = np.ones(3, dtype=np.float64)
        self.matrix = np.ones((4, 4), dtype=np.float64)*1.4
        self.out_matrix = np.ones((4, 4), dtype=np.float64)*1.4
        self.derivate_matrix = np.ones((4, 4), dtype=np.float64)

    def test_out(self):
        self.assertAlmostEqual(self.fonction.out(self.x), self.out_x)
        self.assertTrue(np.allclose(self.fonction.out(self.vector), self.out_vector))
        self.assertTrue(np.allclose(self.fonction.out(self.matrix), self.out_matrix))

    def test_derivate(self):
        self.assertAlmostEqual(self.fonction.derivate(self.x), self.derivate_x)
        self.assertTrue(np.allclose(self.fonction.derivate(self.vector), self.derivate_vector))
        self.assertTrue(np.allclose(self.fonction.derivate(self.matrix), self.derivate_matrix))

