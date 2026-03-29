"""Verification tests for the course portal-frame reference problem."""

import math
import unittest

from frame_analysis.input_data import FrameModel
from frame_analysis.assembler import (
    assign_equation_numbers,
    compute_half_bandwidth,
    assemble_global_system,
    element_dof_vector,
)
from frame_analysis.solver import solve_system


class TestPortalFrameReference(unittest.TestCase):
    """Check the key values highlighted in the homework and course PDF."""

    def setUp(self):
        xy = [
            [0.0, 0.0],
            [0.0, 3.0],
            [4.0, 3.0],
            [4.0, 0.0],
        ]
        materials = [
            [0.02, 0.08, 200000.0],
            [0.01, 0.01, 200000.0],
        ]
        connectivity = [
            [1, 2, 1],
            [2, 3, 1],
            [4, 3, 1],
            [1, 3, 2],
        ]
        supports = [
            [1, 1, 1, 0],
            [4, 0, 1, 0],
        ]
        loads = [
            [2, 10.0, -10.0, 0.0],
            [3, 10.0, -10.0, 0.0],
        ]
        self.model = FrameModel(xy, materials, connectivity, supports, loads)
        self.E, self.num_eq = assign_equation_numbers(self.model)
        self.hbw = compute_half_bandwidth(self.model, self.E)
        self.K, self.F, *_ = assemble_global_system(self.model, self.E, self.num_eq, self.hbw)
        self.D = solve_system(self.K, self.F)

    def test_equation_numbering(self):
        self.assertEqual(self.num_eq, 9)
        self.assertEqual(self.E, [
            [0, 0, 1],
            [2, 3, 4],
            [5, 6, 7],
            [8, 0, 9],
        ])

    def test_g_vectors(self):
        expected = [
            [0, 0, 1, 2, 3, 4],
            [2, 3, 4, 5, 6, 7],
            [8, 0, 9, 5, 6, 7],
            [0, 0, 1, 5, 6, 7],
        ]
        actual = [element_dof_vector(self.E, elem) for elem in self.model.elements]
        self.assertEqual(actual, expected)

    def test_global_stiffness_key_entries(self):
        dense = self.K.to_dense()
        self.assertAlmostEqual(dense[0, 0], 22933.33333333, places=5)
        self.assertAlmostEqual(dense[0, 4], 288.0, places=6)
        self.assertAlmostEqual(dense[4, 5], 99.84, places=6)
        self.assertAlmostEqual(dense[8, 8], 21333.33333333, places=5)
        self.assertTrue(dense.is_symmetric())

    def test_load_vector(self):
        expected = [0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 0.0, 0.0]
        self.assertEqual(self.F.to_list(), expected)

    def test_displacements(self):
        expected = [
            -1.145460e-02,
             3.127718e-02,
             5.451075e-04,
            -8.036203e-03,
             3.579076e-02,
            -1.875000e-02,
            -3.399152e-03,
             2.559331e-02,
            -3.399152e-03,
        ]
        for actual, target in zip(self.D.to_list(), expected):
            self.assertAlmostEqual(actual, target, places=7)


if __name__ == '__main__':
    unittest.main()
