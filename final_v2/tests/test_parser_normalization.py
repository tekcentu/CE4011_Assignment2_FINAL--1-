"""Tests for the text-file parser normalisation logic."""

import tempfile
import textwrap
import unittest
from pathlib import Path

from frame_analysis.file_io import read_input_file


class TestParserNormalization(unittest.TestCase):
    """Check that explicit file IDs are handled robustly."""

    def test_unsorted_ids_and_optional_record_ids(self):
        data = textwrap.dedent(
            """
            TITLE
            Unsorted IDs Example

            NODES 4
            40  4.0  0.0
            10  0.0  0.0
            30  4.0  3.0
            20  0.0  3.0

            MATERIALS 2
            2  0.01  0.01  200000.0
            1  0.02  0.08  200000.0

            ELEMENTS 4
            300  40  30  1
            100  10  20  1
            400  10  30  2
            200  20  30  1

            SUPPORTS 2
            11  10  1  1  0
            22  40  0  1  0

            LOADS 2
            7  20  10.0  -10.0  0.0
            8  30  10.0  -10.0  0.0
            """
        ).strip()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'input.txt'
            path.write_text(data, encoding='utf-8')
            title, xy, materials, connectivity, supports, loads = read_input_file(path)

        self.assertEqual(title, 'Unsorted IDs Example')
        self.assertEqual(xy, [[0.0, 0.0], [0.0, 3.0], [4.0, 3.0], [4.0, 0.0]])
        self.assertEqual(materials, [[0.02, 0.08, 200000.0], [0.01, 0.01, 200000.0]])
        self.assertEqual(connectivity, [[1, 2, 1], [2, 3, 1], [4, 3, 1], [1, 3, 2]])
        self.assertEqual(supports, [[1, 1, 1, 0], [4, 0, 1, 0]])
        self.assertEqual(loads, [[2, 10.0, -10.0, 0.0], [3, 10.0, -10.0, 0.0]])


if __name__ == '__main__':
    unittest.main()
