"""
Custom Object-Oriented Matrix Library for Structural Analysis
=============================================================

This library provides matrix classes and operations specifically designed
for structural analysis applications, including:
- Dense matrix operations with Gaussian elimination solver (DenseMatrix)
- Symmetric matrix storage with solver (SymmetricMatrix)
- Banded symmetric matrix storage with Cholesky LL^T solver (BandedSymmetricMatrix)
- Skyline (variable-bandwidth) matrix storage with LDLT solver (SkylineMatrix)
- Sparse matrix storage in COO and CSR formats (SparseMatrix)
- Vector operations (Vector)

No external numerical libraries (numpy, scipy, etc.) are used.

Author: CE Student
Course: Structural Analysis Programming
"""

from matrix_library.vector import Vector
from matrix_library.dense_matrix import DenseMatrix
from matrix_library.symmetric_matrix import SymmetricMatrix
from matrix_library.banded_matrix import BandedSymmetricMatrix
from matrix_library.skyline_matrix import SkylineMatrix
from matrix_library.sparse_matrix import SparseMatrix

__all__ = [
    "Vector",
    "DenseMatrix",
    "SymmetricMatrix",
    "BandedSymmetricMatrix",
    "SkylineMatrix",
    "SparseMatrix",
]
