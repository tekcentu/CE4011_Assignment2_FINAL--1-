"""
Solver Module
=============

Purpose
-------
Solve the global system of equations [K]{D} = {F} using the
Cholesky LL^T solver from the custom banded symmetric matrix library.

Algorithm
---------
The BandedSymmetricMatrix.solve() method performs:
  1. Cholesky factorisation: K = L * L^T  (banded, O(n * hbw^2))
  2. Forward substitution:   L * y = F
  3. Back substitution:      L^T * D = y

Assumptions
-----------
- K is symmetric and positive definite.
- K is stored in BandedSymmetricMatrix format.
- The reduced system (restrained DOFs already removed via E matrix)
  is passed in.

Units
-----
No specific units assumed; result units depend on input units.
"""

from matrix_library.banded_matrix import BandedSymmetricMatrix
from matrix_library.vector import Vector


def solve_system(K, F):
    """
    Solve [K]{D} = {F} for {D}.

    Uses Cholesky LL^T factorisation for banded symmetric
    positive-definite systems.

    Inputs
    ------
    K : BandedSymmetricMatrix  — Global stiffness matrix (banded symmetric storage).
    F : Vector                 — Global force vector (length NumEq).

    Outputs
    -------
    D : Vector  — Global displacement vector (length NumEq).
    """
    # Cholesky LL^T factorisation + forward/back substitution
    D = K.solve(F)
    return D
