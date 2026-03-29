"""
BandedSymmetricMatrix Class
===========================

Purpose
-------
Implements banded symmetric matrix storage and Cholesky-based
linear solver.  Structural stiffness matrices are typically banded
and symmetric, so this class dramatically reduces both storage and
computation compared to a full n x n matrix.

Storage Scheme
--------------
Only the upper band of each row is stored.  For an n x n symmetric
matrix with half-bandwidth *hbw* (number of off-diagonal entries
stored per row), the storage is a rectangular array of size
n x (hbw + 1).

    stored[i][j - i]  for  i <= j <= min(i + hbw, n - 1)

The total storage is n * (hbw + 1) instead of n * n.

Solution Method
---------------
Cholesky factorisation (LL^T) for symmetric positive-definite
banded systems.  The factorisation and back-substitution both operate
only within the band, achieving O(n * hbw^2) complexity.

Assumptions
-----------
- Matrix is symmetric and positive definite.
- Half-bandwidth is determined by the connectivity pattern.

Units
-----
No specific units assumed.
"""

import math
from matrix_library.vector import Vector
from matrix_library.dense_matrix import DenseMatrix


class BandedSymmetricMatrix:
    """
    Banded symmetric matrix with Cholesky factorisation and solve.

    Inputs (constructor)
    --------------------
    n   : int  — Dimension of the matrix.
    hbw : int  — Half-bandwidth (max column offset from diagonal stored).

    Outputs
    -------
    BandedSymmetricMatrix object of size n x n with half-bandwidth hbw.

    Attributes
    ----------
    _n        : int                  — Matrix dimension.
    _hbw      : int                  — Half-bandwidth (number of super-diagonals).
    _band     : list[list[float]]    — Band storage: _band[i][k] = A[i, i+k].
    _factored : bool                 — Whether Cholesky factorisation has been done.
    _L_band   : list[list[float]]    — Cholesky factor L in lower-banded form.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, n, hbw):
        """
        Create an n x n banded symmetric matrix with half-bandwidth hbw.

        Inputs
        ------
        n   : int  — Dimension of the matrix (must be > 0).
        hbw : int  — Half-bandwidth (must be >= 0).

        Outputs
        -------
        BandedSymmetricMatrix object.
        """
        if n <= 0:
            raise ValueError("Matrix dimension must be positive.")
        if hbw < 0:
            raise ValueError("Half-bandwidth must be non-negative.")
        self._n = n
        self._hbw = hbw
        # _band[i] stores row i: elements A[i, i], A[i, i+1], ..., A[i, i+hbw]
        self._band = [[0.0] * (hbw + 1) for _ in range(n)]
        self._factored = False
        self._L_band = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def size(self):
        """Matrix dimension n."""
        return self._n

    @property
    def half_bandwidth(self):
        """Half-bandwidth."""
        return self._hbw

    @property
    def shape(self):
        """(n, n) tuple."""
        return (self._n, self._n)

    @property
    def storage_count(self):
        """Total number of stored elements."""
        return self._n * (self._hbw + 1)

    # ------------------------------------------------------------------
    # Element access (0-based)
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        """Get element A[i, j].  Returns 0 if outside the band."""
        i, j = idx
        if i < 0 or i >= self._n or j < 0 or j >= self._n:
            raise IndexError(f"Index ({i},{j}) out of bounds.")
        if i > j:
            i, j = j, i  # symmetry
        k = j - i
        if k > self._hbw:
            return 0.0
        return self._band[i][k]

    def __setitem__(self, idx, value):
        """Set element A[i, j].  Both (i,j) and (j,i) are set."""
        i, j = idx
        if i < 0 or i >= self._n or j < 0 or j >= self._n:
            raise IndexError(f"Index ({i},{j}) out of bounds.")
        if i > j:
            i, j = j, i
        k = j - i
        if k > self._hbw:
            raise IndexError(
                f"Column offset {k} exceeds half-bandwidth {self._hbw}.")
        self._band[i][k] = float(value)
        self._factored = False

    def add_value(self, i, j, value):
        """
        Add a value to element (i, j).  Used during stiffness assembly.

        Inputs
        ------
        i, j  : int    — 0-based indices.
        value : float  — Value to add.

        Outputs
        -------
        None (modifies matrix in place).
        """
        if i < 0 or i >= self._n or j < 0 or j >= self._n:
            raise IndexError(f"Index ({i},{j}) out of bounds.")
        if i > j:
            i, j = j, i
        k = j - i
        if k > self._hbw:
            return  # outside band — should be zero for a banded system
        self._band[i][k] += float(value)
        self._factored = False

    # ------------------------------------------------------------------
    # Helper: get from the full matrix view (for factorisation)
    # ------------------------------------------------------------------
    def _get(self, i, j):
        """Get A[i,j] with symmetry, returning 0 if outside band."""
        if i > j:
            i, j = j, i
        k = j - i
        if k > self._hbw:
            return 0.0
        return self._band[i][k]

    # ------------------------------------------------------------------
    # Cholesky Factorisation (A = L L^T)
    # ------------------------------------------------------------------
    def factorise(self):
        """
        Perform banded Cholesky factorisation: A = L * L^T.

        Computes lower triangular L such that A = L * L^T.
        L is banded with the same half-bandwidth as A.

        L is stored in _L_band where:
            _L_band[j][i-j] = L[i,j]  for j <= i <= min(j+hbw, n-1)

        This means _L_band[j][0] = L[j,j] (diagonal),
                   _L_band[j][k] = L[j+k, j] for k = 0..hbw.

        Inputs
        ------
        None (operates on self._band).

        Outputs
        -------
        None (stores result in self._L_band, sets self._factored = True).

        Raises
        ------
        ArithmeticError  — If a zero or negative diagonal is encountered
                           (matrix is not symmetric positive definite).
        """
        n = self._n
        hbw = self._hbw

        # Build working array L in COLUMN-oriented lower-band storage.
        # Convention: L[j][k] stores L[j+k, j]  for k = 0, 1, ..., hbw.
        #   - L[j][0] = L[j,j]  (diagonal)
        #   - L[j][k] = L[j+k, j]  (sub-diagonal entries of column j)
        #
        # We initialise L from the UPPER-band storage of A.
        # Our _band stores the upper triangle row-wise:
        #   _band[i][k] = A[i, i+k]  (row i, column i+k)
        #
        # Since A is symmetric, A[i, i+k] = A[i+k, i].
        # To fill L[j][k] = A[j+k, j] = A[j, j+k] = _band[j][k].
        # So the copy is index-compatible despite the different orientation.
        L = [[0.0] * (hbw + 1) for _ in range(n)]
        for j in range(n):
            for k in range(min(hbw + 1, n - j)):
                L[j][k] = self._band[j][k]  # A[j, j+k] = A[j+k, j]

        # Cholesky: column-by-column
        for j in range(n):
            # Compute L[j,j]
            s = L[j][0]  # A[j,j]
            for k in range(max(0, j - hbw), j):
                diff = j - k
                if diff <= hbw:
                    lval = L[k][diff]
                    s -= lval * lval

            if s <= 0:
                raise ArithmeticError(
                    f"Non-positive diagonal at row {j}: matrix not SPD.")
            L[j][0] = math.sqrt(s)

            # Compute L[i, j] for i = j+1 .. min(j+hbw, n-1)
            for kk in range(1, min(hbw + 1, n - j)):
                i = j + kk
                s = L[j][kk]  # A[i, j] = A[j, i] stored in _band[j][kk]
                for k in range(max(0, max(i, j) - hbw), j):
                    di = i - k
                    dj = j - k
                    if di <= hbw and dj <= hbw:
                        s -= L[k][di] * L[k][dj]
                L[j][kk] = s / L[j][0]

        self._L_band = L
        self._factored = True

    # ------------------------------------------------------------------
    # Solve  A x = b  using Cholesky L L^T
    # ------------------------------------------------------------------
    def solve(self, rhs):
        """
        Solve the linear system A x = rhs using Cholesky LL^T.

        Algorithm:
          1. Forward substitution:  L y = rhs
          2. Back substitution:     L^T x = y

        Inputs
        ------
        rhs : Vector  — Right-hand side vector of length n.

        Outputs
        -------
        x : Vector  — Solution vector of length n.
        """
        if rhs.size != self._n:
            raise ValueError(
                f"RHS size ({rhs.size}) != matrix dim ({self._n}).")
        if not self._factored:
            self.factorise()

        n = self._n
        hbw = self._hbw
        L = self._L_band
        y = rhs.copy()

        # 1) Forward substitution: L y = b
        # L[j][0] = L[j,j], L[j][k] = L[j+k, j]
        for j in range(n):
            s = y[j]
            for k in range(max(0, j - hbw), j):
                dj = j - k
                if dj <= hbw:
                    s -= L[k][dj] * y[k]
            y[j] = s / L[j][0]

        # 2) Back substitution: L^T x = y
        # L^T[j, i] = L[i, j] for i >= j
        for j in range(n - 1, -1, -1):
            s = y[j]
            for kk in range(1, min(hbw + 1, n - j)):
                i = j + kk
                # L^T[j, i] = L[i, j] = L[j][kk]
                s -= L[j][kk] * y[i]
            y[j] = s / L[j][0]

        return y

    # ------------------------------------------------------------------
    # Matrix-vector product (using band storage)
    # ------------------------------------------------------------------
    def mat_vec(self, v):
        """
        Compute A * v using banded storage.

        Inputs
        ------
        v : Vector  — Input vector of length n.

        Outputs
        -------
        result : Vector  — Result vector A*v of length n.
        """
        if v.size != self._n:
            raise ValueError(
                f"Vector size ({v.size}) != matrix dim ({self._n}).")
        result = Vector(self._n)
        for i in range(self._n):
            # Diagonal
            result[i] = result[i] + self._band[i][0] * v[i]
            # Off-diagonals (symmetric)
            for k in range(1, min(self._hbw + 1, self._n - i)):
                j = i + k
                val = self._band[i][k]
                result[i] = result[i] + val * v[j]
                result[j] = result[j] + val * v[i]
        return result

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------
    def to_dense(self):
        """
        Convert to a full DenseMatrix (for display / verification).

        Inputs
        ------
        None.

        Outputs
        -------
        m : DenseMatrix  — Full n x n matrix with both triangles filled.
        """
        m = DenseMatrix(self._n, self._n)
        for i in range(self._n):
            for k in range(min(self._hbw + 1, self._n - i)):
                j = i + k
                val = self._band[i][k]
                m[i, j] = val
                m[j, i] = val
        return m

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self):
        return (f"BandedSymmetricMatrix(n={self._n}, "
                f"hbw={self._hbw})")

    def __str__(self):
        return self.to_dense().__str__()
