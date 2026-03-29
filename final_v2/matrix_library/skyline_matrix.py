"""
SkylineMatrix Class
===================

Purpose
-------
Implements skyline (variable-bandwidth / envelope) storage for symmetric
matrices, along with an LDLT factorisation and solve.

Skyline storage is the natural generalisation of banded storage:
instead of a fixed half-bandwidth for every row, each row stores only
from the first non-zero entry to the diagonal.  For stiffness matrices
whose bandwidth varies by row, skyline storage is more compact than
banded storage.

Storage Scheme
--------------
We store the upper-triangle columns in a single flat array `_data`.
For each column j, the entries from the first non-zero row to the
diagonal (inclusive) are stored contiguously.

    _col_start[j]  = index into _data where column j begins
    _col_height[j] = number of entries stored above (and including)
                      the diagonal in column j

    A[i, j]  (i <= j)  is stored at _data[_col_start[j] + (i - (j - _col_height[j] + 1))]
    provided  i >= (j - _col_height[j] + 1),  else the element is zero.

Solution Method
---------------
LDLT factorisation for symmetric systems (D is diagonal, L is
unit lower triangular).  This avoids square roots, unlike Cholesky.

Assumptions
-----------
- Matrix is symmetric and positive definite.
- Column heights are determined from the sparsity pattern.

Units
-----
No specific units assumed.
"""

import math
from matrix_library.vector import Vector
from matrix_library.dense_matrix import DenseMatrix


class SkylineMatrix:
    """
    Symmetric matrix with skyline (variable-bandwidth) storage
    and LDLT factorisation.

    Inputs (constructor)
    --------------------
    n            : int        — Dimension of the matrix (n x n).
    col_heights  : list[int]  — col_heights[j] = number of entries stored in
                                column j (from the first non-zero row to the
                                diagonal, inclusive). col_heights[0] = 1 always.

    Outputs
    -------
    SkylineMatrix object of size n x n.

    Attributes
    ----------
    _n           : int         — Matrix dimension.
    _col_height  : list[int]   — Column heights.
    _col_start   : list[int]   — Starting index in _data for each column.
    _data        : list[float] — Flat skyline storage.
    _factored    : bool        — Whether LDLT factorisation has been done.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, n, col_heights):
        """
        Create an n x n skyline matrix with given column heights.

        Inputs
        ------
        n           : int        — Matrix dimension.
        col_heights : list[int]  — Column heights (length n).
                                   col_heights[j] >= 1 (at least the diagonal).

        Outputs
        -------
        SkylineMatrix object.
        """
        if n <= 0:
            raise ValueError("Matrix dimension must be positive.")
        if len(col_heights) != n:
            raise ValueError("col_heights must have length n.")
        for j in range(n):
            if col_heights[j] < 1:
                raise ValueError(
                    f"Column height[{j}] must be >= 1 (at least diagonal).")

        self._n = n
        self._col_height = list(col_heights)

        # Compute column start indices
        self._col_start = [0] * n
        total = 0
        for j in range(n):
            self._col_start[j] = total
            total += self._col_height[j]

        self._data = [0.0] * total
        self._factored = False
        self._L_data = None   # factorised data
        self._D_diag = None   # diagonal of D in LDLT

    # ------------------------------------------------------------------
    # Alternate constructor: from column heights computed via connectivity
    # ------------------------------------------------------------------
    @classmethod
    def from_dof_map(cls, n, element_dof_lists):
        """
        Create a SkylineMatrix by computing column heights from
        the DOF connectivity pattern (list of G vectors).

        Inputs
        ------
        n                : int              — Number of equations (NumEq).
        element_dof_lists: list[list[int]]  — For each element, a list of global
                                              equation numbers (1-based, 0=restrained).

        Outputs
        -------
        sky : SkylineMatrix  — Skyline matrix with correct column heights.

        Notes
        -----
        For each column j, the column height is determined by the minimum
        active equation number that appears in the same element as j.
        """
        # min_row[j] = minimum row index (0-based) that couples with column j
        min_row = list(range(n))  # initially, only diagonal

        for G in element_dof_lists:
            active = [g - 1 for g in G if g > 0]  # convert to 0-based
            if len(active) < 2:
                continue
            mn = min(active)
            for g in active:
                if mn < min_row[g]:
                    min_row[g] = mn

        col_heights = [0] * n
        for j in range(n):
            col_heights[j] = j - min_row[j] + 1

        return cls(n, col_heights)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def size(self):
        """Matrix dimension n."""
        return self._n

    @property
    def shape(self):
        """(n, n) tuple."""
        return (self._n, self._n)

    @property
    def storage_count(self):
        """Total number of stored elements."""
        return len(self._data)

    @property
    def col_heights(self):
        """Column heights (read-only copy)."""
        return list(self._col_height)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _first_row(self, j):
        """
        First row index (0-based) stored in column j.

        Inputs
        ------
        j : int  — Column index (0-based).

        Outputs
        -------
        row : int  — First stored row in column j.
        """
        return j - self._col_height[j] + 1

    def _in_skyline(self, i, j):
        """
        Check if (i, j) is within the skyline envelope (upper triangle).

        Inputs
        ------
        i, j : int  — 0-based row and column (i <= j).

        Outputs
        -------
        result : bool  — True if within envelope.
        """
        return i >= self._first_row(j)

    def _flat_idx(self, i, j):
        """
        Flat index into _data for element (i, j) with i <= j.

        Inputs
        ------
        i, j : int  — 0-based indices, i <= j, within skyline.

        Outputs
        -------
        idx : int  — Index into self._data.
        """
        return self._col_start[j] + (i - self._first_row(j))

    # ------------------------------------------------------------------
    # Element access (0-based, symmetric)
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        """
        Get element A[i, j].  Returns 0 if outside the skyline envelope.
        Symmetric: A[i, j] = A[j, i].
        """
        i, j = idx
        if i < 0 or i >= self._n or j < 0 or j >= self._n:
            raise IndexError(f"Index ({i},{j}) out of bounds.")
        if i > j:
            i, j = j, i  # symmetry
        if not self._in_skyline(i, j):
            return 0.0
        return self._data[self._flat_idx(i, j)]

    def __setitem__(self, idx, value):
        """
        Set element A[i, j].  Both (i,j) and (j,i) are set (symmetric).
        Raises error if outside the skyline envelope.
        """
        i, j = idx
        if i < 0 or i >= self._n or j < 0 or j >= self._n:
            raise IndexError(f"Index ({i},{j}) out of bounds.")
        if i > j:
            i, j = j, i
        if not self._in_skyline(i, j):
            raise IndexError(
                f"({i},{j}) is outside the skyline envelope for column {j} "
                f"(first row = {self._first_row(j)}).")
        self._data[self._flat_idx(i, j)] = float(value)
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
        if not self._in_skyline(i, j):
            return  # outside envelope — structurally zero
        self._data[self._flat_idx(i, j)] += float(value)
        self._factored = False

    # ------------------------------------------------------------------
    # LDLT Factorisation (A = L D L^T)
    # ------------------------------------------------------------------
    def factorise(self):
        """
        Perform skyline LDLT factorisation: A = L D L^T.

        L is unit lower triangular (stored in skyline format),
        D is diagonal.

        Inputs
        ------
        None (operates on self._data).

        Outputs
        -------
        None (stores L in self._L_data, D diagonal in self._D_diag).

        Raises
        ------
        ArithmeticError  — If a zero diagonal pivot is encountered.
        """
        n = self._n
        # Work on a copy of the skyline data.
        # After factorisation:
        #   work[diag_pos(j)] = D[j]          (diagonal of D)
        #   work[off-diag]    = L[j,i] values (unit lower triangular L)
        # Convention: we store the UPPER triangle columns.
        #   In column j, entry at row i (i < j) corresponds to L[j, i]
        #   (note: L is lower-triangular, so L[j,i] for j > i).
        work = self._data[:]

        for j in range(n):
            fr = self._first_row(j)

            # Step 1: For each off-diagonal row i in column j (i = fr..j-1),
            # compute the modified value:
            #   work[i,j] = A[i,j] - sum_{k} L[i,k] * D[k] * L[j,k]
            # where the sum is over k in the intersection of column i and
            # column j active rows, with k < i.
            #
            # After this, work[i,j] holds L[j,i] * D[i].

            for i in range(fr, j):
                fi = self._first_row(i)
                k_start = max(fr, fi)
                s = work[self._col_start[j] + (i - fr)]
                for k in range(k_start, i):
                    # L[i,k] is at column i, row k: work[col_start[i] + (k - fi)]
                    # L[j,k] is at column j, row k: work[col_start[j] + (k - fr)]
                    lik = work[self._col_start[i] + (k - fi)]
                    ljk = work[self._col_start[j] + (k - fr)]
                    s -= lik * ljk
                work[self._col_start[j] + (i - fr)] = s

            # Step 2: Compute D[j] and divide off-diagonals by D[i].
            # At this point, work[i,j] = L[j,i] * D[i] for i < j.
            # Save these as temp, then D[j] = A[j,j] - sum L[j,i]^2 * D[i]
            #                                = A[j,j] - sum temp[i] * (temp[i]/D[i])
            #                                = A[j,j] - sum temp[i]^2 / D[i]

            diag_idx = self._col_start[j] + (j - fr)
            s = work[diag_idx]  # A[j,j] value
            for i in range(fr, j):
                off_idx = self._col_start[j] + (i - fr)
                temp_val = work[off_idx]  # = L[j,i] * D[i]
                # D[i] is stored at the diagonal of column i
                Di = work[self._col_start[i] + (i - self._first_row(i))]
                if abs(Di) < 1e-30:
                    raise ArithmeticError(
                        f"Zero pivot at row {i} during LDLT factorisation.")
                s -= temp_val * temp_val / Di
                # Divide to get L[j,i] = temp_val / D[i]
                work[off_idx] = temp_val / Di

            if abs(s) < 1e-30:
                raise ArithmeticError(
                    f"Zero pivot at row {j} during LDLT factorisation.")
            # Store D[j] at the diagonal position
            work[diag_idx] = s

        self._L_data = work
        # Extract diagonal D for convenience
        D = [0.0] * n
        for j in range(n):
            D[j] = work[self._col_start[j] + (j - self._first_row(j))]
        self._D_diag = D
        self._factored = True

    # ------------------------------------------------------------------
    # Solve  A x = b  using LDLT
    # ------------------------------------------------------------------
    def solve(self, rhs):
        """
        Solve A x = rhs using skyline LDLT factorisation.

        Algorithm:
          1. LDLT factorisation (if not already done)
          2. Forward substitution:  L y = rhs
          3. Diagonal scaling:      D z = y
          4. Back substitution:     L^T x = z

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
        L = self._L_data
        D = self._D_diag
        x = rhs.copy()

        # 1) Forward substitution: L y = b
        # L is unit lower triangular: L[j,j] = 1, L[i,j] stored for i > j
        # But our storage is column-oriented upper triangle of L^T,
        # i.e., L[j, i] for i < j is stored at _col_start[j] + (i - first_row(j))
        for j in range(n):
            fr = self._first_row(j)
            for i in range(fr, j):
                idx = self._col_start[j] + (i - fr)
                x[j] = x[j] - L[idx] * x[i]

        # 2) Diagonal scaling: z = D^{-1} y
        for j in range(n):
            x[j] = x[j] / D[j]

        # 3) Back substitution: L^T x = z
        for j in range(n - 1, -1, -1):
            fr = self._first_row(j)
            for i in range(fr, j):
                idx = self._col_start[j] + (i - fr)
                x[i] = x[i] - L[idx] * x[j]

        return x

    # ------------------------------------------------------------------
    # Matrix-vector product
    # ------------------------------------------------------------------
    def mat_vec(self, v):
        """
        Compute A * v using skyline storage.

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
        for j in range(self._n):
            fr = self._first_row(j)
            # Diagonal contribution
            diag_idx = self._flat_idx(j, j)
            result[j] = result[j] + self._data[diag_idx] * v[j]
            # Off-diagonal (symmetric)
            for i in range(fr, j):
                idx = self._flat_idx(i, j)
                val = self._data[idx]
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
        m : DenseMatrix  — Full n x n matrix.
        """
        m = DenseMatrix(self._n, self._n)
        for j in range(self._n):
            fr = self._first_row(j)
            for i in range(fr, j + 1):
                val = self._data[self._flat_idx(i, j)]
                m[i, j] = val
                m[j, i] = val
        return m

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self):
        return (f"SkylineMatrix(n={self._n}, "
                f"storage={self.storage_count})")

    def __str__(self):
        return self.to_dense().__str__()
