"""
SymmetricMatrix Class
=====================

Purpose
-------
Store a symmetric matrix using only the upper-triangular portion,
reducing storage by nearly half.  Provides the same public interface
as DenseMatrix for read/write access and matrix-vector products.
This class is useful for structural stiffness matrices.

Storage Scheme
--------------
Only the upper triangle (including diagonal) is stored in a flat list
of length n*(n+1)/2, mapped by the formula:
    index(i, j) = i * n - i*(i-1)//2 + (j - i)      for i <= j
When the user accesses (i, j) with i > j, the class transparently
swaps indices so that storage remains upper-triangular.

Assumptions
-----------
- Square matrix (n x n).
- Symmetric: A[i,j] == A[j,i].
- Real-valued elements.

Units
-----
No specific units assumed.
"""

from matrix_library.vector import Vector
from matrix_library.dense_matrix import DenseMatrix


class SymmetricMatrix:
    """
    A symmetric matrix stored in packed upper-triangular form.

    Inputs (constructor)
    --------------------
    n    : int            — Dimension of the square symmetric matrix.
    fill : float, optional— Initial value for every stored element (default 0.0).

    Outputs
    -------
    SymmetricMatrix object of size n x n.

    Attributes
    ----------
    _n    : int         — Matrix dimension (n x n).
    _data : list[float] — Flat storage of upper triangle, length n*(n+1)/2.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, n, fill=0.0):
        """
        Create an n x n symmetric matrix filled with fill.

        Inputs
        ------
        n    : int    — Dimension of the square symmetric matrix.
        fill : float  — Initial value for every stored element (default 0.0).

        Outputs
        -------
        SymmetricMatrix object.
        """
        if n <= 0:
            raise ValueError("Matrix dimension must be positive.")
        self._n = n
        self._size = n * (n + 1) // 2
        self._data = [float(fill)] * self._size

    # ------------------------------------------------------------------
    # Internal index mapping
    # ------------------------------------------------------------------
    def _idx(self, i, j):
        """
        Map (i, j) to flat index in upper-triangular storage.
        Swaps i, j when i > j to ensure i <= j.

        Inputs
        ------
        i, j : int  — 0-based row and column indices.

        Outputs
        -------
        index : int  — Flat storage index.
        """
        if i > j:
            i, j = j, i
        return i * self._n - i * (i - 1) // 2 + (j - i)

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
        """Number of stored elements (upper triangle)."""
        return self._size

    # ------------------------------------------------------------------
    # Element access (0-based)
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        """Get element at (row, col) — symmetric access."""
        i, j = idx
        if i < 0 or i >= self._n or j < 0 or j >= self._n:
            raise IndexError(f"Index ({i},{j}) out of bounds for "
                             f"{self._n}x{self._n} symmetric matrix.")
        return self._data[self._idx(i, j)]

    def __setitem__(self, idx, value):
        """Set element at (row, col) — sets both (i,j) and (j,i)."""
        i, j = idx
        if i < 0 or i >= self._n or j < 0 or j >= self._n:
            raise IndexError(f"Index ({i},{j}) out of bounds for "
                             f"{self._n}x{self._n} symmetric matrix.")
        self._data[self._idx(i, j)] = float(value)

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------
    def mat_vec(self, v):
        """
        Symmetric matrix-vector product using only upper triangle.

        Inputs
        ------
        v : Vector  — Input vector of length n.

        Outputs
        -------
        result : Vector  — Result vector of length n.
        """
        if v.size != self._n:
            raise ValueError(
                f"Vector size ({v.size}) != matrix dim ({self._n}).")
        result = Vector(self._n)
        for i in range(self._n):
            s = 0.0
            # Diagonal
            s += self._data[self._idx(i, i)] * v[i]
            # Off-diagonal (use symmetry)
            for j in range(i + 1, self._n):
                val = self._data[self._idx(i, j)]
                s += val * v[j]
                result[j] = result[j] + val * v[i]
            result[i] = result[i] + s
        return result

    def add_value(self, i, j, value):
        """
        Add a value to element (i, j).  Useful for assembly.

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
        self._data[self._idx(i, j)] += float(value)

    # ------------------------------------------------------------------
    # Solve (converts to DenseMatrix, uses Gaussian elimination)
    # ------------------------------------------------------------------
    def solve(self, rhs):
        """
        Solve [A]{x} = {b} for this symmetric matrix.

        Converts to DenseMatrix internally and uses Gaussian elimination
        with partial pivoting.  For large systems, prefer
        BandedSymmetricMatrix or SkylineMatrix which exploit structure.

        Inputs
        ------
        rhs : Vector  — Right-hand side vector of length n.

        Outputs
        -------
        x : Vector  — Solution vector of length n.
        """
        dense = self.to_dense()
        return dense.solve(rhs)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------
    def to_dense(self):
        """
        Convert to a full DenseMatrix (for display or non-symmetric ops).

        Inputs
        ------
        None.

        Outputs
        -------
        m : DenseMatrix  — Full n x n matrix with both triangles filled.
        """
        m = DenseMatrix(self._n, self._n)
        for i in range(self._n):
            for j in range(i, self._n):
                val = self._data[self._idx(i, j)]
                m[i, j] = val
                m[j, i] = val
        return m

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self):
        return f"SymmetricMatrix({self._n})"

    def __str__(self):
        return self.to_dense().__str__()
