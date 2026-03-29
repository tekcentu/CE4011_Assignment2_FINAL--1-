"""
DenseMatrix Class
=================

Purpose
-------
General-purpose dense (full-storage) matrix for structural analysis.
Supports addition, subtraction, multiplication, transposition, and
matrix-vector products.  Used for element stiffness matrices and
rotation matrices (typically 6x6).

Assumptions
-----------
- All elements are real floating-point numbers.
- Row-major storage in a flat list of length rows*cols.
- 0-based indexing internally.

Units
-----
No specific units assumed.
"""

from matrix_library.vector import Vector


class DenseMatrix:
    """
    A dense (full) matrix stored in row-major order.

    Inputs (constructor)
    --------------------
    rows : int            — Number of rows (must be > 0).
    cols : int            — Number of columns (must be > 0).
    fill : float, optional— Initial value for every element (default 0.0).

    Outputs
    -------
    DenseMatrix object of size rows x cols.

    Attributes
    ----------
    _rows : int         — Number of rows.
    _cols : int         — Number of columns.
    _data : list[float] — Flat row-major storage of size _rows * _cols.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, rows, cols, fill=0.0):
        """
        Create a rows x cols matrix filled with fill.

        Inputs
        ------
        rows : int    — Number of rows (must be > 0).
        cols : int    — Number of columns (must be > 0).
        fill : float  — Initial value for every element (default 0.0).

        Outputs
        -------
        DenseMatrix object.
        """
        if rows <= 0 or cols <= 0:
            raise ValueError("Matrix dimensions must be positive.")
        self._rows = rows
        self._cols = cols
        self._data = [float(fill)] * (rows * cols)

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_list(cls, data_2d):
        """
        Create a DenseMatrix from a 2-D Python list (list of rows).

        Inputs
        ------
        data_2d : list[list[float]]  — Each inner list is one row.

        Outputs
        -------
        m : DenseMatrix  — New matrix containing the given data.
        """
        rows = len(data_2d)
        cols = len(data_2d[0])
        m = cls(rows, cols)
        for i in range(rows):
            if len(data_2d[i]) != cols:
                raise ValueError("All rows must have the same length.")
            for j in range(cols):
                m._data[i * cols + j] = float(data_2d[i][j])
        return m

    @classmethod
    def identity(cls, n):
        """
        Create an n x n identity matrix.

        Inputs
        ------
        n : int  — Size of the square identity matrix.

        Outputs
        -------
        m : DenseMatrix  — n x n identity matrix.
        """
        m = cls(n, n)
        for i in range(n):
            m._data[i * n + i] = 1.0
        return m

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def rows(self):
        """Number of rows."""
        return self._rows

    @property
    def cols(self):
        """Number of columns."""
        return self._cols

    @property
    def shape(self):
        """(rows, cols) tuple."""
        return (self._rows, self._cols)

    # ------------------------------------------------------------------
    # Element access (0-based)
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        """
        Get element at (row, col) with 0-based indices.

        Usage: value = matrix[i, j]
        """
        i, j = idx
        if i < 0 or i >= self._rows or j < 0 or j >= self._cols:
            raise IndexError(f"Index ({i},{j}) out of bounds for "
                             f"{self._rows}x{self._cols} matrix.")
        return self._data[i * self._cols + j]

    def __setitem__(self, idx, value):
        """
        Set element at (row, col) with 0-based indices.

        Usage: matrix[i, j] = value
        """
        i, j = idx
        if i < 0 or i >= self._rows or j < 0 or j >= self._cols:
            raise IndexError(f"Index ({i},{j}) out of bounds for "
                             f"{self._rows}x{self._cols} matrix.")
        self._data[i * self._cols + j] = float(value)

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------
    def __add__(self, other):
        """Element-wise addition of two matrices."""
        if not isinstance(other, DenseMatrix):
            raise TypeError("Operand must be a DenseMatrix.")
        if self._rows != other._rows or self._cols != other._cols:
            raise ValueError("Matrix dimensions must match for addition.")
        result = DenseMatrix(self._rows, self._cols)
        for k in range(self._rows * self._cols):
            result._data[k] = self._data[k] + other._data[k]
        return result

    def __sub__(self, other):
        """Element-wise subtraction of two matrices."""
        if not isinstance(other, DenseMatrix):
            raise TypeError("Operand must be a DenseMatrix.")
        if self._rows != other._rows or self._cols != other._cols:
            raise ValueError("Matrix dimensions must match for subtraction.")
        result = DenseMatrix(self._rows, self._cols)
        for k in range(self._rows * self._cols):
            result._data[k] = self._data[k] - other._data[k]
        return result

    def __mul__(self, scalar):
        """Scalar multiplication (Matrix * scalar)."""
        result = DenseMatrix(self._rows, self._cols)
        s = float(scalar)
        for k in range(self._rows * self._cols):
            result._data[k] = self._data[k] * s
        return result

    def __rmul__(self, scalar):
        """Scalar multiplication (scalar * Matrix)."""
        return self.__mul__(scalar)

    # ------------------------------------------------------------------
    # Matrix multiplication
    # ------------------------------------------------------------------
    def matmul(self, other):
        """
        Matrix-matrix multiplication: self @ other.

        Inputs
        ------
        other : DenseMatrix  — Right-hand operand (self.cols must equal other.rows).

        Outputs
        -------
        result : DenseMatrix  — Product matrix of size (self.rows x other.cols).
        """
        if not isinstance(other, DenseMatrix):
            raise TypeError("Operand must be a DenseMatrix.")
        if self._cols != other._rows:
            raise ValueError(
                f"Inner dimensions must match: {self._cols} != {other._rows}.")
        r = DenseMatrix(self._rows, other._cols)
        for i in range(self._rows):
            for j in range(other._cols):
                s = 0.0
                for k in range(self._cols):
                    s += self._data[i * self._cols + k] * \
                         other._data[k * other._cols + j]
                r._data[i * other._cols + j] = s
        return r

    def mat_vec(self, v):
        """
        Matrix-vector product: self * v.

        Inputs
        ------
        v : Vector  — Right-hand vector (length must equal self.cols).

        Outputs
        -------
        result : Vector  — Result vector of length self.rows.
        """
        if not isinstance(v, Vector):
            raise TypeError("Operand must be a Vector.")
        if self._cols != v.size:
            raise ValueError(
                f"Matrix cols ({self._cols}) != vector size ({v.size}).")
        result = Vector(self._rows)
        for i in range(self._rows):
            s = 0.0
            for j in range(self._cols):
                s += self._data[i * self._cols + j] * v[j]
            result[i] = s
        return result

    # ------------------------------------------------------------------
    # Transpose
    # ------------------------------------------------------------------
    def transpose(self):
        """
        Return the transpose of this matrix.

        Inputs
        ------
        None.

        Outputs
        -------
        t : DenseMatrix  — Transposed matrix (cols x rows).
        """
        t = DenseMatrix(self._cols, self._rows)
        for i in range(self._rows):
            for j in range(self._cols):
                t._data[j * self._rows + i] = self._data[i * self._cols + j]
        return t

    # ------------------------------------------------------------------
    # Solve (Gaussian Elimination with Partial Pivoting)
    # ------------------------------------------------------------------
    def solve(self, rhs):
        """
        Solve [A]{x} = {b} using Gaussian elimination with partial pivoting.

        This method works for any square non-singular matrix.
        The original matrix and RHS are not modified (copies are used).

        Inputs
        ------
        rhs : Vector  — Right-hand side vector of length n.

        Outputs
        -------
        x : Vector  — Solution vector of length n.

        Raises
        ------
        ValueError  — If matrix is not square.
        ArithmeticError — If matrix is singular (zero pivot encountered).
        """
        if self._rows != self._cols:
            raise ValueError("solve() requires a square matrix.")
        n = self._rows
        if rhs.size != n:
            raise ValueError(
                f"RHS size ({rhs.size}) != matrix dim ({n}).")

        from matrix_library.vector import Vector

        # Work on copies to preserve original data
        A = self._data[:]
        b = rhs.to_list()

        # Forward elimination with partial pivoting
        for k in range(n):
            # Find pivot row
            max_val = abs(A[k * n + k])
            max_row = k
            for i in range(k + 1, n):
                val = abs(A[i * n + k])
                if val > max_val:
                    max_val = val
                    max_row = i

            if max_val < 1e-30:
                raise ArithmeticError(
                    f"Singular matrix: zero pivot at column {k}.")

            # Swap rows k and max_row
            if max_row != k:
                for j in range(n):
                    A[k * n + j], A[max_row * n + j] = (
                        A[max_row * n + j], A[k * n + j])
                b[k], b[max_row] = b[max_row], b[k]

            # Eliminate below pivot
            pivot = A[k * n + k]
            for i in range(k + 1, n):
                factor = A[i * n + k] / pivot
                for j in range(k + 1, n):
                    A[i * n + j] -= factor * A[k * n + j]
                A[i * n + k] = 0.0
                b[i] -= factor * b[k]

        # Back substitution
        x = Vector(n)
        for i in range(n - 1, -1, -1):
            s = b[i]
            for j in range(i + 1, n):
                s -= A[i * n + j] * x[j]
            x[i] = s / A[i * n + i]

        return x

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def copy(self):
        """
        Return a deep copy.

        Inputs
        ------
        None.

        Outputs
        -------
        m : DenseMatrix  — Independent copy with the same elements.
        """
        m = DenseMatrix(self._rows, self._cols)
        m._data = self._data[:]
        return m

    def to_list(self):
        """
        Return the matrix as a 2-D Python list (list of rows).

        Inputs
        ------
        None.

        Outputs
        -------
        data : list[list[float]]  — 2D list representation.
        """
        result = []
        for i in range(self._rows):
            row = []
            for j in range(self._cols):
                row.append(self._data[i * self._cols + j])
            result.append(row)
        return result

    def is_symmetric(self, tol=1e-12):
        """
        Check whether the matrix is symmetric within tolerance.

        Inputs
        ------
        tol : float  — Absolute tolerance for symmetry check (default 1e-12).

        Outputs
        -------
        result : bool  — True if |A[i,j] - A[j,i]| < tol for all i,j.
        """
        if self._rows != self._cols:
            return False
        for i in range(self._rows):
            for j in range(i + 1, self._cols):
                if abs(self[i, j] - self[j, i]) > tol:
                    return False
        return True

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self):
        return f"DenseMatrix({self._rows}, {self._cols})"

    def __str__(self):
        lines = []
        for i in range(self._rows):
            row_str = "  ".join(
                f"{self._data[i * self._cols + j]: .4e}"
                for j in range(self._cols))
            lines.append(f"[ {row_str} ]")
        return "\n".join(lines)

    def format_fixed(self, width=12, decimals=4):
        """
        Return a nicely formatted string in fixed-point notation.

        Inputs
        ------
        width    : int  — Minimum field width per element (default 12).
        decimals : int  — Number of decimal places (default 4).

        Outputs
        -------
        result : str  — Formatted multi-line string.
        """
        lines = []
        fmt = f"{{: {width}.{decimals}f}}"
        for i in range(self._rows):
            row_str = " ".join(
                fmt.format(self._data[i * self._cols + j])
                for j in range(self._cols))
            lines.append(row_str)
        return "\n".join(lines)
