"""
SparseMatrix Class (COO + CSR)
==============================

Purpose
-------
Implements sparse matrix storage using two standard formats:
  1. COO (Coordinate / Triplet) — used during assembly (easy to add entries).
  2. CSR (Compressed Sparse Row) — used for efficient matrix-vector products
     and solving (after assembly is complete).

For structural analysis, the stiffness matrix is assembled entry-by-entry
(COO phase), then converted to CSR for efficient operations.

Storage Schemes
---------------
COO: Three parallel lists (row_idx, col_idx, values).
     Duplicate (i,j) entries are summed during CSR conversion.

CSR: Three arrays:
     - row_ptr[i]  = index into col_idx/values where row i starts
     - col_idx[k]  = column index of entry k
     - values[k]   = value of entry k
     Row i entries are in col_idx[row_ptr[i] : row_ptr[i+1]].

Assumptions
-----------
- Real-valued elements.
- For symmetric systems, both (i,j) and (j,i) are stored explicitly
  (full storage of both triangles) to simplify matrix-vector products.

Units
-----
No specific units assumed.
"""

from matrix_library.vector import Vector
from matrix_library.dense_matrix import DenseMatrix


class SparseMatrix:
    """
    Sparse matrix supporting COO assembly and CSR operations.

    Inputs (constructor)
    --------------------
    n_rows : int  — Number of rows.
    n_cols : int  — Number of columns.

    Outputs
    -------
    SparseMatrix object (starts in COO mode for assembly).

    Attributes
    ----------
    _n_rows  : int  — Number of rows.
    _n_cols  : int  — Number of columns.

    COO storage (assembly phase):
    _coo_rows : list[int]   — Row indices.
    _coo_cols : list[int]   — Column indices.
    _coo_vals : list[float] — Values.

    CSR storage (after finalise):
    _row_ptr  : list[int]   — Row pointers (length n_rows + 1).
    _col_idx  : list[int]   — Column indices.
    _values   : list[float] — Non-zero values.
    _finalised: bool        — Whether CSR has been built.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, n_rows, n_cols):
        """
        Create an empty sparse matrix.

        Inputs
        ------
        n_rows : int  — Number of rows.
        n_cols : int  — Number of columns.

        Outputs
        -------
        SparseMatrix in COO mode (ready for assembly).
        """
        if n_rows <= 0 or n_cols <= 0:
            raise ValueError("Matrix dimensions must be positive.")
        self._n_rows = n_rows
        self._n_cols = n_cols

        # COO storage
        self._coo_rows = []
        self._coo_cols = []
        self._coo_vals = []

        # CSR storage
        self._row_ptr = None
        self._col_idx = None
        self._values = None
        self._finalised = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def rows(self):
        """Number of rows."""
        return self._n_rows

    @property
    def cols(self):
        """Number of columns."""
        return self._n_cols

    @property
    def shape(self):
        """(rows, cols) tuple."""
        return (self._n_rows, self._n_cols)

    @property
    def nnz(self):
        """
        Number of stored non-zero entries.
        Before finalise: count of COO triplets (may include duplicates).
        After finalise: count of unique entries in CSR.
        """
        if self._finalised:
            return len(self._values)
        return len(self._coo_vals)

    @property
    def storage_count(self):
        """Total number of stored entries (same as nnz after finalise)."""
        return self.nnz

    # ------------------------------------------------------------------
    # COO assembly
    # ------------------------------------------------------------------
    def add_value(self, i, j, value):
        """
        Add a value at position (i, j).  Duplicate entries are allowed;
        they will be summed during finalise().

        Inputs
        ------
        i     : int    — Row index (0-based).
        j     : int    — Column index (0-based).
        value : float  — Value to add.

        Outputs
        -------
        None (appends to COO lists).
        """
        if i < 0 or i >= self._n_rows:
            raise IndexError(f"Row {i} out of range [0, {self._n_rows - 1}].")
        if j < 0 or j >= self._n_cols:
            raise IndexError(f"Col {j} out of range [0, {self._n_cols - 1}].")
        if abs(value) < 1e-30:
            return  # skip exact zeros
        self._coo_rows.append(i)
        self._coo_cols.append(j)
        self._coo_vals.append(float(value))
        self._finalised = False

    def add_value_symmetric(self, i, j, value):
        """
        Add a value at (i, j) and also at (j, i) for symmetric assembly.
        If i == j, only one entry is added.

        Inputs
        ------
        i, j  : int    — 0-based indices.
        value : float  — Value to add.

        Outputs
        -------
        None.
        """
        self.add_value(i, j, value)
        if i != j:
            self.add_value(j, i, value)

    # ------------------------------------------------------------------
    # Finalise: COO → CSR conversion
    # ------------------------------------------------------------------
    def finalise(self):
        """
        Convert COO storage to CSR format.
        Duplicate entries at the same (i, j) are summed.

        Inputs
        ------
        None.

        Outputs
        -------
        None (populates self._row_ptr, self._col_idx, self._values).
        """
        n = self._n_rows
        num_entries = len(self._coo_vals)

        # Merge duplicates: group by (row, col) and sum
        merged = {}
        for k in range(num_entries):
            key = (self._coo_rows[k], self._coo_cols[k])
            if key in merged:
                merged[key] += self._coo_vals[k]
            else:
                merged[key] = self._coo_vals[k]

        # Sort by (row, col)
        sorted_entries = sorted(merged.items(), key=lambda x: (x[0][0], x[0][1]))

        # Build CSR
        self._row_ptr = [0] * (n + 1)
        self._col_idx = []
        self._values = []

        for (i, j), val in sorted_entries:
            self._col_idx.append(j)
            self._values.append(val)
            self._row_ptr[i + 1] += 1

        # Cumulative sum for row_ptr
        for i in range(1, n + 1):
            self._row_ptr[i] += self._row_ptr[i - 1]

        self._finalised = True

    # ------------------------------------------------------------------
    # Element access (0-based)
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        """
        Get element A[i, j].  Requires finalise() to have been called.
        Returns 0 if not stored.
        """
        i, j = idx
        if not self._finalised:
            self.finalise()
        if i < 0 or i >= self._n_rows or j < 0 or j >= self._n_cols:
            raise IndexError(f"Index ({i},{j}) out of bounds.")
        # Binary search in row i
        start = self._row_ptr[i]
        end = self._row_ptr[i + 1]
        for k in range(start, end):
            if self._col_idx[k] == j:
                return self._values[k]
        return 0.0

    # ------------------------------------------------------------------
    # Matrix-vector product (CSR)
    # ------------------------------------------------------------------
    def mat_vec(self, v):
        """
        Compute A * v using CSR storage.

        Inputs
        ------
        v : Vector  — Input vector of length n_cols.

        Outputs
        -------
        result : Vector  — Result vector of length n_rows.
        """
        if not self._finalised:
            self.finalise()
        if v.size != self._n_cols:
            raise ValueError(
                f"Vector size ({v.size}) != matrix cols ({self._n_cols}).")
        result = Vector(self._n_rows)
        for i in range(self._n_rows):
            s = 0.0
            for k in range(self._row_ptr[i], self._row_ptr[i + 1]):
                s += self._values[k] * v[self._col_idx[k]]
            result[i] = s
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
        m : DenseMatrix  — Full matrix.
        """
        if not self._finalised:
            self.finalise()
        m = DenseMatrix(self._n_rows, self._n_cols)
        for i in range(self._n_rows):
            for k in range(self._row_ptr[i], self._row_ptr[i + 1]):
                m[i, self._col_idx[k]] = self._values[k]
        return m

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self):
        return (f"SparseMatrix({self._n_rows}x{self._n_cols}, "
                f"nnz={self.nnz})")

    def __str__(self):
        if not self._finalised:
            return f"SparseMatrix(COO, {len(self._coo_vals)} entries)"
        return self.to_dense().__str__()
