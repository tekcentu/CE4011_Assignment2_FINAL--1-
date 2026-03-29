"""
Vector Class
============

Purpose
-------
Provides a one-dimensional vector class with standard operations required
by the structural analysis program, including addition, subtraction,
scalar multiplication, dot product, and norm computation.

Assumptions
-----------
- All elements are real floating-point numbers.
- Indexing is 0-based.

Units
-----
No specific units assumed; the caller is responsible for unit consistency.
"""

import math


class Vector:
    """
    A numerical vector with element-wise and algebraic operations.

    Inputs (constructor)
    --------------------
    n    : int            — Number of elements (must be > 0).
    fill : float, optional— Initial value for every element (default 0.0).

    Outputs
    -------
    Vector object with n elements.

    Attributes
    ----------
    _n    : int         — Number of elements.
    _data : list[float] — Internal storage as a flat Python list.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, n, fill=0.0):
        """
        Create a vector of length n filled with fill.

        Inputs
        ------
        n    : int    — Length of the vector (must be > 0).
        fill : float  — Initial value for every element (default 0.0).

        Outputs
        -------
        Vector object.
        """
        if n <= 0:
            raise ValueError("Vector size must be positive.")
        self._n = n
        self._data = [float(fill)] * n

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_list(cls, values):
        """
        Create a Vector from a Python list.

        Inputs
        ------
        values : list[float]  — The elements of the vector.

        Outputs
        -------
        v : Vector  — New vector containing the given values.
        """
        v = cls(len(values))
        v._data = [float(x) for x in values]
        return v

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def size(self):
        """Return the number of elements."""
        return self._n

    # ------------------------------------------------------------------
    # Element access (0-based)
    # ------------------------------------------------------------------
    def __getitem__(self, i):
        """Return element at 0-based index i."""
        if i < 0 or i >= self._n:
            raise IndexError(f"Index {i} out of range [0, {self._n - 1}].")
        return self._data[i]

    def __setitem__(self, i, value):
        """Set element at 0-based index i."""
        if i < 0 or i >= self._n:
            raise IndexError(f"Index {i} out of range [0, {self._n - 1}].")
        self._data[i] = float(value)

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------
    def __add__(self, other):
        """Element-wise addition of two vectors."""
        if not isinstance(other, Vector):
            raise TypeError("Operand must be a Vector.")
        if self._n != other._n:
            raise ValueError("Vector sizes must match for addition.")
        result = Vector(self._n)
        for i in range(self._n):
            result._data[i] = self._data[i] + other._data[i]
        return result

    def __sub__(self, other):
        """Element-wise subtraction of two vectors."""
        if not isinstance(other, Vector):
            raise TypeError("Operand must be a Vector.")
        if self._n != other._n:
            raise ValueError("Vector sizes must match for subtraction.")
        result = Vector(self._n)
        for i in range(self._n):
            result._data[i] = self._data[i] - other._data[i]
        return result

    def __mul__(self, scalar):
        """Scalar multiplication (Vector * scalar)."""
        result = Vector(self._n)
        s = float(scalar)
        for i in range(self._n):
            result._data[i] = self._data[i] * s
        return result

    def __rmul__(self, scalar):
        """Scalar multiplication (scalar * Vector)."""
        return self.__mul__(scalar)

    def __neg__(self):
        """Negate all elements."""
        return self.__mul__(-1.0)

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------
    def dot(self, other):
        """
        Compute the dot (inner) product of two vectors.

        Inputs
        ------
        other : Vector  — Right-hand operand (same length as self).

        Outputs
        -------
        result : float  — Scalar dot product.
        """
        if not isinstance(other, Vector):
            raise TypeError("Operand must be a Vector.")
        if self._n != other._n:
            raise ValueError("Vector sizes must match for dot product.")
        s = 0.0
        for i in range(self._n):
            s += self._data[i] * other._data[i]
        return s

    def norm(self):
        """
        Compute the Euclidean (L2) norm.

        Inputs
        ------
        None.

        Outputs
        -------
        result : float  — Euclidean norm = sqrt(dot(self, self)).
        """
        return math.sqrt(self.dot(self))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def copy(self):
        """
        Return a deep copy of this vector.

        Inputs
        ------
        None.

        Outputs
        -------
        v : Vector  — Independent copy with the same elements.
        """
        v = Vector(self._n)
        v._data = self._data[:]
        return v

    def to_list(self):
        """
        Return the internal data as a plain Python list.

        Inputs
        ------
        None.

        Outputs
        -------
        data : list[float]  — Copy of internal element storage.
        """
        return self._data[:]

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self):
        return f"Vector({self._data})"

    def __str__(self):
        return "[" + ", ".join(f"{x: .6e}" for x in self._data) + "]"

    def __len__(self):
        return self._n
