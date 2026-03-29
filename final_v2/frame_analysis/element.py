"""
Element Module
==============

Purpose
-------
Compute element-level quantities for a 2D frame element:
  - Element length and direction cosines (c, s — NO angle computation)
  - Local stiffness matrix [k'] (6x6)
  - Rotation matrix [R] (6x6)
  - Global stiffness matrix [k] = [R]^T [k'] [R]
  - Element end forces from global displacements

Assumptions
-----------
- 2D plane frame element with 3 DOFs per node: u_x, u_y, theta_z.
- DOF ordering per element: [u1, v1, theta1, u2, v2, theta2].
- Euler-Bernoulli beam theory (no shear deformation).
- Small displacements.
- Direction cosines c = dx/L, s = dy/L computed directly (no arctan).

Units
-----
Consistent with input (e.g. kN, m -> kN/m^2 for E).
"""

import math
from matrix_library.dense_matrix import DenseMatrix
from matrix_library.vector import Vector


def compute_element_length(x1, y1, x2, y2):
    """
    Compute element length and direction cosines from end-node coordinates.

    NOTE: Direction cosines c = dx/L, s = dy/L are computed directly.
          No angle (theta) is ever calculated — this avoids unnecessary
          trigonometric function calls and potential quadrant issues.

    Inputs
    ------
    x1, y1 : float  — Coordinates of start node (node i).
    x2, y2 : float  — Coordinates of end node (node j).

    Outputs
    -------
    L : float        — Element length.
    c : float        — cos(theta) = dx / L.
    s : float        — sin(theta) = dy / L.
    """
    dx = x2 - x1
    dy = y2 - y1
    L = math.sqrt(dx * dx + dy * dy)
    if L < 1e-15:
        raise ValueError("Zero-length element detected.")
    c = dx / L   # cos(theta) — computed directly, no arctan
    s = dy / L   # sin(theta) — computed directly, no arctan
    return L, c, s


def local_stiffness_matrix(A, I, E, L):
    """
    Form the 6x6 element stiffness matrix in local coordinates [k'].

    Uses structural analysis naming: A (area), I (inertia), E (modulus), L (length).
    DOF ordering (local): [u1, v1, theta1, u2, v2, theta2]

    Inputs
    ------
    A : float  — Cross-section area.
    I : float  — Moment of inertia.
    E : float  — Elastic modulus.
    L : float  — Element length.

    Outputs
    -------
    k : DenseMatrix (6x6)  — Element stiffness matrix in local coordinates.

    Notes
    -----
    The matrix uses axial terms (EA/L) and bending terms (EI/L^3, EI/L^2, EI/L).
    Reference: course stiffness matrix handout (Stiffness-Matrices-1.pdf).
    """
    EA = E * A
    EI = E * I
    L2 = L * L
    L3 = L2 * L

    k = DenseMatrix(6, 6)

    # Axial terms (rows/cols 0, 3)
    k[0, 0] = EA / L
    k[0, 3] = -EA / L
    k[3, 0] = -EA / L
    k[3, 3] = EA / L

    # Bending terms (rows/cols 1, 2, 4, 5)
    k[1, 1] = 12.0 * EI / L3
    k[1, 2] = 6.0 * EI / L2
    k[1, 4] = -12.0 * EI / L3
    k[1, 5] = 6.0 * EI / L2

    k[2, 1] = 6.0 * EI / L2
    k[2, 2] = 4.0 * EI / L
    k[2, 4] = -6.0 * EI / L2
    k[2, 5] = 2.0 * EI / L

    k[4, 1] = -12.0 * EI / L3
    k[4, 2] = -6.0 * EI / L2
    k[4, 4] = 12.0 * EI / L3
    k[4, 5] = -6.0 * EI / L2

    k[5, 1] = 6.0 * EI / L2
    k[5, 2] = 2.0 * EI / L
    k[5, 4] = -6.0 * EI / L2
    k[5, 5] = 4.0 * EI / L

    return k


def rotation_matrix(c, s):
    """
    Form the 6x6 rotation (transformation) matrix [R] for a 2D frame element.

    Transforms from global to local coordinates:
        {d_local} = [R] {d_global}
        [k_global] = [R]^T [k_local] [R]

    Inputs
    ------
    c : float  — cos(theta), direction cosine (C = dx/L).
    s : float  — sin(theta), direction cosine (S = dy/L).

    Outputs
    -------
    R : DenseMatrix (6x6)  — Rotation (transformation) matrix.

    Notes
    -----
    Block structure:  R = | [T] [0] |
                          | [0] [T] |
    where T = | c  s  0 |
              |-s  c  0 |
              | 0  0  1 |
    """
    R = DenseMatrix(6, 6)

    # Start node block (rows 0-2, cols 0-2)
    R[0, 0] = c
    R[0, 1] = s
    R[1, 0] = -s
    R[1, 1] = c
    R[2, 2] = 1.0

    # End node block (rows 3-5, cols 3-5)
    R[3, 3] = c
    R[3, 4] = s
    R[4, 3] = -s
    R[4, 4] = c
    R[5, 5] = 1.0

    return R


def global_stiffness_matrix(k_local, R):
    """
    Transform element stiffness from local to global coordinates.
        [k_global] = [R]^T [k_local] [R]

    Inputs
    ------
    k_local : DenseMatrix (6x6)  — Element stiffness in local coordinates [k'].
    R       : DenseMatrix (6x6)  — Rotation matrix [R].

    Outputs
    -------
    k_global : DenseMatrix (6x6)  — Element stiffness in global coordinates [k].
    """
    RT = R.transpose()
    temp = RT.matmul(k_local)
    k_global = temp.matmul(R)
    return k_global


def element_end_forces(k_local, R, d_global_elem):
    """
    Compute element end forces in local coordinates.

    Steps (from course notes, Section F):
      1. d_local = [R] * {d_global}     — transform to local coordinates
      2. f_local = [k'] * {d_local}     — multiply by local stiffness

    Inputs
    ------
    k_local      : DenseMatrix (6x6)  — Element stiffness in local coordinates [k'].
    R            : DenseMatrix (6x6)  — Rotation matrix [R].
    d_global_elem: Vector (6)         — Element end displacements in global coords.

    Outputs
    -------
    d_local : Vector (6)  — Element end displacements in local coordinates.
    f_local : Vector (6)  — Element end forces in local coordinates.
                            [N_i, V_i, M_i, N_j, V_j, M_j]
    """
    d_local = R.mat_vec(d_global_elem)
    f_local = k_local.mat_vec(d_local)
    return d_local, f_local
