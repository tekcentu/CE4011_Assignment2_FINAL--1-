"""
Postprocessor Module
====================

Purpose
-------
Extract and display results after solving the global system:
  - Nodal displacements mapped back to each node.
  - Element end displacements (global and local).
  - Element end forces in local coordinates.

Algorithm (from course notes, Section E-F)
------------------------------------------
For each element:
  1. Build the G vector from the E matrix (equation numbering).
  2. Extract element global displacements from D using G.
     If G[p] == 0, the DOF is restrained -> displacement = 0.
  3. Transform to local coordinates: {d'} = [R] {d}
  4. Compute local end forces: {f'} = [k'] {d'}

Assumptions
-----------
- The global displacement vector D has been computed.
- Equation numbering array E is available.

Units
-----
Result units depend on input units.
"""

from matrix_library.vector import Vector
from matrix_library.dense_matrix import DenseMatrix
from frame_analysis.element import (
    compute_element_length,
    local_stiffness_matrix,
    rotation_matrix,
    element_end_forces,
)
from frame_analysis.assembler import element_dof_vector


def extract_nodal_displacements(D, E, num_nodes):
    """
    Map the global displacement vector back to nodal DOFs.

    For each node and each DOF (u_x, u_y, theta_z):
      - If E[node][dof] > 0: displacement = D[E[node][dof] - 1]
      - If E[node][dof] == 0: restrained DOF -> displacement = 0.0

    Inputs
    ------
    D         : Vector         — Global displacement vector (length NumEq).
    E         : list[list[int]]— Equation numbering array, E[node][dof].
    num_nodes : int            — Number of nodes in the model.

    Outputs
    -------
    nodal_disp : list[list[float]]
        nodal_disp[node] = [u_x, u_y, theta_z].
        Restrained DOFs are set to 0.0.
    """
    nodal_disp = []
    for i in range(num_nodes):
        disp = [0.0, 0.0, 0.0]
        for j in range(3):
            eq = E[i][j]
            if eq > 0:
                disp[j] = D[eq - 1]   # eq is 1-based, D is 0-based
        nodal_disp.append(disp)
    return nodal_disp


def compute_member_forces(model, D, E, k_locals, R_mats):
    """
    Compute member end forces in local coordinates for all elements.

    Steps per element (from course notes):
      1. Build G vector from E matrix and connectivity.
      2. Extract element end displacements from D using G.
         G[p] == 0 means restrained DOF -> d = 0.
      3. Transform to local coordinates: {d'} = [R] {d_global}
      4. Compute end forces: {f'} = [k'] {d'}

    Inputs
    ------
    model    : FrameModel       — The structural model (nodes, elements, etc.).
    D        : Vector           — Global displacement vector (length NumEq).
    E        : list[list[int]]  — Equation numbering array.
    k_locals : list[DenseMatrix]— Element local stiffness matrices [k'] (6x6 each).
    R_mats   : list[DenseMatrix]— Element rotation matrices [R] (6x6 each).

    Outputs
    -------
    all_forces  : list[Vector]  — all_forces[elem] = Vector(6) of local end forces
                                   [N_i, V_i, M_i, N_j, V_j, M_j].
    all_d_local : list[Vector]  — Element end displacements in local coordinates.
    all_d_global: list[Vector]  — Element end displacements in global coordinates.
    """
    all_forces = []
    all_d_local = []
    all_d_global = []

    for idx, elem in enumerate(model.elements):
        # Step 1: Build G vector (DOF mapping for this element)
        G = element_dof_vector(E, elem)

        # Step 2: Extract element global displacements using G
        d_glob = Vector(6)
        for p in range(6):
            eq = G[p]
            if eq > 0:
                d_glob[p] = D[eq - 1]   # eq is 1-based, D is 0-based
            else:
                d_glob[p] = 0.0          # restrained DOF

        # Steps 3 & 4: Transform to local and compute forces
        # d_local = [R] * {d_global}
        # f_local = [k'] * {d_local}
        d_loc, f_loc = element_end_forces(k_locals[idx], R_mats[idx],
                                          d_glob)

        all_forces.append(f_loc)
        all_d_local.append(d_loc)
        all_d_global.append(d_glob)

    return all_forces, all_d_local, all_d_global
