"""
Assembler Module
================

Purpose
-------
Handle equation numbering and assembly of the global stiffness matrix
and load vector from element contributions.

Main Functions
--------------
- assign_equation_numbers : Build the DOF map (equation numbering array E).
- element_dof_vector      : Build the G vector for an element.
- compute_half_bandwidth  : Determine the half-bandwidth for banded storage.
- assemble_global_system  : Build K (banded) and F (vector) from elements.

Key Concepts (from course notes)
--------------------------------
- E matrix is EVERYTHING: it defines the reduced system.
- G vector is the bridge: it connects element DOFs to global DOFs.
- Assembly is mapping: no magic, just indexing using G.
- Skip restrained DOFs using zero: this automatically reduces the system.

Assumptions
-----------
- 3 DOFs per node: u_x, u_y, theta_z.
- Restrained DOFs are assigned equation number 0 (inactive).
- Active DOFs are numbered consecutively starting from 1.

Units
-----
No specific units assumed.
"""

import math
from matrix_library.vector import Vector
from matrix_library.dense_matrix import DenseMatrix
from matrix_library.banded_matrix import BandedSymmetricMatrix
from frame_analysis.element import (
    compute_element_length,
    local_stiffness_matrix,
    rotation_matrix,
    global_stiffness_matrix,
)


def assign_equation_numbers(model):
    """
    Construct the equation numbering array E.

    Algorithm (from course notes, Section B):
      Step 1: Create E[num_nodes][3] filled with zeros.
      Step 2: Mark restrained DOFs using the support data (S array).
      Step 3: Scan E row-wise; assign consecutive equation numbers
              to free DOFs (starting at 1) and 0 to restrained DOFs.

    The last equation number assigned = NumEq = total active DOFs.

    Inputs
    ------
    model : FrameModel  — The structural model containing nodes, supports, etc.

    Outputs
    -------
    E      : list[list[int]]  — E[node][dof] -> global equation number (1-based),
                                 or 0 if restrained. Node and dof are 0-based.
                                 dof: 0=u_x, 1=u_y, 2=theta_z
    num_eq : int              — Total number of active equations (NumEq).

    Example
    -------
    For the sample structure:
        E = [[0, 0, 1],      # Node 1: u_x, u_y restrained; theta_z = eq 1
             [2, 3, 4],      # Node 2: all free
             [5, 6, 7],      # Node 3: all free
             [8, 0, 9]]      # Node 4: u_y restrained
    """
    n = model.num_nodes

    # Step 1: initialise E with zeros
    E = [[0, 0, 0] for _ in range(n)]

    # Step 2: mark restrained DOFs (from support array S)
    for sup in model.supports:
        node = sup.node_id
        if sup.rx == 1:
            E[node][0] = -1  # mark restrained
        if sup.ry == 1:
            E[node][1] = -1
        if sup.rz == 1:
            E[node][2] = -1

    # Step 3: assign consecutive equation numbers to free DOFs
    eq_count = 0
    for i in range(n):
        for j in range(3):
            if E[i][j] == -1:
                E[i][j] = 0    # restrained -> equation number 0
            else:
                eq_count += 1
                E[i][j] = eq_count

    return E, eq_count


def element_dof_vector(E, elem):
    """
    Build the auxiliary vector G of global equation numbers for an element.

    G maps each element DOF (1..6) to its global equation number.
    G[p] = 0 means element DOF p is restrained (skip during assembly).

    Inputs
    ------
    E    : list[list[int]]  — Equation numbering array.
    elem : ElementConn      — Element connectivity (start_node, end_node).

    Outputs
    -------
    G : list[int]  — Length-6 list of global equation numbers.
                     G[0..2] = start node [u_x, u_y, theta_z]
                     G[3..5] = end node   [u_x, u_y, theta_z]
                     0 = restrained DOF.

    Example
    -------
    For Element 4 (node 1 -> node 3) in the sample structure:
        G = [0, 0, 1, 5, 6, 7]
    Meaning: start node DOFs 1,2 restrained (0), DOF 3 = eq 1;
             end node DOFs 4,5,6 = equations 5,6,7.
    """
    sn = elem.start_node
    en = elem.end_node
    G = [E[sn][0], E[sn][1], E[sn][2],
         E[en][0], E[en][1], E[en][2]]
    return G


def compute_half_bandwidth(model, E):
    """
    Determine the half-bandwidth of the global stiffness matrix
    from the equation numbering and connectivity.

    The half-bandwidth = max difference between any two active
    equation numbers within the same element, over all elements.

    Inputs
    ------
    model : FrameModel          — The structural model.
    E     : list[list[int]]     — Equation numbering array.

    Outputs
    -------
    hbw : int  — Half-bandwidth of the global stiffness matrix.
    """
    hbw = 0
    for elem in model.elements:
        G = element_dof_vector(E, elem)
        active = [g for g in G if g > 0]
        if len(active) >= 2:
            diff = max(active) - min(active)
            if diff > hbw:
                hbw = diff
    return hbw


def assemble_global_system(model, E, num_eq, hbw):
    """
    Assemble the global stiffness matrix K and force vector F.

    Assembly algorithm (from course notes, Section C and D):

    Stiffness assembly — for each element:
        1. Compute L, c, s from node coordinates
        2. Form local stiffness k' (6x6)
        3. Form rotation matrix R (6x6) using c, s (NOT angle)
        4. Transform: k = R^T k' R (global stiffness)
        5. Build G vector from E
        6. Assemble into K:
             for p in 1..6:
                 P = G[p]
                 if P == 0: continue
                 for q in 1..6:
                     Q = G[q]
                     if Q == 0: continue
                     K[P, Q] += k[p, q]

    Load assembly — for each loaded joint:
        N = node ID
        for q in 1..3:
            Q = E[N, q]
            if Q != 0:
                F[Q] += load_value

    Inputs
    ------
    model  : FrameModel          — The structural model.
    E      : list[list[int]]     — Equation numbering array.
    num_eq : int                 — Total number of active equations (NumEq).
    hbw    : int                 — Half-bandwidth for banded storage.

    Outputs
    -------
    K        : BandedSymmetricMatrix  — Global stiffness matrix (banded storage).
    F        : Vector                 — Global force vector (length num_eq).
    k_globals: list[DenseMatrix]      — Element global stiffness matrices [k] (for verification).
    k_locals : list[DenseMatrix]      — Element local stiffness matrices [k'] (for verification).
    R_mats   : list[DenseMatrix]      — Element rotation matrices [R] (for post-processing).
    """
    K = BandedSymmetricMatrix(num_eq, hbw)
    F = Vector(num_eq)

    k_globals = []
    k_locals = []
    R_mats = []

    # ===== STIFFNESS ASSEMBLY (Section C of course notes) =====
    for elem in model.elements:
        sn = elem.start_node
        en = elem.end_node
        mat = model.materials[elem.mat_id]

        # Step 1: Geometry — compute L, c, s directly (no angle!)
        x1, y1 = model.nodes[sn].x, model.nodes[sn].y
        x2, y2 = model.nodes[en].x, model.nodes[en].y
        L, c, s = compute_element_length(x1, y1, x2, y2)

        # Step 2: Form local stiffness k' (6x6)
        k_loc = local_stiffness_matrix(mat.area, mat.inertia,
                                       mat.elastic_mod, L)

        # Step 3: Form rotation matrix R (6x6)
        R = rotation_matrix(c, s)

        # Step 4: Transform to global: k = R^T k' R
        k_glob = global_stiffness_matrix(k_loc, R)

        k_locals.append(k_loc)
        R_mats.append(R)
        k_globals.append(k_glob)

        # Step 5: Build G vector (DOF mapping)
        G = element_dof_vector(E, elem)

        # Step 6: Assemble into K
        # Equation numbers are 1-based; convert to 0-based for K storage.
        # Since K is banded symmetric, only add upper triangle (P <= Q)
        # to avoid double-counting.
        for p in range(6):
            P = G[p]
            if P == 0:          # restrained DOF -> skip
                continue
            for q in range(6):
                Q = G[q]
                if Q == 0:      # restrained DOF -> skip
                    continue
                if P <= Q:      # upper triangle only (symmetric storage)
                    K.add_value(P - 1, Q - 1, k_glob[p, q])

    # ===== LOAD ASSEMBLY (Section D of course notes) =====
    for load in model.loads:
        node = load.node_id             # N = node ID
        load_vals = [load.fx, load.fy, load.mz]
        for dof in range(3):            # for q in 1..3
            Q = E[node][dof]            # Q = E[N, q]
            if Q != 0:                  # if Q != 0
                F[Q - 1] = F[Q - 1] + load_vals[dof]

    return K, F, k_globals, k_locals, R_mats
