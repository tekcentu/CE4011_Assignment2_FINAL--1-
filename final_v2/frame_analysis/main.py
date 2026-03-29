"""
Main Driver — 2D Frame Analysis
================================

Purpose
-------
Entry point that ties together all modules following the course algorithm:
  A. Read / define input data (from file or hardcoded)
  B. Equation numbering (build E matrix)
  C. Assemble global stiffness K and load vector F
  D. Solve for displacements: [K]{D} = {F}
  E. Compute member end forces
  F. Verify against reference solution
  G. Report storage efficiency

Usage
-----
  python -m frame_analysis.main                      # hardcoded sample
  python -m frame_analysis.main input.txt            # read from file
  python -m frame_analysis.main input.txt output.txt # read + write results

Units
-----
kN, m, kN/m^2 (consistent).
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matrix_library.vector import Vector
from matrix_library.dense_matrix import DenseMatrix
from matrix_library.banded_matrix import BandedSymmetricMatrix
from frame_analysis.input_data import FrameModel
from frame_analysis.assembler import (
    assign_equation_numbers,
    compute_half_bandwidth,
    assemble_global_system,
    element_dof_vector,
)
from frame_analysis.solver import solve_system
from frame_analysis.postprocessor import (
    extract_nodal_displacements,
    compute_member_forces,
)
from frame_analysis.element import (
    compute_element_length,
    rotation_matrix,
    global_stiffness_matrix,
)
from frame_analysis.file_io import (
    read_input_file,
    write_output_file,
    write_sample_input_file,
)


def print_separator(title=""):
    """
    Print a visual separator with an optional title.

    Inputs
    ------
    title : str  — Optional section title to display.

    Outputs
    -------
    None (prints to stdout).
    """
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def check_global_equilibrium(model, E, D, K, F, all_forces, R_mats):
    """
    Verify equilibrium at each node by summing element end forces
    in global coordinates and comparing to applied loads.

    At every free node, the sum of element end forces (in global coords)
    should equal the applied loads.  At support nodes, the residual
    gives the support reactions.

    Inputs
    ------
    model      : FrameModel             — The structural model.
    E          : list[list[int]]        — Equation numbering array.
    D          : Vector                 — Global displacement vector.
    K          : BandedSymmetricMatrix  — Global stiffness matrix.
    F          : Vector                 — Global force vector.
    all_forces : list[Vector]           — Member end forces (local coords, 6 each).
    R_mats     : list[DenseMatrix]      — Rotation matrices for each element.

    Outputs
    -------
    None (prints equilibrium check to stdout).
    """
    num_nodes = model.num_nodes

    # Accumulate global-coordinate element end forces at each node
    # nodal_forces[node] = [Fx_sum, Fy_sum, Mz_sum]
    nodal_forces = [[0.0, 0.0, 0.0] for _ in range(num_nodes)]

    for idx, elem in enumerate(model.elements):
        # Transform local end forces to global: {f_global} = [R]^T {f_local}
        RT = R_mats[idx].transpose()
        f_global = RT.mat_vec(all_forces[idx])

        sn = elem.start_node
        en = elem.end_node

        # Start node contributes DOFs 0,1,2
        for dof in range(3):
            nodal_forces[sn][dof] += f_global[dof]

        # End node contributes DOFs 3,4,5
        for dof in range(3):
            nodal_forces[en][dof] += f_global[3 + dof]

    # Build applied load map
    applied = [[0.0, 0.0, 0.0] for _ in range(num_nodes)]
    for load in model.loads:
        applied[load.node_id][0] += load.fx
        applied[load.node_id][1] += load.fy
        applied[load.node_id][2] += load.mz

    print("\n  Node  |  Sum(Fx)      Sum(Fy)      Sum(Mz)     | "
          " Applied Fx    Applied Fy    Applied Mz")
    print("  ------+--------------------------------------------+"
          "------------------------------------------")

    max_residual = 0.0
    for i in range(num_nodes):
        # Check if this is a free node (all DOFs active)
        is_free = all(E[i][d] > 0 for d in range(3))
        marker = " " if is_free else "*"

        # Residual = element forces - applied loads (should be ~0 at free nodes)
        res = [nodal_forces[i][d] - applied[i][d] for d in range(3)]

        if is_free:
            for d in range(3):
                if abs(res[d]) > max_residual:
                    max_residual = abs(res[d])

        print(f"  {i+1:4d}{marker} |"
              f" {nodal_forces[i][0]: .4f}"
              f"  {nodal_forces[i][1]: .4f}"
              f"  {nodal_forces[i][2]: .4f}"
              f"    |"
              f" {applied[i][0]: .4f}"
              f"  {applied[i][1]: .4f}"
              f"  {applied[i][2]: .4f}")

    print(f"\n  (* = support node — residual gives reaction forces)")
    print(f"  Max equilibrium residual at free nodes: {max_residual:.6e}")
    if max_residual < 1e-8:
        print("  CHECK: Equilibrium satisfied at all free nodes. ✓")
    else:
        print("  WARNING: Equilibrium residual is large!")


def run_sample_structure():
    """
    Define and analyse the sample structure from the course notes.

    Sample Structure
    ----------------
    4 nodes, 4 elements
      Node 1: (0, 0) — pin support (u_x, u_y restrained; rotation free)
      Node 2: (0, 3) — free
      Node 3: (4, 3) — free
      Node 4: (4, 0) — roller support (u_y restrained; u_x, rotation free)

    Elements: 1(1→2), 2(2→3), 3(4→3), 4(1→3)
    Materials: Set 1: A=0.02, I=0.08, E=200000
               Set 2: A=0.01, I=0.01, E=200000
    Loads: Node 2: Fx=10, Fy=-10, Mz=0
           Node 3: Fx=10, Fy=-10, Mz=0

    Inputs
    ------
    None (input data hardcoded for the sample structure).

    Outputs
    -------
    model      : FrameModel             — The structural model.
    E          : list[list[int]]        — Equation numbering array.
    num_eq     : int                    — Total active equations.
    K          : BandedSymmetricMatrix  — Global stiffness matrix.
    F          : Vector                 — Global force vector.
    D          : Vector                 — Global displacement vector.
    k_locals   : list[DenseMatrix]      — Element local stiffness matrices.
    k_globals  : list[DenseMatrix]      — Element global stiffness matrices.
    R_mats     : list[DenseMatrix]      — Element rotation matrices.
    all_forces : list[Vector]           — Member end forces (local coords).
    """
    # ================================================================
    # A. INPUT PHASE
    # ================================================================
    print_separator("A. INPUT DATA")

    # Nodal coordinates [x, y] (1-based ordering in the list)
    xy = [
        [0.0, 0.0],  # Node 1
        [0.0, 3.0],  # Node 2
        [4.0, 3.0],  # Node 3
        [4.0, 0.0],  # Node 4
    ]

    # Material properties [A, I, E]
    materials = [
        [0.02, 0.08, 200000.0],  # Set 1
        [0.01, 0.01, 200000.0],  # Set 2
    ]

    # Connectivity [start_node, end_node, material_set] (1-based)
    connectivity = [
        [1, 2, 1],  # Element 1
        [2, 3, 1],  # Element 2
        [4, 3, 1],  # Element 3
        [1, 3, 2],  # Element 4
    ]

    # Supports [node, rx, ry, rz] (1-based node)
    # rx, ry, rz: 1=restrained, 0=free
    supports = [
        [1, 1, 1, 0],  # Node 1: pin (u_x, u_y fixed; rotation free)
        [4, 0, 1, 0],  # Node 4: roller (u_y fixed; u_x, rotation free)
    ]

    # Loads [node, Fx, Fy, Mz] (1-based node)
    loads = [
        [2, 10.0, -10.0, 0.0],
        [3, 10.0, -10.0, 0.0],
    ]

    return run_analysis("Sample 4-Node Frame (Course Notes)",
                        xy, materials, connectivity, supports, loads)


def run_from_file(input_path, output_path=None):
    """
    Read input from file, run analysis, optionally write output file.

    Inputs
    ------
    input_path  : str       — Path to input file.
    output_path : str|None  — Path to output file (optional).

    Outputs
    -------
    Same as run_analysis().
    """
    print(f"Reading input from: {input_path}")
    title, xy, materials, connectivity, supports, loads = \
        read_input_file(input_path)

    results = run_analysis(title, xy, materials, connectivity, supports, loads)

    if output_path:
        model, E, num_eq, K, F, D, k_locals, k_globals, R_mats, all_forces = results
        nodal_disp = extract_nodal_displacements(D, E, model.num_nodes)
        _, all_d_local, all_d_global = compute_member_forces(
            model, D, E, k_locals, R_mats)
        write_output_file(output_path, title, model, E, num_eq, K, F, D,
                          k_locals, k_globals, R_mats,
                          nodal_disp, all_forces, all_d_local, all_d_global)
        print(f"\nResults written to: {output_path}")

    return results


def run_analysis(title, xy, materials, connectivity, supports, loads):
    """
    Core analysis routine — used by both hardcoded and file-driven modes.

    Inputs
    ------
    title        : str                    — Model title.
    xy           : list of [x, y]        — Nodal coordinates.
    materials    : list of [A, I, E]     — Material properties.
    connectivity : list of [sn, en, mat] — Element connectivity (1-based).
    supports     : list of [node, rx, ry, rz] — Boundary conditions (1-based).
    loads        : list of [node, Fx, Fy, Mz] — Nodal loads (1-based).

    Outputs
    -------
    model, E, num_eq, K, F, D, k_locals, k_globals, R_mats, all_forces
    """
    model = FrameModel(xy, materials, connectivity, supports, loads)

    print(f"Title: {title}")
    print(f"Number of nodes:    {model.num_nodes}")
    print(f"Number of elements: {model.num_elements}")
    print(f"Number of supports: {len(model.supports)}")
    print(f"Number of load cases: {len(model.loads)}")

    print("\nNode Coordinates:")
    for i, nd in enumerate(model.nodes):
        print(f"  Node {i + 1}: ({nd.x}, {nd.y})")

    print("\nMaterial Properties:")
    for i, m in enumerate(model.materials):
        print(f"  Set {i + 1}: A={m.area}, I={m.inertia}, E={m.elastic_mod}")

    print("\nElement Connectivity:")
    for i, e in enumerate(model.elements):
        print(f"  Elem {i + 1}: {e.start_node + 1} -> {e.end_node + 1}, "
              f"MatSet={e.mat_id + 1}")

    # ================================================================
    # B. EQUATION NUMBERING (Build E matrix)
    # ================================================================
    print_separator("B. EQUATION NUMBERING")

    # E matrix: maps each node DOF to a global equation number.
    # E[node][dof] = equation number (1-based), or 0 if restrained.
    E, num_eq = assign_equation_numbers(model)

    print("Equation numbers (E array):")
    print("  Node  |  u_x  u_y  theta_z")
    print("  ------+---------------------")
    for i in range(model.num_nodes):
        print(f"  {i + 1:4d}  |  {E[i][0]:4d}  {E[i][1]:4d}  {E[i][2]:4d}")
    print(f"\nTotal equations (NumEq): {num_eq}")

    # Verification: E matrix should match the course notes example
    E_expected = [[0, 0, 1], [2, 3, 4], [5, 6, 7], [8, 0, 9]]
    if E == E_expected:
        print("CHECK: E matrix matches course notes. ✓")

    # ================================================================
    # C. GLOBAL STIFFNESS MATRIX AND FORCE VECTOR
    # ================================================================
    print_separator("C. ASSEMBLY")

    hbw = compute_half_bandwidth(model, E)
    print(f"Half-bandwidth: {hbw}")

    K, F, k_globals, k_locals, R_mats = assemble_global_system(
        model, E, num_eq, hbw)

    # Print element local stiffness matrices
    for i in range(model.num_elements):
        print(f"\nElement {i + 1} - Local Stiffness [k'] (6x6):")
        print(k_locals[i].format_fixed(width=14, decimals=2))

    # Print element global stiffness matrices
    for i in range(model.num_elements):
        print(f"\nElement {i + 1} - Global Stiffness [k] (6x6):")
        print(k_globals[i].format_fixed(width=14, decimals=2))

    # Print global stiffness matrix (convert banded to dense for display)
    K_dense = K.to_dense()
    print(f"\nGlobal Stiffness Matrix K ({num_eq}x{num_eq}):")
    print(K_dense.format_fixed(width=14, decimals=2))

    # Verification: K should be symmetric
    if K_dense.is_symmetric():
        print("CHECK: K is symmetric. ✓")
    else:
        print("WARNING: K is NOT symmetric!")

    print(f"\nGlobal Force Vector F:")
    for i in range(num_eq):
        print(f"  F[{i + 1}] = {F[i]: .4f}")

    # ================================================================
    # D. SOLVE FOR DISPLACEMENTS
    # ================================================================
    print_separator("D. SOLUTION — NODAL DISPLACEMENTS")

    D = solve_system(K, F)

    print("Global Displacement Vector D:")
    for i in range(num_eq):
        print(f"  D[{i + 1}] = {D[i]: .6e}")

    # Verification: check K*D = F residual
    KD = K.mat_vec(D)
    residual = KD - F
    res_norm = residual.norm()
    print(f"\nResidual check: ||K*D - F|| = {res_norm:.6e}")
    if res_norm < 1e-10:
        print("CHECK: K*D = F satisfied (residual < 1e-10). ✓")
    else:
        print("WARNING: K*D = F residual is large!")

    # Map displacements back to nodes
    nodal_disp = extract_nodal_displacements(D, E, model.num_nodes)
    print("\nNodal Displacements:")
    print("  Node  |     u_x          u_y        theta_z")
    print("  ------+------------------------------------------")
    for i in range(model.num_nodes):
        print(f"  {i + 1:4d}  | {nodal_disp[i][0]: .6e}  "
              f"{nodal_disp[i][1]: .6e}  {nodal_disp[i][2]: .6e}")

    # ================================================================
    # E. MEMBER END FORCES
    # ================================================================
    print_separator("E. MEMBER END FORCES")

    all_forces, all_d_local, all_d_global = compute_member_forces(
        model, D, E, k_locals, R_mats)

    for i in range(model.num_elements):
        sn = model.elements[i].start_node + 1
        en = model.elements[i].end_node + 1
        print(f"\nElement {i + 1} ({sn} -> {en}):")
        print(f"  Global displacements: {all_d_global[i]}")
        print(f"  Local displacements:  {all_d_local[i]}")
        print(f"  Local end forces:     {all_forces[i]}")
        print(f"    N_i = {all_forces[i][0]: .4f} kN")
        print(f"    V_i = {all_forces[i][1]: .4f} kN")
        print(f"    M_i = {all_forces[i][2]: .4f} kN·m")
        print(f"    N_j = {all_forces[i][3]: .4f} kN")
        print(f"    V_j = {all_forces[i][4]: .4f} kN")
        print(f"    M_j = {all_forces[i][5]: .4f} kN·m")

    # Equilibrium check at each node
    print_separator("E2. EQUILIBRIUM CHECK")
    check_global_equilibrium(model, E, D, K, F, all_forces, R_mats)

    # ================================================================
    # F. VERIFICATION AGAINST REFERENCE
    # ================================================================
    print_separator("F. VERIFICATION")

    # Reference displacements from course notes (9 values, 3-sig-fig approx)
    D_ref = [
        -1.15e-2, 3.13e-2, 5.45e-4,
        -8.04e-3, 3.58e-2, -1.87e-2,
        -3.40e-3, 2.56e-2, -3.40e-3
    ]

    print("Displacement Comparison (Computed vs. Reference):")
    print("  Eq#  |    Computed       Reference       Rel. Error")
    print("  -----+----------------------------------------------")
    max_err = 0.0
    for i in range(num_eq):
        computed = D[i]
        ref = D_ref[i]
        if abs(ref) > 1e-15:
            err = abs(computed - ref) / abs(ref) * 100
        else:
            err = abs(computed - ref) * 100
        if err > max_err:
            max_err = err
        print(f"  {i + 1:4d} | {computed: .6e}  {ref: .6e}  {err: .4f}%")

    print(f"\nMaximum relative error: {max_err:.4f}%")
    if max_err < 1.0:
        print("VERIFICATION PASSED: All displacements match within 1%. ✓")
    else:
        print("WARNING: Some displacements differ by more than 1%.")

    # Note: Reference values are 3-significant-figure approximations
    # from the course notes, so small differences (<1%) are expected.

    # ================================================================
    # G. STORAGE COMPARISON
    # ================================================================
    print_separator("G. STORAGE EFFICIENCY")
    full_storage = num_eq * num_eq
    banded_storage = K.storage_count
    savings = (1 - banded_storage / full_storage) * 100
    print(f"Full matrix storage:   {full_storage} elements")
    print(f"Banded storage:        {banded_storage} elements")
    print(f"Storage savings:       {savings:.1f}%")

    return model, E, num_eq, K, F, D, k_locals, k_globals, R_mats, all_forces


if __name__ == "__main__":
    # Generate sample input file for reference
    sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "sample_input.txt")
    write_sample_input_file(sample_path)

    if len(sys.argv) >= 3:
        # File-driven mode with output
        run_from_file(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        # File-driven mode (stdout only)
        run_from_file(sys.argv[1])
    else:
        # Default: hardcoded sample structure
        run_sample_structure()
