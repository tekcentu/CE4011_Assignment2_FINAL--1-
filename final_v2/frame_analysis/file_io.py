"""
File I/O Module
===============

Purpose
-------
Read structural model data from a plain-text input file and write
analysis results to a plain-text output file.

This module supports the course-style sectioned text format and performs
basic validation / normalisation so the analysis core receives compact,
1-based consecutive IDs even if the file uses explicit IDs that are
out-of-order.

Input File Format
-----------------
The input file uses a keyword-based plain-text format. Lines starting
with '#' are comments and are ignored. Blank lines are also ignored.
Keywords are case-insensitive.

Supported sections:

    TITLE
    <title string on next line>

    NODES  <NumNode>
    <node_id>  <x>  <y>
    ...

    MATERIALS  <NumMat>
    <mat_id>  <A>  <I>  <E>
    ...

    ELEMENTS  <NumElem>
    <elem_id>  <start_node>  <end_node>  <mat_id>
    ...

    SUPPORTS  <NumSupport>
    <node_id>              <rx>  <ry>  <rz>
    or
    <support_id> <node_id> <rx>  <ry>  <rz>

    LOADS  <NumLoadJoint>
    <node_id>           <Fx>  <Fy>  <Mz>
    or
    <load_id> <node_id> <Fx>  <Fy>  <Mz>

All IDs are user-facing and 1-based in the file. Internally, the data
is normalised to compact 1..n ordering for nodes and material sets.

Assumptions
-----------
- The structure is a 2D plane frame with 3 DOFs per node.
- Node and material references in ELEMENTS / SUPPORTS / LOADS must exist.
- Units must be consistent (for example: kN, m, kN/m^2).
- Duplicate IDs are not allowed within a section.

Units
-----
User-defined; the program does not enforce specific units.
"""

from __future__ import annotations


def _normalise_id_table(records, id_name):
    """
    Sort records by explicit ID and build a compact 1-based remapping.

    Inputs
    ------
    records : list[tuple[int, object]]
        Pairs of (explicit_id, payload) extracted from the input file.
    id_name : str
        Human-readable ID name used in error messages.

    Outputs
    -------
    ordered_payloads : list[object]
        Payloads sorted by their explicit IDs.
    id_map : dict[int, int]
        Mapping from explicit input ID to compact 1-based internal ID.
    """
    seen = set()
    for rec_id, _ in records:
        if rec_id in seen:
            raise ValueError(f"Duplicate {id_name}: {rec_id}")
        seen.add(rec_id)

    ordered = sorted(records, key=lambda item: item[0])
    id_map = {rec_id: idx + 1 for idx, (rec_id, _) in enumerate(ordered)}
    payloads = [payload for _, payload in ordered]
    return payloads, id_map


def read_input_file(filepath):
    """
    Read structural model data from a plain-text input file.

    Inputs
    ------
    filepath : str
        Path to the input file.

    Outputs
    -------
    title : str
        Model title.
    xy : list[list[float]]
        Nodal coordinates in compact node order [[x, y], ...].
    materials : list[list[float]]
        Material/section properties in compact material order [[A, I, E], ...].
    connectivity : list[list[int]]
        Element connectivity [[start_node, end_node, mat_id], ...] using compact
        1-based node/material IDs expected by the analysis core.
    supports : list[list[int]]
        Support data [[node_id, rx, ry, rz], ...] using compact 1-based node IDs.
    loads : list[list[float]]
        Nodal loads [[node_id, Fx, Fy, Mz], ...] using compact 1-based node IDs.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is invalid or references undefined IDs.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    lines = []
    for line in raw_lines:
        stripped = line.strip()
        if stripped == '' or stripped.startswith('#'):
            continue
        lines.append(stripped)

    title = ""
    node_records = []
    material_records = []
    element_records = []
    support_records = []
    load_records = []

    idx = 0
    while idx < len(lines):
        token = lines[idx].upper().split()

        if token[0] == 'TITLE':
            idx += 1
            if idx < len(lines):
                title = lines[idx]
            idx += 1
            continue

        if token[0] == 'NODES':
            num_nodes = int(token[1])
            for _ in range(num_nodes):
                idx += 1
                parts = lines[idx].split()
                if len(parts) != 3:
                    raise ValueError('Each NODES row must contain: node_id x y')
                node_id = int(parts[0])
                node_records.append((node_id, [float(parts[1]), float(parts[2])]))
            idx += 1
            continue

        if token[0] == 'MATERIALS':
            num_mats = int(token[1])
            for _ in range(num_mats):
                idx += 1
                parts = lines[idx].split()
                if len(parts) != 4:
                    raise ValueError('Each MATERIALS row must contain: mat_id A I E')
                mat_id = int(parts[0])
                material_records.append((mat_id, [float(parts[1]), float(parts[2]), float(parts[3])]))
            idx += 1
            continue

        if token[0] == 'ELEMENTS':
            num_elem = int(token[1])
            for _ in range(num_elem):
                idx += 1
                parts = lines[idx].split()
                if len(parts) != 4:
                    raise ValueError('Each ELEMENTS row must contain: elem_id start_node end_node mat_id')
                elem_id = int(parts[0])
                element_records.append((elem_id, [int(parts[1]), int(parts[2]), int(parts[3])]))
            idx += 1
            continue

        if token[0] == 'SUPPORTS':
            num_sup = int(token[1])
            for row_idx in range(num_sup):
                idx += 1
                parts = lines[idx].split()
                if len(parts) == 4:
                    record_id = row_idx + 1
                    node_id, rx, ry, rz = map(int, parts)
                elif len(parts) == 5:
                    record_id = int(parts[0])
                    node_id, rx, ry, rz = map(int, parts[1:])
                else:
                    raise ValueError('Each SUPPORTS row must contain either: node_id rx ry rz or support_id node_id rx ry rz')
                support_records.append((record_id, [node_id, rx, ry, rz]))
            idx += 1
            continue

        if token[0] == 'LOADS':
            num_loads = int(token[1])
            for row_idx in range(num_loads):
                idx += 1
                parts = lines[idx].split()
                if len(parts) == 4:
                    record_id = row_idx + 1
                    node_id = int(parts[0])
                    fx, fy, mz = map(float, parts[1:])
                elif len(parts) == 5:
                    record_id = int(parts[0])
                    node_id = int(parts[1])
                    fx, fy, mz = map(float, parts[2:])
                else:
                    raise ValueError('Each LOADS row must contain either: node_id Fx Fy Mz or load_id node_id Fx Fy Mz')
                load_records.append((record_id, [node_id, fx, fy, mz]))
            idx += 1
            continue

        idx += 1

    if not node_records:
        raise ValueError('No NODES section found in input file.')
    if not material_records:
        raise ValueError('No MATERIALS section found in input file.')
    if not element_records:
        raise ValueError('No ELEMENTS section found in input file.')

    xy, node_id_map = _normalise_id_table(node_records, 'node ID')
    materials, mat_id_map = _normalise_id_table(material_records, 'material ID')

    ordered_elements, _ = _normalise_id_table(element_records, 'element ID')
    ordered_supports, _ = _normalise_id_table(support_records, 'support ID')
    ordered_loads, _ = _normalise_id_table(load_records, 'load ID')

    connectivity = []
    for start_node, end_node, mat_id in ordered_elements:
        if start_node not in node_id_map or end_node not in node_id_map:
            raise ValueError('ELEMENTS section references undefined node ID.')
        if mat_id not in mat_id_map:
            raise ValueError('ELEMENTS section references undefined material ID.')
        connectivity.append([node_id_map[start_node], node_id_map[end_node], mat_id_map[mat_id]])

    supports = []
    for node_id, rx, ry, rz in ordered_supports:
        if node_id not in node_id_map:
            raise ValueError('SUPPORTS section references undefined node ID.')
        supports.append([node_id_map[node_id], rx, ry, rz])

    loads = []
    for node_id, fx, fy, mz in ordered_loads:
        if node_id not in node_id_map:
            raise ValueError('LOADS section references undefined node ID.')
        loads.append([node_id_map[node_id], fx, fy, mz])

    return title, xy, materials, connectivity, supports, loads


def write_output_file(filepath, title, model, E, num_eq, K, F, D,
                      k_locals, k_globals, R_mats,
                      nodal_disp, all_forces, all_d_local, all_d_global):
    """
    Write complete analysis results to a plain-text output file.

    Inputs
    ------
    filepath : str
        Path to the output file.
    title : str
        Model title.
    model : FrameModel
        Structural model container.
    E : list[list[int]]
        Equation numbering array.
    num_eq : int
        Total number of active equations.
    K : BandedSymmetricMatrix
        Global stiffness matrix in banded symmetric storage.
    F : Vector
        Global force vector.
    D : Vector
        Global displacement vector.
    k_locals : list[DenseMatrix]
        Element local stiffness matrices.
    k_globals : list[DenseMatrix]
        Element global stiffness matrices.
    R_mats : list[DenseMatrix]
        Element rotation matrices.
    nodal_disp : list[list[float]]
        Nodal displacement table [u_x, u_y, theta_z].
    all_forces : list[Vector]
        Element end-force vectors in local coordinates.
    all_d_local : list[Vector]
        Element displacement vectors in local coordinates.
    all_d_global : list[Vector]
        Element displacement vectors in global coordinates.

    Outputs
    -------
    None
        Results are written to disk at ``filepath``.
    """
    sep = "=" * 70

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"{sep}\n")
        f.write("  2D FRAME ANALYSIS - RESULTS\n")
        f.write(f"  {title}\n")
        f.write(f"{sep}\n\n")

        f.write(f"{sep}\n  A. INPUT SUMMARY\n{sep}\n")
        f.write(f"Number of nodes: {model.num_nodes}\n")
        f.write(f"Number of elements: {model.num_elements}\n")
        f.write(f"Number of material sets: {len(model.materials)}\n")
        f.write(f"Number of supports: {len(model.supports)}\n")
        f.write(f"Number of load entries: {len(model.loads)}\n\n")

        f.write(f"{sep}\n  B. EQUATION NUMBERING\n{sep}\n")
        f.write("Node   u_x   u_y   theta_z\n")
        for i, row in enumerate(E, start=1):
            f.write(f"{i:4d} {row[0]:5d} {row[1]:5d} {row[2]:8d}\n")
        f.write(f"\nNumEq = {num_eq}\n\n")

        f.write(f"{sep}\n  C. GLOBAL STIFFNESS MATRIX\n{sep}\n")
        K_dense = K.to_dense()
        for i in range(K_dense.rows):
            row_vals = [f"{K_dense[i, j]:12.4f}" for j in range(K_dense.cols)]
            f.write(" ".join(row_vals) + "\n")
        f.write("\n")

        f.write(f"{sep}\n  D. GLOBAL LOAD VECTOR\n{sep}\n")
        for i in range(F.size):
            f.write(f"F[{i+1}] = {F[i]: .6e}\n")
        f.write("\n")

        f.write(f"{sep}\n  E. DISPLACEMENTS\n{sep}\n")
        for i in range(D.size):
            f.write(f"D[{i+1}] = {D[i]: .6e}\n")
        f.write("\n")

        f.write(f"{sep}\n  F. MEMBER END FORCES (LOCAL)\n{sep}\n")
        for idx, force_vec in enumerate(all_forces, start=1):
            vals = ", ".join(f"{force_vec[k]: .6e}" for k in range(6))
            f.write(f"Element {idx}: [{vals}]\n")
        f.write("\n")


def write_sample_input_file(filepath):
    """
    Write the course portal-frame verification problem to a sample input file.

    Inputs
    ------
    filepath : str
        Destination path for the sample input file.

    Outputs
    -------
    None
        Sample input text is written to disk.
    """
    sample = """# ======================================================
# Sample Structure - 2D Frame Analysis
# From Course Notes (FrameAnalysisProgram_Details-1.pdf)
# ======================================================
# Units: kN, m, kN/m^2

TITLE
Sample 4-Node Frame (Course Notes)

NODES 4
1  0.0  0.0
2  0.0  3.0
3  4.0  3.0
4  4.0  0.0

MATERIALS 2
1  0.02  0.08  200000.0
2  0.01  0.01  200000.0

ELEMENTS 4
1  1  2  1
2  2  3  1
3  4  3  1
4  1  3  2

SUPPORTS 2
1  1  1  0
4  0  1  0

LOADS 2
2  10.0  -10.0  0.0
3  10.0  -10.0  0.0
"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(sample)
