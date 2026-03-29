"""
Input Data Module
=================

Purpose
-------
Define data structures (dataclass-like plain classes) that represent
the input data for a 2D frame analysis:
  - Node coordinates
  - Material / section properties
  - Element connectivity
  - Boundary conditions (supports)
  - Applied nodal loads

Input Format
------------
All input is provided as plain Python lists/dicts and wrapped into
typed objects by the FrameModel class.

Assumptions
-----------
- 2D plane frame: 3 DOFs per node (u_x, u_y, theta_z).
- Node IDs and element IDs are 1-based in the user interface
  but converted to 0-based internally.
- Units must be consistent across all input arrays.

Units
-----
User-defined (e.g. kN, m, MPa).  The program does not enforce units.
"""


class NodeCoord:
    """
    Stores the X-Y coordinates of a single node.

    Inputs
    ------
    x : float  — Global X coordinate.
    y : float  — Global Y coordinate.

    Outputs
    -------
    Object with attributes x, y (both float).
    """
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __repr__(self):
        return f"NodeCoord(x={self.x}, y={self.y})"


class MaterialProp:
    """
    Stores cross-section and material properties for one property set.

    Inputs
    ------
    area       : float  — Cross-section area (A).
    inertia    : float  — Moment of inertia (I).
    elastic_mod: float  — Elastic modulus (E).

    Outputs
    -------
    Object with attributes area (A), inertia (I), elastic_mod (E).
    """
    def __init__(self, area, inertia, elastic_mod):
        self.area = float(area)
        self.inertia = float(inertia)
        self.elastic_mod = float(elastic_mod)

    def __repr__(self):
        return (f"MaterialProp(A={self.area}, I={self.inertia}, "
                f"E={self.elastic_mod})")


class ElementConn:
    """
    Stores the connectivity and property assignment for one element.

    Inputs
    ------
    start_node : int  — Start node ID (0-based internally).
    end_node   : int  — End node ID (0-based internally).
    mat_id     : int  — Material property set ID (0-based internally).

    Outputs
    -------
    Object with attributes start_node, end_node, mat_id (all int).
    """
    def __init__(self, start_node, end_node, mat_id):
        self.start_node = int(start_node)
        self.end_node = int(end_node)
        self.mat_id = int(mat_id)

    def __repr__(self):
        return (f"ElementConn(start={self.start_node}, "
                f"end={self.end_node}, mat={self.mat_id})")


class Support:
    """
    Stores boundary condition data for one support node.

    Inputs
    ------
    node_id : int  — Node ID (0-based internally).
    rx      : int  — Restraint code for X-translation (0=free, 1=fixed).
    ry      : int  — Restraint code for Y-translation (0=free, 1=fixed).
    rz      : int  — Restraint code for Z-rotation    (0=free, 1=fixed).

    Outputs
    -------
    Object with attributes node_id, rx, ry, rz (all int).
    """
    def __init__(self, node_id, rx, ry, rz):
        self.node_id = int(node_id)
        self.rx = int(rx)
        self.ry = int(ry)
        self.rz = int(rz)

    def __repr__(self):
        return (f"Support(node={self.node_id}, "
                f"rx={self.rx}, ry={self.ry}, rz={self.rz})")


class NodalLoad:
    """
    Stores an applied load vector at a node.

    Inputs
    ------
    node_id : int    — Node ID (0-based internally).
    fx      : float  — Force in global X direction.
    fy      : float  — Force in global Y direction.
    mz      : float  — Moment about global Z axis.

    Outputs
    -------
    Object with attributes node_id (int), fx, fy, mz (all float).
    """
    def __init__(self, node_id, fx, fy, mz):
        self.node_id = int(node_id)
        self.fx = float(fx)
        self.fy = float(fy)
        self.mz = float(mz)

    def __repr__(self):
        return (f"NodalLoad(node={self.node_id}, "
                f"fx={self.fx}, fy={self.fy}, mz={self.mz})")


class FrameModel:
    """
    Top-level container that holds all input data for a frame analysis.

    Inputs
    ------
    xy           : list of [x, y] pairs     — Nodal coordinates (1-based ordering).
    materials    : list of [A, I, E] triples — Material / section property sets.
    connectivity : list of [start, end, mat] — Element connectivity (1-based IDs).
    supports     : list of [node, rx, ry, rz]— Boundary conditions (1-based node).
    loads        : list of [node, fx, fy, mz]— Applied nodal loads (1-based node).

    Outputs
    -------
    Object with attributes:
        nodes        : list[NodeCoord]     — Node coordinates (0-based).
        num_nodes    : int                 — Number of nodes.
        materials    : list[MaterialProp]  — Material properties (0-based).
        elements     : list[ElementConn]   — Element connectivity (0-based).
        num_elements : int                 — Number of elements.
        supports     : list[Support]       — Boundary conditions (0-based node).
        loads        : list[NodalLoad]     — Applied loads (0-based node).

    Notes
    -----
    All IDs are converted from 1-based (user input) to 0-based
    (internal storage) upon construction.
    """

    def __init__(self, xy, materials, connectivity, supports, loads):
        # Nodes (0-based)
        self.nodes = [NodeCoord(r[0], r[1]) for r in xy]
        self.num_nodes = len(self.nodes)

        # Materials (0-based)
        self.materials = [MaterialProp(r[0], r[1], r[2]) for r in materials]

        # Elements (convert 1-based → 0-based)
        self.elements = [
            ElementConn(r[0] - 1, r[1] - 1, r[2] - 1) for r in connectivity
        ]
        self.num_elements = len(self.elements)

        # Supports (convert 1-based node → 0-based)
        self.supports = [
            Support(r[0] - 1, r[1], r[2], r[3]) for r in supports
        ]

        # Loads (convert 1-based node → 0-based)
        self.loads = [
            NodalLoad(r[0] - 1, r[1], r[2], r[3]) for r in loads
        ]

    def __repr__(self):
        return (f"FrameModel(nodes={self.num_nodes}, "
                f"elements={self.num_elements})")
