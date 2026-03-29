## Diagram 1: Matrix Library Class Hierarchy

```mermaid
classDiagram
    class Vector {
        -int _n
        -list _data
        +from_list(values)$ Vector
        +dot(other) float
        +norm() float
        +copy() Vector
        +to_list() list
        +size int
    }

    class DenseMatrix {
        -int _rows
        -int _cols
        -list _data
        +from_list(data_2d)$ DenseMatrix
        +identity(n)$ DenseMatrix
        +matmul(other) DenseMatrix
        +mat_vec(v) Vector
        +transpose() DenseMatrix
        +solve(b) Vector
        +is_symmetric() bool
        +rows int
        +cols int
    }

    class SymmetricMatrix {
        -int _n
        -int _size
        -list _data
        +add_value(i, j, v) None
        +mat_vec(v) Vector
        +solve(b) Vector
        +to_dense() DenseMatrix
        -_idx(i, j) int
        +size int
    }

    class BandedSymmetricMatrix {
        -int _n
        -int _hbw
        -list _band
        -bool _factored
        -list _L_band
        +add_value(i, j, v) None
        +factorise() None
        +solve(b) Vector
        +mat_vec(v) Vector
        +to_dense() DenseMatrix
        +size int
        +half_bandwidth int
        +storage_count int
    }

    class SkylineMatrix {
        -int _n
        -list _col_height
        -list _col_start
        -list _data
        -bool _factored
        -list _L_data
        +from_dof_map(n, G_list)$ SkylineMatrix
        +add_value(i, j, v) None
        +factorise() None
        +solve(b) Vector
        +mat_vec(v) Vector
        +to_dense() DenseMatrix
        +size int
        +storage_count int
    }

    class SparseMatrix {
        -int _n_rows
        -int _n_cols
        -list _coo_rows
        -list _coo_cols
        -list _coo_vals
        -list _row_ptr
        -list _col_idx
        -list _values
        +add_value(i, j, v) None
        +finalise() None
        +mat_vec(v) Vector
        +to_dense() DenseMatrix
        +nnz int
    }

    DenseMatrix ..> Vector : returns
    SymmetricMatrix ..> DenseMatrix : to_dense
    SymmetricMatrix ..> Vector : mat_vec / solve
    BandedSymmetricMatrix ..> DenseMatrix : to_dense
    BandedSymmetricMatrix ..> Vector : solve
    SkylineMatrix ..> DenseMatrix : to_dense
    SkylineMatrix ..> Vector : solve
    SparseMatrix ..> Vector : mat_vec

    note for BandedSymmetricMatrix "★ Used for global K\nCholesky LL^T solver\nStorage: n × hbw"
    note for SkylineMatrix "Available alternative\nLDLT solver\nStorage: Σ col_heights"
```
## Diagram 2: Frame Analysis Program Structure

```mermaid
classDiagram
    class NodeCoord {
        +float x
        +float y
    }
    class MaterialProp {
        +float area
        +float inertia
        +float elastic_mod
    }
    class ElementConn {
        +int start_node
        +int end_node
        +int mat_id
    }
    class Support {
        +int node_id
        +int rx
        +int ry
        +int rz
    }
    class NodalLoad {
        +int node_id
        +float fx
        +float fy
        +float mz
    }
    class FrameModel {
        +list nodes
        +list materials
        +list elements
        +list supports
        +list loads
        +int num_nodes
        +int num_elements
        +__init__(xy, materials, connectivity, supports, loads)
    }

    FrameModel "1" *-- "n" NodeCoord : nodes
    FrameModel "1" *-- "n" MaterialProp : materials
    FrameModel "1" *-- "n" ElementConn : elements
    FrameModel "1" *-- "0..*" Support : supports
    FrameModel "1" *-- "0..*" NodalLoad : loads

    class assembler {
        <<module>>
        +assign_equation_numbers(model) E, num_eq
        +element_dof_vector(E, elem) G
        +compute_half_bandwidth(model, E) hbw
        +assemble_global_system(model, E, n, hbw) K, F
    }
    class element {
        <<module>>
        +compute_element_length(x1,y1,x2,y2) L, c, s
        +local_stiffness_matrix(A,I,E,L) DenseMatrix
        +rotation_matrix(c, s) DenseMatrix
        +global_stiffness_matrix(k_loc, R) DenseMatrix
        +element_end_forces(k_loc, R, d_global) Vector
    }
    class solver {
        <<module>>
        +solve_system(K, F) Vector
    }
    class postprocessor {
        <<module>>
        +extract_nodal_displacements(D, E, n) list
        +compute_member_forces(model, D, E, k_locals, R_mats) list
    }
    class file_io {
        <<module>>
        +read_input_file(path) title, xy, mats, conn, sups, loads
        +write_output_file(path, ...) None
    }
    class main {
        <<module>>
        +run_sample_structure()
        +run_from_file(path)
        +run_analysis(title, ...)
    }

    main --> file_io : A. read
    main --> FrameModel : builds
    main --> assembler : B+C. E, G, K, F
    main --> solver : D. K·D=F
    main --> postprocessor : E+F. forces
    assembler --> element : calls per member
    assembler ..> BandedSymmetricMatrix : K storage
    assembler ..> Vector : F storage
    element ..> DenseMatrix : k', R, k_global
    solver ..> BandedSymmetricMatrix : .solve()
    postprocessor ..> Vector : D, forces
```

## Diagram 3: Analysis Data Flow (Steps A–G)

```mermaid
flowchart TD
    A["A — Input\nfile_io.read_input_file()\n→ FrameModel"] --> B
    B["B — Equation Numbering\nassembler.assign_equation_numbers()\n→ E matrix (nNodes×3)\n→ G vectors per element\n→ NumEq = 9"] --> C
    C["C — Assembly\nassembler.assemble_global_system()\n→ K: BandedSymmetric (63 stored)\n→ F: Vector (9 entries)\nCHECK: K symmetric ✓"] --> D
    D["D — Solve\nsolver.solve_system(K, F)\nCholesky LL^T\n→ D: Vector (9 DOFs)\nResidual: 1.4×10⁻¹³ ✓"] --> E
    E["E — Member Forces\npostprocessor.compute_member_forces()\nd' = R·d_global\nf' = k'·d'\n→ N, V, M per element"] --> F
    F["F — Equilibrium Check\nΣF_node = F_applied\nMax residual: 5.7×10⁻¹⁴ ✓\nReactions at supports"] --> G
    G["G — Storage Report\nDense: 81\nBanded: 63\nSavings: 22%"]

    style A fill:#CECBF6,stroke:#534AB7
    style B fill:#B5D4F4,stroke:#185FA5
    style C fill:#B5D4F4,stroke:#185FA5
    style D fill:#C0DD97,stroke:#3B6D11
    style E fill:#C0DD97,stroke:#3B6D11
    style F fill:#FAC775,stroke:#854F0B
    style G fill:#F1EFE8,stroke:#5F5E5A
```



# 2D Frame Analysis with Custom Matrix Library

This submission contains:

- **Q1:** an object-oriented custom matrix library written without NumPy
- **Q2:** a modular 2D frame analysis program that uses the custom matrix library for all matrix operations and the linear solve

## Main features

- Dense, symmetric, banded symmetric, sparse, and skyline matrix classes
- Banded Cholesky solver for symmetric positive-definite systems
- Course-style direct stiffness workflow using:
  - equation numbering matrix **E**
  - element address vector **G**
  - reduced-system assembly
- Plain-text input file format
- Verification using the 4-node portal frame from the course PDF

## Folder contents

- `matrix_library/` — custom matrix and vector classes
- `frame_analysis/` — structural analysis modules
- `sample_input.txt` — portal-frame verification problem
- `full_output.txt` — full sample output
- `output_results.txt` — sample output file
- `uml_diagrams.html` — UML diagrams for the matrix library and analysis flow
- `tests/` — small verification tests added for submission quality

## How to run

Run the built-in sample:

```bash
python -m frame_analysis.main
```

Run from the provided text file:

```bash
python -m frame_analysis.main sample_input.txt
```

Run from a file and also write results to disk:

```bash
python -m frame_analysis.main sample_input.txt output_results.txt
```

Run the verification tests:

```bash
python -m unittest discover -s tests -v
```

## Input format

Sections are keyword-based and case-insensitive:

- `TITLE`
- `NODES <count>`
- `MATERIALS <count>`
- `ELEMENTS <count>`
- `SUPPORTS <count>`
- `LOADS <count>`

The parser accepts explicit IDs and normalises them internally. `SUPPORTS` and `LOADS` may be written either with or without a separate record ID.

## Homework alignment notes

- **No NumPy or similar library** is used in the matrix library.
- The program uses **symmetry** and a **banded symmetric storage / Cholesky solution scheme**.
- Source files are modular, and public subroutines include docstrings describing purpose, inputs, outputs, assumptions, and units.
- The portal-frame verification reproduces the expected `E`, `G`, `K`, `F`, and `D` results from the course reference problem.
