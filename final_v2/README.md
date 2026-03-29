```mermaid
classDiagram
    class Vector {
        _n : int
        _data : list
        from_list(values) Vector
        dot(other) float
        norm() float
        copy() Vector
        to_list() list
        size : int
    }

    class DenseMatrix {
        _rows : int
        _cols : int
        _data : list
        from_list(data_2d) DenseMatrix
        identity(n) DenseMatrix
        matmul(other) DenseMatrix
        mat_vec(v) Vector
        transpose() DenseMatrix
        solve(rhs) Vector
        is_symmetric(tol) bool
        copy()
        to_list()
        format_fixed(width, decimals) str
        rows : int
        cols : int
        shape : tuple
    }

    class SymmetricMatrix {
        _n : int
        _size : int
        _data : list
        add_value(i, j, value)
        mat_vec(v) Vector
        solve(rhs) Vector
        to_dense() DenseMatrix
        size : int
        shape : tuple
        storage_count : int
    }

    class BandedSymmetricMatrix {
        _n : int
        _hbw : int
        _band : list
        _factored : bool
        _L_band : list
        add_value(i, j, value)
        factorise()
        solve(rhs) Vector
        mat_vec(v) Vector
        to_dense() DenseMatrix
        size : int
        half_bandwidth : int
        shape : tuple
        storage_count : int
    }

    class SkylineMatrix {
        _n : int
        _col_height : list
        _col_start : list
        _data : list
        _factored : bool
        from_dof_map(n, element_dof_lists) SkylineMatrix
        add_value(i, j, value)
        factorise()
        solve(rhs) Vector
        mat_vec(v) Vector
        to_dense() DenseMatrix
        size : int
        shape : tuple
        storage_count : int
        col_heights : list
    }

    class SparseMatrix {
        _n_rows : int
        _n_cols : int
        _coo_rows : list
        _coo_cols : list
        _coo_vals : list
        _row_ptr : list
        _col_idx : list
        _values : list
        _finalised : bool
        add_value(i, j, value)
        add_value_symmetric(i, j, value)
        finalise()
        mat_vec(v) Vector
        to_dense() DenseMatrix
        rows : int
        cols : int
        shape : tuple
        nnz : int
        storage_count : int
    }

    DenseMatrix ..> Vector : uses
    SymmetricMatrix ..> Vector : uses
    SymmetricMatrix ..> DenseMatrix : to_dense
    BandedSymmetricMatrix ..> Vector : uses
    BandedSymmetricMatrix ..> DenseMatrix : to_dense
    SkylineMatrix ..> Vector : uses
    SkylineMatrix ..> DenseMatrix : to_dense
    SparseMatrix ..> Vector : uses
    SparseMatrix ..> DenseMatrix : to_dense


```

# 2D Frame Analysis UML

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
        +list[NodeCoord] nodes
        +list[MaterialProp] materials
        +list[ElementConn] elements
        +list[Support] supports
        +list[NodalLoad] loads
        +int num_nodes
        +int num_elements
        +__init__(xy, materials, connectivity, supports, loads)
    }

    FrameModel o-- NodeCoord : nodes
    FrameModel o-- MaterialProp : materials
    FrameModel o-- ElementConn : elements
    FrameModel o-- Support : supports
    FrameModel o-- NodalLoad : loads

    class element {
        <<module>>
        +compute_element_length()
        +local_stiffness_matrix()
        +rotation_matrix()
        +global_stiffness_matrix()
        +element_end_forces()
    }
    class assembler {
        <<module>>
        +assign_equation_numbers()
        +element_dof_vector()
        +compute_half_bandwidth()
        +assemble_global_system()
    }
    class solver {
        <<module>>
        +solve_system()
    }
    class postprocessor {
        <<module>>
        +extract_nodal_displacements()
        +compute_member_forces()
    }
    class file_io {
        <<module>>
        +read_input_file()
        +write_output_file()
        +write_sample_input_file()
    }
    class main {
        <<module>>
        +run_sample_structure()
        +run_from_file()
        +run_analysis()
        +check_global_equilibrium()
    }

    assembler ..> element : depends
    postprocessor ..> element : depends
    postprocessor ..> assembler : depends
    main ..> element : depends
    main ..> assembler : depends
    main ..> solver : depends
    main ..> postprocessor : depends
    main ..> file_io : depends
    solver ..> BandedSymmetricMatrix : external dependency

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
