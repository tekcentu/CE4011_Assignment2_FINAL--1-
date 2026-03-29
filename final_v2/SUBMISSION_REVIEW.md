# Homework Compliance Review

This note summarizes how the code package aligns with the homework requirements and what was improved before packaging.

## Improvements made in this cleaned version

1. Added a `README.md` with run instructions, folder summary, and homework-fit notes.
2. Improved `frame_analysis/file_io.py` so the parser:
   - respects explicit IDs in the text file,
   - normalises out-of-order node/material IDs,
   - validates duplicate IDs and bad references,
   - accepts optional record IDs in `SUPPORTS` and `LOADS`.
3. Added `tests/` with:
   - portal-frame verification tests,
   - parser-normalisation tests.

## Requirement-by-requirement check

### Q1 — Matrix library

- Object-oriented matrix library: **Yes**
- Handles required matrix operations: **Yes**
- Uses symmetry to reduce storage: **Yes**
- Implements at least one required storage / solution scheme: **Yes**
  - Banded symmetric storage + Cholesky solver
  - Skyline storage is also included
- No NumPy or similar in the custom matrix library: **Yes**

### Q2 — Structural analysis program

- Uses custom matrix library for matrix operations and solve: **Yes**
- Modular source files: **Yes**
- Consistent naming convention: **Yes (mostly snake_case with standard structural-analysis symbols such as K, F, D, E, G, k)**
- Subroutines include documentation sections for purpose, inputs, outputs, assumptions, units: **Yes for public-facing analysis routines and matrix methods**
- Input and output formats are described: **Yes**
- Verification included: **Yes**
  - E matrix
  - G vectors
  - element stiffness matrices
  - reduced global K
  - load vector F
  - displacement vector D
  - equilibrium / residual checks

