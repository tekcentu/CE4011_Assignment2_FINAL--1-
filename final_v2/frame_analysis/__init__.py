"""
2D Frame Analysis Program
=========================

A modular, object-oriented structural analysis program for 2D plane
frames.  Uses the custom matrix library for all matrix operations
and the banded symmetric solver for the global system.

Modules
-------
- input_data     : Data structures for nodes, elements, supports, loads
- file_io        : File-based input/output (read/write text files)
- element        : Element stiffness, rotation, and force recovery
- assembler      : Equation numbering and global assembly
- solver         : Wrapper around the banded solver
- postprocessor  : Displacement and member-force output
- main           : Driver / entry point
"""
