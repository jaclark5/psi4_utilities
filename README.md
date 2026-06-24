Psi4 Utilities
==============================
<!-- [//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/psi4_utilities/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/psi4_utilities/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/Psi4 Utilities/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/Psi4 Utilities/branch/main)
-->

Utilities for Psi4 calculations, including basis set querying, manipulation, and memory estimation.

## Features

### 1. Basis Set Information Queries (`bse_utilities.py`)
Query and filter basis sets from the [Basis Set Exchange](https://www.basissetexchange.org/) (BSE) by supported elements.
Find auxiliary basis sets corresponding to orbital basis prefixes.

**Use case examples**: See module docstring in [`bse_utilities.py`](psi4_utilities/bse_utilities.py)  
**Tests**: [`test_psi4_utilities.py`](psi4_utilities/test_bse_utilities.ipynb)

### 2. Basis Set File Operations (`separate_basis_set.py`)
Download basis sets from BSE, extract augmented/diffuse primitives, and save in Psi4 .gbs format.
Useful for separating core and augmented basis sets.

**Use case examples**: See module docstring and typical workflow in [`separate_basis_set.py`](psi4_utilities/separate_basis_set.py)  
**Tests and demo**: [`test_separate_basis_set.ipynb`](psi4_utilities/test_separate_basis_set.ipynb)

### 3. Memory Estimation (Experimental ⚠️)
Estimate memory requirements for Psi4 calculations. **These estimates are experimental and should not be trusted, at all, for production use.**

Supported methods:
- `scf_memory.py` — Hartree-Fock / Self-Consistent Field
- `cc_memory.py` — Coupled Cluster (CC, CCSD, CCSD(T))
- `mp2_memory.py` — Møller-Plesset second-order perturbation theory
- `mr_memory.py` — Multi-reference methods

**Tests and demo**: [`test_memory_estimate.ipynb`](psi4_utilities/test_memory_estimate.ipynb)

### Copyright

Copyright (c) 2025, Jennifer A Clark


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.10.
