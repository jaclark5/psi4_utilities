"""Query and retrieve basis sets from Basis Set Exchange (BSE).

This module provides utilities for searching and filtering basis sets available
in the Basis Set Exchange, including finding auxiliary basis sets and retrieving
basis sets that support specific elements.

Usage
-----
**Find auxiliary basis sets for an orbital basis:**
    >>> from psi4_utilities.bse_utilities import find_aux_sets
    >>> aux_sets = find_aux_sets("cc-pVTZ")
    >>> print(aux_sets)
    ['cc-pVTZ-jkfit', 'cc-pVTZ-ri', 'cc-pVTZ-cabs']

**Get all basis sets supporting specific elements:**
    >>> from psi4_utilities.bse_utilities import get_basis_sets
    >>> basis_sets = get_basis_sets(elements=["C", "H", "O"])
    >>> basis_sets = get_basis_sets(elements=[6, 1, 8])  # Also accepts atomic numbers

Functions
---------
    - `find_aux_sets`: Find auxiliary basis sets matching an orbital basis prefix.
    - `get_basis_sets`: Retrieve all basis sets that support a given set of elements.
"""

import basis_set_exchange as bse
from periodictable import elements as pt_elements

def find_aux_sets(orbital_basis_prefix):
    """Find auxiliary basis sets matching an orbital basis prefix.

    Searches the Basis Set Exchange for auxiliary basis sets (jkfit, mp2fit, ri, cabs)
    that correspond to a given orbital basis prefix.

    Parameters
    ----------
    orbital_basis_prefix : str
        The prefix of the orbital basis set (e.g., "cc-pVTZ").

    Returns
    -------
    list[str]
        Auxiliary basis set names containing the prefix and auxiliary set keywords.
    """
    orbital_basis_prefix = orbital_basis_prefix.lower()
    all_bases = bse.get_all_basis_names()
    basis_sets = [name for name in all_bases if orbital_basis_prefix in name.lower()]
    matched_aux_sets = [name for name in basis_sets if any(tag.lower() in name.lower() for tag in ['jkfit', 'mp2fit', 'ri', 'cabs'])]
    return matched_aux_sets


def get_basis_sets(elements=None):
    """Retrieve all basis sets that support a given set of elements.

    Filters basis sets from the Basis Set Exchange, excluding fitting sets,
    and returns only those that support all specified elements.

    Parameters
    ----------
    elements : list, optional
        A list of atomic numbers (int) or element symbols (str).
        If None, returns all available basis sets. Default is None.

    Returns
    -------
    list[str]
        Basis set names that support all specified elements.
    """
    kwargs = {}
    if elements is not None:
        if all(isinstance(elem, str) and elem.isalpha() for elem in elements):
            atomic_nums = [pt_elements.symbol(elem).number for elem in elements]
        else:
            atomic_nums = [int(x) for x in elements]
        kwargs["elements"] = atomic_nums

    supported_basis_sets = []
    for basis_name in bse.get_all_basis_names():
        if "FIT" in basis_name:
            continue
        try:
            _ = bse.get_basis(basis_name, **kwargs)
            supported_basis_sets.append(basis_name)
        except Exception as e:
            continue
        
    return supported_basis_sets