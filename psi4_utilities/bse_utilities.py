"""Provide useful function for dealing with basis set exchange."""

import basis_set_exchange as bse
from periodictable import elements as pt_elements

def find_aux_sets(orbital_basis_prefix):
    """Finds and returns auxiliary basis sets matching a given orbital basis prefix.

    This function searches through all available basis set names from the Basis Set Exchange (BSE)
    and returns those that contain the specified orbital basis prefix and are identified as auxiliary
    sets (e.g., those containing 'jkfit', 'mp2fit', 'ri', or 'cabs' in their names).

    Parameters
    ----------
        orbital_basis_prefix (str): The prefix of the orbital basis set to search for.

    Returns
    -------
        list[str]: A list of auxiliary basis set names matching the given prefix.
    """
    orbital_basis_prefix = orbital_basis_prefix.lower()
    all_bases = bse.get_all_basis_names()
    basis_sets = [name for name in all_bases if orbital_basis_prefix in name.lower()]
    matched_aux_sets = [name for name in basis_sets if any(tag.lower() in name.lower() for tag in ['jkfit', 'mp2fit', 'ri', 'cabs'])]
    return matched_aux_sets


def get_basis_sets(elements=None):
    """Get a list of basis sets on basis set exchange that support a set of elements.

    Parameters
    ----------
        elements (list, optional): A list of either atomic numbers (int) or symbols. Defaults to None.
        
    Returns
    -------
        basis_sets (list): List of basis set names.
    """
    kwargs = {}
    if elements is not None:
        if all(isinstance(elem, str) and elem.isalpha() for elem in elements):
            atomic_nums = [pt_elements.symbol(elem).number for elem in elements]
        else:
            atomic_nums = [int(x) for x in atomic_nums]
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