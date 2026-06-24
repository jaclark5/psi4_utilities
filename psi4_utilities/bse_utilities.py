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

**Get basis sets with ECPs for transition metals:**
    >>> basis_sets_with_ecp = get_basis_sets(elements=["Fe"], has_ecp=True)
    >>> basis_sets_no_ecp = get_basis_sets(elements=["Fe"], has_ecp=False)

**Print details about basis sets (verbose mode):**
    >>> basis_sets = get_basis_sets(elements=["C", "H"], verbose=True)
    ✓ cc-pVDZ
      Description: Dunning's correlation consistent basis set
      Shells: 9 | Primitives: 24 (C:5s/14p, H:2s/5p)
    ✓ cc-pVTZ
      Shells: 12 | Primitives: 39 (C:7s/23p, H:3s/14p)
    
    >>> basis_sets = get_basis_sets(elements=["Fe"], has_ecp=True, verbose=True)
    ✓ cc-pV5Z-PP
      Shells: 18 | Primitives: 52 (Fe:18s/52p)
      ECPs: Fe(28e)
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


def get_basis_sets(elements=None, has_ecp=None, verbose=False):
    """Retrieve all basis sets that support a given set of elements.

    Filters basis sets from the Basis Set Exchange, excluding fitting sets,
    and returns only those that support all specified elements.

    Parameters
    ----------
    elements : list, optional
        A list of atomic numbers (int) or element symbols (str).
        If None, returns all available basis sets. Default is None.
    has_ecp : bool, optional
        Filter by ECP (effective core potential) availability:
        - True: only basis sets with ECPs for at least one element
        - False: only basis sets without any ECPs
        - None: all basis sets regardless of ECP status. Default is None.
    verbose : bool, optional
        If True, print details about each basis set found including name,
        description, number of basis functions, and ECP information.
        Default is False.

    Returns
    -------
    list[str]
        Basis set names that support all specified elements and match ECP filter.
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

        # Fetch basis without element filtering to avoid KeyError
        basis_data = bse.get_basis(basis_name)
        
        # Check if this basis has the requested elements
        basis_elements = basis_data.get("elements", {})
        if isinstance(basis_elements, dict):
            available_elements = set(str(k) for k in basis_elements.keys())
        else:
            # Handle list format [Z, {...}, Z, {...}, ...]
            available_elements = set(str(basis_elements[i]) for i in range(0, len(basis_elements), 2))
        
        # Check if all requested elements are available
        if elements is not None:
            atomic_nums_set = set(str(z) for z in atomic_nums) if atomic_nums else set()
            if not atomic_nums_set.issubset(available_elements):
                continue  # Skip if not all requested elements are in this basis
        
        # Filter by ECP status if requested
        if has_ecp is not None:
            # Check if any element has ECP potentials
            has_any_ecp = False
            if isinstance(basis_elements, list):
                for i in range(1, len(basis_elements), 2):
                    if isinstance(basis_elements[i], dict) and basis_elements[i].get("ecp_potentials"):
                        has_any_ecp = True
                        break
            else:
                for elem_Z, elem_data in basis_elements.items():
                    if elem_data.get("ecp_potentials"):
                        has_any_ecp = True
                        break
            
            if has_ecp and not has_any_ecp:
                continue  # Requested ECPs but none found
            if not has_ecp and has_any_ecp:
                continue  # Requested no ECPs but some found
        
        supported_basis_sets.append(basis_name)
        
        # Print verbose information if requested
        if verbose:
            description = basis_data.get("basis_set_description", "")
            element_basis_count = {}
            ecp_elements = []
            
            # Convert requested atomic numbers to strings for comparison
            requested_atomic_nums = set(str(z) for z in atomic_nums) if elements is not None and atomic_nums else None
            
            # Count basis function shells and primitives per element, and find ECPs
            elements_data = basis_data.get("elements", {})
            
            # Handle both dict and list formats for elements
            if isinstance(elements_data, list):
                # Some basis sets (e.g., LANL2DZ) use list format: [Z, {...}, Z, {...}, ...]
                for i in range(0, len(elements_data), 2):
                    if i + 1 < len(elements_data):
                        elem_Z = elements_data[i]
                        elem_data = elements_data[i + 1]
                        if isinstance(elem_data, dict):
                            # Use electron_shells which is the actual structure in BSE
                            electron_shells = elem_data.get("electron_shells", [])
                            num_shells = len(electron_shells)
                            
                            # Count total primitives (exponents) across all shells
                            num_primitives = sum(
                                len(shell.get("exponents", []))
                                for shell in electron_shells
                            )
                            
                            element_basis_count[elem_Z] = (num_shells, num_primitives)
                            
                            # Check for ECPs and extract electron count, but only for requested elements
                            if requested_atomic_nums is None or str(elem_Z) in requested_atomic_nums:
                                ecp_potentials = elem_data.get("ecp_potentials", {})
                                if ecp_potentials:
                                    # Get electron count from ecp_electrons field
                                    ecp_electron_count = elem_data.get("ecp_electrons")
                                    ecp_elements.append((elem_Z, ecp_electron_count))
            else:
                # Standard dict format
                for elem_Z, elem_data in elements_data.items():
                    # Use electron_shells which is the actual structure in BSE
                    electron_shells = elem_data.get("electron_shells", [])
                    num_shells = len(electron_shells)
                    
                    # Count total primitives (exponents) across all shells
                    num_primitives = sum(
                        len(shell.get("exponents", []))
                        for shell in electron_shells
                    )
                    
                    element_basis_count[elem_Z] = (num_shells, num_primitives)
                    
                    # Check for ECPs and extract electron count, but only for requested elements
                    if requested_atomic_nums is None or str(elem_Z) in requested_atomic_nums:
                        ecp_potentials = elem_data.get("ecp_potentials", {})
                        if ecp_potentials:
                            # Get electron count from ecp_electrons field
                            ecp_electron_count = elem_data.get("ecp_electrons")
                            ecp_elements.append((elem_Z, ecp_electron_count))
            
            print(f"\n{basis_name}")
            if description:
                print(f"  Description: {description}")
            if element_basis_count:
                total_shells = sum(shells for shells, _ in element_basis_count.values())
                total_primitives = sum(prims for _, prims in element_basis_count.values())
                print(f"  Shells: {total_shells} | Primitives: {total_primitives}", end="")
                if len(element_basis_count) <= 5:
                    details = ", ".join(
                        f"{z}:{shells}s/{prims}p"
                        for z, (shells, prims) in sorted(element_basis_count.items())
                    )
                    print(f" ({details})")
                else:
                    print(f" across {len(element_basis_count)} elements")
            if ecp_elements:
                ecp_info = []
                for elem_Z, electron_count in sorted(ecp_elements):
                    symbol = pt_elements[int(elem_Z)].symbol
                    if electron_count is not None:
                        ecp_info.append(f"{symbol}({electron_count}e)")
                    else:
                        ecp_info.append(symbol)
                print(f"  ECPs: {', '.join(ecp_info)}")
                

        
    return supported_basis_sets