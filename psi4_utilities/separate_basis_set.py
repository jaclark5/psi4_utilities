"""Separate augmented basis set primitives from a full basis set.

This module downloads basis set definitions from the Basis Set Exchange (BSE),
compares a full basis set with a base basis set, and extracts only the additional
primitives in the full set. Output files are saved in Psi4 .gbs format.

Usage
-----
**In Python:**
    >>> from psi4_utilities.separate_basis_set import separate_basis_functions
    >>> separate_basis_functions("aug-cc-pVTZ", "cc-pVTZ", path="./basis_sets")
    # Creates: aug-cc-pVTZ.gbs, cc-pVTZ.gbs, stripped-aug-cc-pVTZ.gbs

**From command line:**
    $ python -m psi4_utilities.separate_basis_set aug-cc-pVTZ cc-pVTZ
    # Creates files in current directory

Functions
---------
    - `separate_basis_functions`: Main entry point; compares two basis sets and saves results.
    - `get_basis_set_file`: Download a single basis set from BSE and save to file.
    - `get_basis_set_dict`: Parse a basis set string into a nested dictionary structure.
    - `subtract_dict_contents`: Extract primitives in full set but not in base set.
    - `form_basis_set_file`: Serialize a basis set dictionary back to .gbs format.
"""

import os
import sys
from collections import defaultdict

import basis_set_exchange as bse

def get_basis_set_file(basis_set, path="."):
    """Download a basis set from BSE and save to a .gbs file.

    Parameters
    ----------
    basis_set : str
        Name of the basis set in Basis Set Exchange.
    path : str, optional
        Directory path for output (no filename). Default is current directory.
    """
    
    data = bse.get_basis(basis_set, fmt='psi4', header=True, optimize_general=True)
    data_dict, header = get_basis_set_dict(data)
    data = "\n".join(header) + "\n" + form_basis_set_file(data_dict) # Round trip for formatting
    open(os.path.join(path,f"{basis_set}.gbs"), 'w').write(data)


def subtract_dict_contents(full_dict, base_dict):
    """Extract basis set primitives in full_dict but not in base_dict.

    Compares nested dictionaries of basis set primitives and returns only the
    entries (elements and their primitive shells) that appear in full_dict but
    not in base_dict.

    Parameters
    ----------
    full_dict : dict
        Complete basis set dictionary with all primitives.
    base_dict : dict
        Base basis set dictionary; assumed to be a subset of full_dict.

    Returns
    -------
    dict
        Dictionary containing only the primitives in full_dict that are absent in base_dict.

    Raises
    ------
    ValueError
        If base_dict contains elements or primitives not present in full_dict.
    """
    new_dict = {}

    for element, full_value in full_dict.items():
        if element not in base_dict:
            # If the element is not in the base dictionary, add it entirely
            new_dict[element] = full_value
        else:
            # Compare the substructure of the element
            base_value = base_dict[element][1]
            full_value[1].sort(key=lambda x: len(x[1]))
            new_element_data = [full_value[0], []]
            
            primitive_lookup = list(set(primitive) for shell in base_value for primitive in shell[1])
            for full_shell, full_primitives in full_value[1]:
                uncommon_primitives = []
                for primitive in full_primitives:
                    if not any(set(primitive) == set(x) for x in primitive_lookup):
                        uncommon_primitives.append(primitive)
                    else:
                        primitive_lookup.remove(set(primitive))
                if uncommon_primitives:
                    full_shell[1] = str(len(uncommon_primitives))
                    new_element_data[1].append([full_shell, uncommon_primitives])

            if new_element_data[1]:
                new_dict[element] = new_element_data

    return new_dict


def get_basis_set_dict(basis_set_data):
    """Parse a basis set string into a nested dictionary structure.

    Converts a Psi4 .gbs format basis set string into a nested dictionary
    keyed by element, with shells and their primitive exponent–coefficient pairs.

    Parameters
    ----------
    basis_set_data : str
        Basis set data in Psi4 .gbs format.

    Returns
    -------
    dict
        Dictionary mapping element symbols to [Z, shells], where shells is a list
        of [shell_info, primitives] pairs.
    list
        Header lines preceding the basis set data.
    """
    basis_set_dict = defaultdict(list)
    flag_start = False
    flag_restart = False
    current_element = None
    header = []
    for line in basis_set_data.split("\n"):
        if flag_start:
            if len(line) == 0:
                continue
        
            if line.strip() == "****":
                flag_start, flag_restart = True, True
            elif flag_restart:
                line_array = line.strip().split()
                current_element = line_array[0]
                basis_set_dict[current_element] = [line_array[1], []]
                flag_restart = False
            else:
                line_array = line.strip().split()
                if line_array[0].isalpha():
                    basis_set_dict[current_element][1].append([line_array, []])
                else:
                    basis_set_dict[current_element][1][-1][1].append(line_array)
        elif line.strip() == "****":
            header.append(line)
            flag_start, flag_restart = True, True
        else:
            header.append(line)
    return basis_set_dict, header


def form_basis_set_file(basis_set_dict):
    """Serialize a basis set dictionary to Psi4 .gbs format string.

    Parameters
    ----------
    basis_set_dict : dict
        Dictionary mapping element symbols to [Z, shells] pairs, where shells
        contain [shell_info, primitives] pairs.

    Returns
    -------
    str
        Basis set data formatted as a Psi4 .gbs file string.
    """

    file_data = []
    for element, (number, shells) in basis_set_dict.items():
        file_data.append(f"{element}     {number} ")
        for shell, primitives in shells:
            file_data.append(f"{shell[0]:<1}   {shell[1]}   {shell[2]}")
            for primitive in primitives:
                coefficient = float(primitive[0])
                exponent = float(primitive[1])
                lx = len(str(int(coefficient)))
                if lx < 7:
                    file_data.append(f"{coefficient:>15.7f}{exponent:>23.7f}        ")
                else:
                    int1, int2 = 15 + (8 - lx), 23 - (8 - lx)
                    file_data.append(f"{coefficient:>{int1}.7f}{exponent:>{int2}.7f}        ")
        file_data.append("****")
    return "\n".join(file_data)


def separate_basis_functions(full_basis_set, base_basis_set, path="."):
    """Download two basis sets, extract the difference, and save all three.

    Downloads a full basis set and base basis set from BSE, computes the difference,
    and saves three .gbs files: base, full, and augmented (difference).

    Parameters
    ----------
    full_basis_set : str
        Name of the complete basis set in Basis Set Exchange.
    base_basis_set : str
        Name of the base basis set in Basis Set Exchange (must be a subset of full).
    path : str, optional
        Directory path for output files (no filename). Default is current directory.
    """
    
    # Save starting basis set files
    get_basis_set_file(base_basis_set, path)
    get_basis_set_file(full_basis_set, path)

    # Get the uncommon basis sets from the file
    full_data = bse.get_basis(full_basis_set, fmt='psi4', header=True, optimize_general=True)
    base_data = bse.get_basis(base_basis_set, fmt='psi4', header=True, optimize_general=True)
    
    full_dict, header = get_basis_set_dict(full_data)
    base_dict, _ = get_basis_set_dict(base_data)

    new_dict = subtract_dict_contents(full_dict, base_dict)
    new_data = form_basis_set_file(new_dict)
    output = "\n".join(header) + "\n" + new_data
    open(os.path.join(path,f"stripped-{full_basis_set}.gbs"), 'w').write(output)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Two basis set names, the full and base, must be provided.")
    separate_basis_functions(sys.argv[1], sys.argv[2])