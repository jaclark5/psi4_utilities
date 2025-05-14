import os
import sys
from collections import defaultdict

import basis_set_exchange as bse

def get_basis_set_file(basis_set, path="."):
    """Get basis set from Basis Set Exchange and save to a file

    Args:
        basis_set (str): Name of basis set from Basis Set Exchange
        path (str, optional): Path to which to save the file, DO NOT INCLUDE FILENAME. Defaults to ".".
    """
    
    data = bse.get_basis(basis_set, fmt='psi4', header=True, optimize_general=True)
    data_dict, header = get_basis_set_dict(data)
    data = "\n".join(header) + "\n" + form_basis_set_file(data_dict) # Round trip for formatting
    open(os.path.join(path,f"{basis_set}.gbs"), 'w').write(data)


def subtract_dict_contents(full_dict, base_dict):
    """Return a dictionary of the uncommon entries between two inputs.

    Args:
        full_dict (dict): Dictionary that is the same as ``base_dict`` plus additional values
        base_dict (dict): The base dictionary for comparison

    Raises:
        ValueError: If the ``base_dict`` contains values not present in ``full_dict`` an error is raised.

    Returns:
        dict: Dictionary of uncommon values added to the dictionary
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
    """Create an output string of a file from a basis set dictionary

    Args:
        basis_set_dict (dict[list]): Dictionary of each element and the basis function primitives that make it up.

    Returns:
        str: String of .gbs file to be save to a file.
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
    """Download a full basis set and it's base to separate additional terms.

    Args:
        full_basis_set (str): Name of basis set from Basis Set Exchange.
        base_basis_set (str): Name of basis set from Basis Set Exchange.
        path (str, optional): Path to which to save the file, DO NOT INCLUDE FILENAME. Defaults to ".".
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