"""Provide the primary functions."""

import basis_set_exchange as bse

def find_aux_sets(orbital_basis_prefix):
    orbital_basis_prefix = orbital_basis_prefix.lower()
    all_bases = bse.get_all_basis_names()
    print(all_bases)
    matched_aux_sets = [name for name in all_bases if orbital_basis_prefix in name.lower() and any(tag.lower() in name.lower() for tag in ['jkfit', 'mp2fit', 'ri', 'cabs'])]
    return matched_aux_sets

def count_functions(basis_data):
    """Counts basis functions based on angular momentum.
    Input 
    
    orbital_basis = bse.get_basis('cc-pVDZ', elements=[element_Z], fmt='python', optimize_general=True)
    aux_basis = bse.get_basis('cc-pVDZ-JKFIT', elements=[element_Z], fmt='python', optimize_general=True)
    """
    count = 0
    for element_data in basis_data['elements'].values():
        for shell in element_data['electron_shells']:
            for l in shell['angular_momentum']:
                if l == 0:
                    count += 1
                elif l == 1:
                    count += 3
                elif l == 2:
                    count += 6
                elif l == 3:
                    count += 10
                elif l == 4:
                    count += 15
                else:
                    print(f"Warning: Angular momentum {l} not supported")
    return count

