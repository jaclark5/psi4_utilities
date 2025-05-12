import psi4

from . import psi4_utilities

_dft_methods = [
    'b2plyp', 'b2plyp-d', 'b2plyp-d3', 'b2plyp-d3bj', 'b3lyp', 
    'b3lyp-chg', 'b3lyp-d', 'b3lyp-d1', 'b3lyp-d3', 'b3lyp-d3bj', 
    'b3_x', 'b88_x', 'b97-0', 'b97-1', 'b97-2', 
    'b97-d', 'b97-d3', 'b97-d3bj', 'blyp', 'blyp-d', 
    'blyp-d1', 'blyp-d3', 'blyp-d3bj', 'bp86', 'bp86-d', 
    'bp86-d1', 'bp86-d3', 'bp86-d3bj', 'dsd-blyp', 'dsd-pbep86', 
    'dsd-pbepbe', 'ft97', 'ft97b_x', 'ft97_c', 'hcth', 
    'hcth120', 'hcth120-d3', 'hcth120-d3bj', 'hcth147', 'hcth407', 
    'hf+d', 'lyp_c', 'm05', 'm05-2x', 'm05-2x-d3', 
    'm05-d3', 'p86_c', 'pbe', 'pbe-d', 'pbe-d1', 
    'pbe-d3', 'pbe-d3bj', 'pbe0', 'pbe0-2', 'pbe0-d', 
    'pbe0-d3', 'pbe0-d3bj', 'pbesol_x', 'pbe_c', 'pbe_x', 
    'pw91', 'pw91_c', 'pw91_x', 'pw92_c', 'pz81_c', 
    'rpbe_x', 'sogga', 'sogga_x', 'svwn', 's_x', 
    'vwn3rpa_c', 'vwn3_c', 'vwn5rpa_c', 'vwn5_c', 'dldf', 
    'dldf+d', 'dldf+d09', 'wb88_x', 'wb97', 'wb97x', 
    'wb97x-2(lp)', 'wb97x-2(tqz)', 'wb97x-d', 'wblyp', 'wpbe', 
    'wpbe0', 'wpbe_x', 'wpbesol', 'wpbesol0', 'wpbesol_x', 
    'wsvwn', 'ws_x',
    'pbeh-3c', 'b97-3c', 'r2scan-3c', 'wB97X-3c',
]

def get_auxiliary_basis_size(qce_mol, aux_basis_name):
    """
    Determine the number of auxiliary functions for a given auxiliary basis set.

    Args:
        qce_mol (psi4.core.Molecule): Psi4 molecule object.
        aux_basis_name (str): Name of the auxiliary basis set (e.g., 'cc-pVDZ-RI').

    Returns:
        int: Number of auxiliary functions (naux).
    """
    # Build the auxiliary basis set
    aux_basis = psi4.core.BasisSet.build(qce_mol, "DF_BASIS_SCF", aux_basis_name, puream=0)
    
    # Get the number of auxiliary functions
    naux = aux_basis.nbf()
    return naux


def get_integral_memory(nbasis, int_type="df", naux=None):
    """_summary_

    Args:
        nbasis (_type_): _description_
        int_type (str, optional): _description_. Defaults to "df".

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    
    if int_type == 'pk':
        # Full integrals stored in memory (nbasis⁴ scaling)
        integral_memory = nbasis**4 * 8
    elif int_type == 'df':
        # Density fitting reduces memory usage (nbasis² × naux scaling)
        if naux is None:
            raise ValueError("The number of auxiliary functions is needed.")
        integral_memory = nbasis**2 * naux * 8
    elif int_type == 'cd':
        # Cholesky decomposition reduces memory usage (nbasis² × ncholesky scaling)
        ncholesky = nbasis * 2  # Approximation for Cholesky vectors
        integral_memory = nbasis**2 * ncholesky * 8
    elif int_type == 'out_of_core':
        # Minimal memory for integrals, stored on disk
        integral_memory = 0  # Assume no memory for integrals
    elif int_type == 'direct':
        # On-the-fly computation, no integrals stored
        integral_memory = 0  # Assume no memory for integrals
    elif int_type == 'conv':
        # Full integrals stored in memory (nbasis⁴ scaling)
        integral_memory = nbasis**4 * 8
    else:
        raise ValueError(f"Unsupported scf_type/mp2_type: {int_type}")

    return integral_memory
    

def memory_from_psi4(qce_mol, method="scf", basis=None, scf_type=None, mp2_type=None, freeze_core=False, aux_type=None):
    """Estimate the memory required for a given Psi4 calculation (SCF, MP2, or DFT).

    Args:
        qce_mol (qcelemental.models.Molecule): QCElemental molecule
        method (str, optional): Energy method name, e.g., 'scf', 'mp2', 'ccsd', 'b3lyp'... Defaults to "scf".
        basis (_type_, optional): Basis set name. Defaults to None.
        scf_type (str, optional): SCF type (e.g., 'df'). Defaults to None.
        mp2_type (str, optional): MP2 type (e.g., 'df'). Defaults to None.
        freeze_core (bool, optional): Whether to freeze core orbitals. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        float: Estimated memory in MB.
    """
    psi4_mol = psi4.geometry(qce_mol.to_string(dtype='psi4'))

    options = {
        key: value for key, value in {
            'basis': basis,
            'scf_type': scf_type,
            'mp2_type': mp2_type,
            'freeze_core': freeze_core,
            'print': 5
        }.items() if value is not None
    }
    psi4.set_options(options)
    
    try:
        method = method.lower()
        _, wfn = psi4.energy(method, return_wfn=True)
    except psi4.ValidationError as e:
        raise ValueError(f"Unsupported method '{method}' or invalid Psi4 configuration.") from e

    # Extract orbital info
    nmo = wfn.nmo()
    nocc = wfn.nalpha()
    if freeze_core:
        nocc -= wfn.ncore()  # Subtract the number of frozen core orbitals
    nvirt = nmo - nocc
    nbasis = wfn.basisset().nbf()  # Number of basis functions

    if aux_type is None:
        aux_types = [psi4_utilities.find_aux_sets(basis, )]
        raise ValueError(f"Specify aux_type as one of the following: {aux_types}")
    if scf_type == 'df':
        aux_basis_role = "DF_BASIS_SCF"
    elif scf_type == 'cd':
        aux_basis_role = "CHOL_BASIS"
    else:
        aux_basis_role = None  # No auxiliary basis set used
    
    if aux_basis_role is not None:
        aux_basis_name = f"{basis}-{aux_type}"
        aux_basis = psi4.core.BasisSet.build(qce_mol, aux_basis_role, aux_basis_name)
        naux = aux_basis.nbf()  # Get the number of auxiliary functions
    else:
        naux = 0
            
    if method.lower() in ['scf', 'dft', 'hf-3c',] + _dft_methods:
        # Memory for Fock matrix and density matrix (nmo² scaling)
        fock_memory = nmo**2 * 8
        integral_memory = get_integral_memory(nbasis, int_type=scf_type, naux=naux)
        total_memory = fock_memory + integral_memory
    elif method.lower() in ['mp2', 'df-mp2', 'conv-mp2', 'sos-mp2', 'scs-mp2']:
        # Memory for MP2 amplitudes (O²V² scaling)
        amp_memory = nocc**2 * nvirt**2 * 8
        integral_memory = get_integral_memory(nbasis, int_type=mp2_type, naux=naux)
        total_memory = amp_memory + integral_memory
    elif method.lower() in ['mp3', 'sos-mp3', 'scs-mp3']:
        # Memory for MP3 amplitudes (O³V³ scaling)
        amp_memory = nocc**3 * nvirt**3 * 8
        integral_memory = get_integral_memory(nbasis, int_type=mp2_type, naux=naux)
        total_memory = amp_memory + integral_memory
    elif method.lower() in ['omp2']:
        # Memory for OMP2 amplitudes (O²V² scaling) + orbital optimization
        amp_memory = nocc**2 * nvirt**2 * 8
        integral_memory = get_integral_memory(nbasis, int_type=mp2_type, naux=naux)
        orbital_grad_memory = nmo**2 * 8  # Orbital gradients and Hessians
        total_memory = amp_memory + max(integral_memory, orbital_grad_memory)
    elif method.lower() in ['ccsd', 'ccsd(t)', 'cc3', 'qcisd', 'cc2', 'bccd', 'qcisd(t)']:
        # Memory for CCSD amplitudes (O²V⁴ scaling)
        amp_memory = nocc**2 * nvirt**4 * 8
        integral_memory = get_integral_memory(nbasis, int_type=scf_type, naux=naux)
        total_memory = amp_memory + integral_memory

    # Convert to MB
    estimated_MB = total_memory / (1024**2)  # Convert bytes to MB

    print(f"Exact {method.upper()} memory: ~{estimated_MB:.2f} MB (assuming doubles precision)")
    return estimated_MB
    # For df (the default), memory is lower, but you can still use this as a worst-case estimate.