import psi4
import numpy as np

from psi4_utilities import bse_utilities

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


def get_integral_memory(nbasis, int_type="df", naux=None):
    """Calculate the memory required for storing integrals based on the specified method.
    
    Args:
        nbasis (int): Number of basis functions.
        int_type (str, optional): Type of integral storage or computation method. 
            Options include:
            
            - 'pk': Full integrals stored in memory with nbasis⁴ scaling.
            - 'df': Density fitting with nbasis² × naux scaling.
            - 'cd': Cholesky decomposition with nbasis² × ncholesky scaling.
            - 'out_of_core': Minimal memory usage, integrals stored on disk.
            - 'direct': On-the-fly computation, no integrals stored.
            - 'conv': Full integrals stored in memory with nbasis⁴ scaling.
            
            Defaults to "df".
        naux (int, optional): Number of auxiliary functions, required for 'df' type. 
            Defaults to None.
            
    Raises:
        ValueError: If `naux` is not provided for 'df' type or if an unsupported 
            `int_type` is specified.
            
    Returns:
        int: Memory required for storing integrals in bytes.
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


def get_orbital_counts(mol: psi4.core.Molecule, basis: str, scf_type: str = None, mp2_type: str = None, frozen_core: bool = True):
    """Computes and returns the number of molecular orbitals (MOs), alpha and beta electrons,
    frozen core orbitals, total occupied orbitals, correlated occupied orbitals, and virtual orbitals
    for a given molecule and basis set, with options for frozen core and MP2 type.

    Args:
        mol (psi4.core.Molecule): The Psi4 molecule object for which orbital counts are to be determined.
        basis (str): The basis set to use for the calculation (e.g., 'cc-pVDZ').
        scf_type (str, optional): SCF type (e.g., 'df'). Defaults to None.
        mp2_type (str, optional): MP2 type (e.g., 'df'). Defaults to None.
        frozen_core (bool, optional): Whether to freeze core orbitals in post-HF calculations. Defaults to True.

    Returns:
        dict: A dictionary containing:
            - "nmo" (int): Total number of molecular orbitals.
            - "nalpha" (int): Number of alpha electrons.
            - "nbeta" (int): Number of beta electrons.
            - "ncore" (int): Number of frozen core orbitals.
            - "nocc_total" (int): Total number of occupied orbitals.
            - "nocc_corr" (int): Number of correlated (non-core) occupied orbitals.
            - "nvirt" (int): Number of virtual orbitals.

    """

    # Prepare molecule and settings
    psi4.set_options({
        "basis": basis,
        "scf_type": scf_type,
        "mp2_type": mp2_type,
        "reference": "rhf" if mol.multiplicity() == 1 else "uhf",
    })

    # Run HF to get orbitals
    _, wfn = psi4.energy("scf", molecule=mol, return_wfn=True, frozen_core=frozen_core)

    nbasis = wfn.basisset().nbf()
    if wfn.reference_wavefunction() == "RHF":
        nocc = wfn.nalpha()
        nvirt = wfn.nmo() - nocc
    elif wfn.reference_wavefunction() == "UHF":
        nocc = wfn.nalpha() + wfn.nbeta()
        nvirt = 2 * wfn.nmo() - nocc
    else:
        raise ValueError("Unsupported reference wavefunction type.")
    
    # Access auxiliary basis set directly
    aux_basis_name = psi4.get_option("SCF", "DF_BASIS_SCF") if psi4.has_option("SCF", "DF_BASIS_SCF") else None
    if aux_basis_name is None:
        # Try to guess auxiliary basis name from basis
        aux_basis_name = bse_utilities.guess_aux_basis(basis)
    aux_basis = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", aux_basis_name, wfn.basisset())
    naux = aux_basis.nbf()

    return {
        "nbasis": nbasis,
        "nocc": nocc,
        "nvirt": nvirt,
        "naux": naux,
        "aux_basis": aux_basis,
    }


def memory_from_psi4(
    qce_mol,
    method="scf",
    basis=None,
    scf_type=None,
    mp2_type=None,
    frozen_core=False,
    aux_type=None,
    ):
    """Calculate the maximum memory required for a Psi4 calculation.

    Args:
        qce_mol (qcelemental.models.Molecule): QCElemental molecule
        method (str, optional): Energy method name, e.g., 'scf', 'mp2', 'ccsd', 'b3lyp'... Defaults to "scf".
        basis (_type_, optional): Basis set name. Defaults to None.
        scf_type (str, optional): SCF type (e.g., 'df'). Defaults to None.
        mp2_type (str, optional): MP2 type (e.g., 'df'). Defaults to None.
        frozen_core (bool, optional): (For Psi4 1.10) Specifies how many core orbitals to freeze in correlated computations. 
        TRUE or 1 will default to freezing the previous noble gas shell on each atom. In case of positive charges
        on fragments, an additional shell may be unfrozen, to ensure there are valence electrons in each fragment.
        With FALSE or 0, no electrons are frozen (with the exception of electrons treated by an ECP). With -1, -2,
        and -3, the user might request strict freezing of the previous first/second/third noble gas shell on every
        atom. In this case, when there are no valence electrons, the code raises an exception. Defaults to False.
        aux_type (str, optional): Auxiliary basis type, e.g., RIFIT. Defaults to None.

    Returns:
        float: Estimated memory in GB.
    """
    if method[:-3].lower() == "3c":
        basis = psi4.driver.proc_table[method].get('default_basis', None)
    if basis is None:
        raise ValueError("No basis set was provided.")

    orbital_counts = get_orbital_counts(
        psi4.geometry(qce_mol.to_string(dtype='psi4')),
        basis, 
        scf_type=scf_type,
        mp2_type=mp2_type,
        frozen_core=frozen_core
    )
            
    if method.lower() in ['scf', 'dft', 'hf-3c',] + _dft_methods:
        # Memory for Fock matrix and density matrix (nmo² scaling)
        fock_memory = orbital_counts["nbasis"]**2 * 8
        integral_memory = get_integral_memory(orbital_counts["nbasis"], int_type=scf_type, naux=orbital_counts["naux"])
        max_memory = max(fock_memory, integral_memory)
    elif method.lower() in ['mp2', 'df-mp2', 'conv-mp2', 'sos-mp2', 'scs-mp2']:
        # Memory for MP2 amplitudes (O²V² scaling)
        amp_memory = orbital_counts["nocc_corr"]**2 * orbital_counts["nvirt"]**2 * 8  # MP2 amplitudes (O²V² scaling)
        integral_memory = get_integral_memory(orbital_counts["nbasis"], int_type=mp2_type, naux=orbital_counts["naux"])
        max_memory = max(amp_memory, integral_memory)
    elif method.lower() in ['mp3', 'sos-mp3', 'scs-mp3']:
        # Memory for MP3 amplitudes (O³V³ scaling)
        amp_memory = orbital_counts["nocc_corr"]**3 * orbital_counts["nvirt"]**3 * 8
        integral_memory = get_integral_memory(orbital_counts["nbasis"], int_type=mp2_type, naux=orbital_counts["naux"])
        max_memory = max(amp_memory, integral_memory)
    elif method.lower() in ['omp2']:
        # Memory for OMP2 amplitudes (O²V² scaling) + orbital optimization
        amp_memory = orbital_counts["nocc_corr"]**2 * orbital_counts["nvirt"]**2 * 8
        integral_memory = get_integral_memory(orbital_counts["nbasis"], int_type=mp2_type, naux=orbital_counts["naux"])
        orbital_grad_memory = orbital_counts["nbasis"]**2 * 8  # Orbital gradients and Hessians
        max_memory = np.max([amp_memory, integral_memory, orbital_grad_memory])
    elif method.lower() in ['ccsd', 'ccsd(t)', 'cc3', 'qcisd', 'cc2', 'bccd', 'qcisd(t)']:
        # Memory for CCSD amplitudes (O²V⁴ scaling)
        amp_memory = orbital_counts["nocc_corr"]**2 * orbital_counts["nvirt"]**4 * 8
        integral_memory = get_integral_memory(orbital_counts["nbasis"], int_type=scf_type, naux=orbital_counts["naux"])
        max_memory = max(amp_memory, integral_memory)
    else:
        raise ValueError(f"Unsupported method '{method}'.")

    # Convert bytes to GB
    return max_memory / (1024**2) / 1000