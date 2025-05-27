import warnings
import io
import contextlib

import psi4
import numpy as np

psi4.core.be_quiet()

_DFT_METHODS = [
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
    'pbeh3c', 'b973c', 'r2scan3c', 'wb97x3c', 'pbeh-3c', 'b97-3c', 'r2scan-3c', 'wb97x-3c',
]
_SCF_METHODS = ['scf', 'dft', "hf", 'hf3c', 'hf-3c'] + _DFT_METHODS
_CC_METHODS = ['ccsd', 'ccsd(t)', 'cc3', 'qcisd', 'cc2', 'bccd', 'qcisd(t)']
_MP2_METHODS = ['mp2', 'df-mp2', 'conv-mp2', 'sos-mp2', 'scs-mp2', 'omp2', 'mp3', 'sos-mp3', 'scs-mp3']


def get_calculation_type(method):
    """Return the general calculation type (i.e., CC, MP, or SCF) based on the energy method.

    Args:
        method (str): Psi4 energy method

    Returns:
        str: Energy method type, either SCF, CC, or MP2
    """
    if method.lower() in _SCF_METHODS:
        return "SCF"
    elif method.lower() in _MP2_METHODS:
        return "MP2"
    elif method.lower() in _CC_METHODS:
        return "CC"
    else:
        raise ValueError(f"The method, {method}, is not recognized. Consider adding this option to `get_calculation_type`")


def get_auxiliary_basis_size(molecule, primary_basis, method_type=None, method=None, aux_basis_name=None):
    """
    Get the number of auxiliary basis functions for a given molecule, primary basis, and method.
    
    Parameters:
    -----------
    molecule (psi4.core.Molecule): The molecule object
    primary_basis (str): Name of the primary basis set
    method_type (str, optional): Type of method (e.g., 'DF', 'CD', etc.). If None, will use the Psi4 default
    method (str, optional): Name of the method (e.g., 'SCF', 'B3LYP', 'MP2', 'MP3', 'OMP2', 'CCSD')
    aux_basis_name (str): Name of auxiliary basis set
    
    Returns:
    --------
    dict
        Dictionary containing number of auxiliary basis functions and basis set details
    """
    result = {}

    calculation_type = get_calculation_type(method)
    result["calculation_type"] = calculation_type
    
    # If no method_type is specified, check Psi4's current options
    if method_type is None:
        option_name = calculation_type + "_TYPE"
        method_type = psi4.core.get_option(calculation_type, option_name)
    
    # Determine key and fitrole based on calculation type and method
    if method_type.upper() in ['DF', 'DENSITY_FITTED']:
        if calculation_type.upper() == 'SCF':
            key = "DF_BASIS_SCF"
            fitrole = "JKFIT"
        elif calculation_type.upper() == 'MP2':
            key = "DF_BASIS_MP2"
            fitrole = "RIFIT"
        elif calculation_type.upper() == 'CC':
            key = "DF_BASIS_CC"
            fitrole = "RIFIT"
        else:
            key = "DF_BASIS_SCF"  # Default to SCF for unknown types
            fitrole = "JKFIT"
    else:
        result["naux"] = 0
        result["aux_basis_used"] = "None"
        return result
   
    if aux_basis_name is None:
        aux_basis_obj = psi4.core.BasisSet.build(molecule, key, "", fitrole, primary_basis)
        aux_basis_name = aux_basis_obj.name()
    else:
        aux_basis_obj = psi4.core.BasisSet.build(molecule, key, aux_basis_name)
    
    # Get the number of auxiliary basis functions
    result["naux"] = aux_basis_obj.nbf()
    result["aux_basis_used"] = aux_basis_name
    result["key_used"] = key
    result["fitrole_used"] = fitrole
    
    return result


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


def get_orbital_counts( mol: psi4.core.Molecule, basis: str, frozen_core: bool = True,  use_wfn: bool = False, method: str = None, ):
    """Computes and returns the number of molecular orbitals (MOs), alpha and beta electrons,
    frozen core orbitals, total occupied orbitals, correlated occupied orbitals, and virtual orbitals
    for a given molecule and basis set, with options for frozen core and MP2 type.

    Args:
        mol (psi4.core.Molecule): The Psi4 molecule object for which orbital counts are to be determined.
        basis (str): The basis set to use for the calculation (e.g., 'cc-pVDZ').
        frozen_core (bool, optional): Whether to freeze core orbitals in post-HF calculations. Defaults to True.
        use_wfn (bool, optional): Choose whether to use the wave function or not
        method (str, optional): Method used to classify the calculation type. Defaults to None

    Returns:
        dict: A dictionary containing:

            - "nbasis" (int): Total number of basis functions.
            - "nocc" (int): Number of correlated (non-core) occupied orbitals.
            - "nvirt" (int): Number of virtual orbitals.

    """

    calc_type = get_calculation_type(method)
    if calc_type == "CC" or calc_type == "MP2" or method.lower() not in _DFT_METHODS:
        reference = "rhf" if mol.multiplicity() == 1 else "rohf"
    else: # DFT Method
        reference = "rks" if mol.multiplicity() == 1 else "uks"
    options = {"reference": reference}

    flag_3c = method[-2:].lower() == "3c"
    if not flag_3c:
        options.update({"basis": basis})
    else:
        if not use_wfn:
            warnings.warn("WFN must be used for estimating memory of *3c energy method.")
            use_wfn = True
            
    psi4.core.clean_options()
    psi4.set_options(options)


    if use_wfn:
        # Run HF to get orbitals and wavefunction
        psi_output = io.StringIO()
        with contextlib.redirect_stdout(psi_output), contextlib.redirect_stderr(psi_output):
            try:
                if not flag_3c:
                    _, wfn = psi4.energy("scf", molecule=mol, return_wfn=True, frozen_core=frozen_core)
                else:
                    _, wfn = psi4.energy(method, molecule=mol, return_wfn=True, frozen_core=frozen_core)
            except Exception as e:
                output = psi_output.getvalue()
                raise RuntimeError(f"Psi4 calculation failed: {output}") from e
        output = psi_output.getvalue()
        if "Traceback" in output or "PsiException" in output or "Fatal Error" in output:
            raise RuntimeError(f"Psi4 calculation error detected:\n{output}")
        
        nbasis = wfn.basisset().nbf()
        if mol.multiplicity() == 1: # RHF
            nocc = wfn.nalpha()
            nvirt = wfn.nmo() - nocc
        else: # rohf for transition metals to avoid spin contamination 
            nocc = wfn.nalpha()
            nvirt = wfn.nmo() - nocc
    else:
        if reference in ["rks", "uks"]:
            basisset = psi4.core.BasisSet.build(mol, "ORBITAL", basis, puream=psi4.core.get_global_option("PUREAM"))
        elif reference in ["rhf", "uhf", "rohf"]:
            basisset = psi4.core.BasisSet.build(mol, "BASIS", basis, puream=psi4.core.get_global_option("PUREAM"))
        else:
            # Default to "BASIS" if unknown reference
            basisset = psi4.core.BasisSet.build(mol, "BASIS", basis, puream=psi4.core.get_global_option("PUREAM"))
        nbasis = basisset.nbf()
        
        nelectrons = sum(mol.Z(ii) for ii in range(mol.natom()))
        nelectrons -= int(mol.molecular_charge())
        nsocc = mol.multiplicity() - 1
        nalpha = (nelectrons + nsocc) // 2
        
        if nsocc == 0: # RHF
            nocc = nelectrons // 2
            nvirt = nbasis - nocc # Assume nmo = nbasis
        else: # Only consider ROHF
            nocc = nalpha
            nvirt = nbasis - nalpha # Assume nmo = nbasis

    return {
        "nbasis": nbasis,
        "nocc": nocc,
        "nvirt": nvirt,
    }


def memory_from_psi4(
    qce_mol,
    method="scf",
    basis=None,
    scf_type=None,
    mp2_type=None,
    cc_type=None,
    frozen_core=False,
    use_wfn=False,
    return_counts=False,
    ):
    """Calculate the maximum memory required for a Psi4 calculation.

    Args:
        qce_mol (qcelemental.models.Molecule): QCElemental molecule
        method (str, optional): Energy method name, e.g., 'scf', 'mp2', 'ccsd', 'b3lyp'... Defaults to "scf".
        basis (_type_, optional): Basis set name. Defaults to None.
        scf_type (str, optional): SCF type (e.g., 'df'). Defaults to None.
        mp2_type (str, optional): MP2 type (e.g., 'df'). Defaults to None.
        cc_type (str, optional): CC type (e.g., 'df'). Defaults to None.
        frozen_core (bool, optional): (For Psi4 1.10) Specifies how many core orbitals to freeze in correlated computations. 
        TRUE or 1 will default to freezing the previous noble gas shell on each atom. In case of positive charges
        on fragments, an additional shell may be unfrozen, to ensure there are valence electrons in each fragment.
        With FALSE or 0, no electrons are frozen (with the exception of electrons treated by an ECP). With -1, -2,
        and -3, the user might request strict freezing of the previous first/second/third noble gas shell on every
        atom. In this case, when there are no valence electrons, the code raises an exception. Defaults to False.
        use_wfn (bool, optional): Choose whether to use the wave function or not
        return_counts (bool, optional): If True, in addition to the memory usage, a dictionary of basis sets will be provided.

    Returns:
        float: Estimated memory in GB.
    """
    
    flag_3c = True if method[-2:].lower() == "3c" else False
    if not flag_3c and (basis is None or basis == "None"):
        raise ValueError("No basis set was provided.")

    psi4_mol = psi4.geometry(qce_mol.to_string(dtype='psi4'))
    orbital_counts = get_orbital_counts(
        psi4_mol,
        basis,
        frozen_core=frozen_core,
        use_wfn=use_wfn,
        method=method,
    )
    
    method_type = scf_type if scf_type is not None else mp2_type
    method_type = method_type if method_type is not None else cc_type
    if flag_3c and method_type.upper() in ['DF', 'DENSITY_FITTED']:
        warnings.warn("3c method do not apply density fitting.")
        aux_info = {"naux": 0, "aux_basis_used": "None"}
    else:
        aux_info = get_auxiliary_basis_size(
            psi4_mol, 
            basis, 
            method_type=None, 
            method=method,
        )
            
    if method.lower() in _SCF_METHODS:
        # Memory for Fock matrix and density matrix (nmo² scaling)
        fock_memory = orbital_counts["nbasis"]**2 * 8
        integral_memory = get_integral_memory(orbital_counts["nbasis"], int_type=scf_type, naux=aux_info["naux"])
        max_memory = max(fock_memory, integral_memory)
    elif method.lower() in ['mp2', 'df-mp2', 'conv-mp2', 'sos-mp2', 'scs-mp2']:
        # Memory for MP2 amplitudes (O²V² scaling)
        amp_memory = orbital_counts["nocc"]**2 * orbital_counts["nvirt"]**2 * 8  # MP2 amplitudes (O²V² scaling)
        integral_memory = get_integral_memory(orbital_counts["nbasis"], int_type=mp2_type, naux=aux_info["naux"])
        max_memory = max(amp_memory, integral_memory)
    elif method.lower() in ['mp3', 'sos-mp3', 'scs-mp3']:
        # Memory for MP3 amplitudes (O³V³ scaling)
        amp_memory = orbital_counts["nocc"]**3 * orbital_counts["nvirt"]**3 * 8
        integral_memory = get_integral_memory(orbital_counts["nbasis"], int_type=mp2_type, naux=aux_info["naux"])
        max_memory = max(amp_memory, integral_memory)
    elif method.lower() in ['omp2']:
        # Memory for OMP2 amplitudes (O²V² scaling) + orbital optimization
        amp_memory = orbital_counts["nocc"]**2 * orbital_counts["nvirt"]**2 * 8
        integral_memory = get_integral_memory(orbital_counts["nbasis"], int_type=mp2_type, naux=aux_info["naux"])
        orbital_grad_memory = orbital_counts["nbasis"]**2 * 8  # Orbital gradients and Hessians
        max_memory = np.max([amp_memory, integral_memory, orbital_grad_memory])
    elif method.lower() in ['ccsd', 'ccsd(t)', 'cc3', 'qcisd', 'cc2', 'bccd', 'qcisd(t)']:
        # Memory for CCSD amplitudes (O²V⁴ scaling)
        amp_memory = orbital_counts["nocc"]**2 * orbital_counts["nvirt"]**4 * 8
        integral_memory = get_integral_memory(orbital_counts["nbasis"], int_type=scf_type, naux=aux_info["naux"])
        max_memory = max(amp_memory, integral_memory)
    else:
        raise ValueError(f"Unsupported method '{method}'.")

    # Convert bytes to GB
    if return_counts:
        orbital_counts.update(aux_info)
        return max_memory / (1024**2) / 1000, orbital_counts
    else:
        return max_memory / (1024**2) / 1000