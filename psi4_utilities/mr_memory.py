"""Get the memory contributions and peak expected memory for an Multi-Reference calculation

Those contributions include:

 - Core memory (orbitals, density matrices, Fock matrices)
 - Method-specific memory (CI vectors, RDMs, MPS tensors)
 - Integral memory (two-electron integrals and transformations)
 - DIIS extrapolation memory
 - Temporary workspace memory

The peak memory usage is the sum of core memory plus the maximum of all temporary
memory contributions during the calculation.

It's important to determine the number of active orbitals with care and reason.

 Notes
 -----
 - The spin active space is defined for a CASSCF and MCSCF methods by defining the method to be "casscf(n_electrons, n_oribtals)"
 - nevpt2, ci, and fci run subsequently after a CASSCF and MCSCF calculation.
 - Similarly, dmrgscf requires (n_electron, n_orbital) notation and dmrgcaspt2 / dmrgci are corrections run after dmrgscf.
 - RASSCF is not implemented. It's a good task for a contributor!
 - Does not account for:
    - CUHF (Constrained UHF)
    - PCM/SMD
    - Memory needed for an external field is not considered
    - Stability analysis is not considered
    - SOSCF (Second-Order SCF) memory
    - Fractional occupation memory
    - Multi-threading buffer duplication

"""

import re
import warnings
from math import comb

import psi4

psi4.core.be_quiet()

BYTES = 8
CONVERSION = 1 / 1000**3  # From bytes to GB
SIZE = BYTES * CONVERSION

METHODS = [
    "casscf", # lowest memory
    "mcscf",
#    "rasscf", Need to accommodate RS1, RS2, and RS3 before opening for use.
    "nevpt2", # must be run after CASSCF
    "ci", # must be run after CASSCF
    "fci", # must be run after CASSCF
    "dmrgscf", # requires plugin PySCF
    "dmrgcaspt2", # requires plugin PySCF, run after dmrgscf
    "dmrgci", # requires plugin PySCF, run after dmrgscf
]
_CALCULATION_TYPE = {
    "casscf": "CASSCF",
    "mcscf": "CASSCF",        # synonym for casscf in Psi4
#    "rasscf": "CASSCF",       # often shares backend and settings with casscf
    "fci": "CI",
    "ci": "CI",
    "nevpt2": "NEVPT2",
    "dmrgscf": "DMRG",       # general name, backend-specific may differ
    "dmrgcaspt2": "DMRG",
    "dmrgci": "DMRG",
}
_AUX_BASIS_ROLES = {
    "casscf": ("DF_BASIS_SCF", "JKFIT"),
    "mcscf": ("DF_BASIS_SCF", "JKFIT"),
#    "rasscf": ("DF_BASIS_SCF", "JKFIT"),
    "fci": ("DF_BASIS_SCF", "JKFIT"),
    "ci": ("DF_BASIS_SCF", "JKFIT"),
    "dmrgscf": ("DF_BASIS_SCF", "JKFIT"),
    "dmrgci": ("DF_BASIS_SCF", "JKFIT"),
    "dmrgcaspt2": ("DF_BASIS_MP2", "RI"),
    "nevpt2": ("DF_BASIS_MP2", "RI"),
}


def _calculate_ci_determinants(ncas, nalpha, nbeta):
    """Calculate number of CI determinants for given active space.
    
    Parameters
    ----------
    ncas : int
        Number of active orbitals
    nalpha : int
        Number of active alpha electrons
    nbeta : int
        Number of active beta electrons
        
    Returns
    -------
    int : Number of determinants
    """
    if nalpha > ncas or nbeta > ncas or nalpha < 0 or nbeta < 0:
        return 0
    
    try:
        ndet = comb(ncas, nalpha) * comb(ncas, nbeta)
        return ndet
    except (ValueError, OverflowError):
        # For very large spaces, use approximation
        warnings.warn("Large active space detected, using approximation for determinant count")
        return min(2**(2*ncas), 10**12)  # Cap at reasonable limit


def get_core_memory(nbasis, nmo, ncas, reference, method, n_act_spin_electrons=None, bond_dim=1000):
    """Get memory used for fundamental matrices needed throughout the multi-reference procedure.

    Contributions include molecular orbitals, density matrices, and Fock matrices.
    These are persistent allocations needed throughout the multi-reference procedure.

    Parameters
    ----------
        nbasis (int): Number of basis functions
        nmo (int): Number of molecular orbitals
        ncas (int, optional): Number of active space orbitals. Required for multi-reference methods.
        For "*ci" methods, ncas should be nmo - ncore.
        reference (str): Reference wavefunction (i.e., 'UHF', 'RHF', 'ROHF')
        method (str): Multi-reference method
        n_act_spin_electrons (int, optional): Active electrons, usually valence electrons that participate in bonding, excitation, etc.
        bond_dim (int, optional): Maximum number of states kept per bond between sites for DMRG method. Default is 1000.

    Returns
    -------
        dict: Memory breakdown in GB containing:

            - 'core_memory'
            - 'orbital'
            - 'density_matrices'
            - 'fock_matrices'
            - 'method_memory'
            - 'sad_memory'

    """

    reference = reference.upper()
    method = method.lower().replace("-", "").replace("_", "").replace("(", "").replace(")", "")

    memory_breakdown = {}

    orbital_factor = 2 if reference in ["UKS", "UHF", "CUHF", "ROHF"] else 1
    dens_fock_factor = 1 if reference in ["RKS", "RHF", "ROHF"] else 2

    memory_breakdown["orbital"] = nbasis * nmo * SIZE * orbital_factor
    memory_breakdown["density_matrices"] = nbasis**2 * SIZE * dens_fock_factor
    memory_breakdown["fock_matrices"] = nbasis**2 * SIZE * dens_fock_factor

    # SAD (Superposition of Atomic Densities) guess memory
    guess_type = psi4.core.get_option("SCF", "GUESS")
    memory_breakdown["sad_memory"] = nbasis**2 * SIZE * 2 if guess_type.upper() == "SAD" else 0

    # Method-specific memory based on active space
    if method in ["fci", "ci", "casscf", "mcscf", "rasscf", "nevpt2"]:
        if n_act_spin_electrons is not None:
            nalpha_act = n_act_spin_electrons // 2 + n_act_spin_electrons % 2
            nbeta_act = n_act_spin_electrons // 2
        else:
            raise ValueError(f"The number of active spin electrons must be provided for method, {method}.")
        
    if method in ["fci", "ci"]:
        # CI vector memory: depends on number of determinants
        ndet = _calculate_ci_determinants(ncas, nalpha_act, nbeta_act)
        memory_breakdown["method_memory"] = ndet * SIZE
        memory_breakdown["ci_determinants"] = ndet
        
    elif method in ["casscf", "mcscf", "rasscf"]:
        # CI vector + density matrices in active space
        ndet = _calculate_ci_determinants(ncas, nalpha_act, nbeta_act)
        ci_memory = ndet * SIZE
        # 1, 2, 3-RDMs in active space
        rdm_memory = ncas**2 * SIZE + ncas**4 * SIZE + ncas**6 * SIZE
        memory_breakdown["method_memory"] = ci_memory + rdm_memory
        memory_breakdown["ci_determinants"] = ndet
        
    elif method.startswith("dmrg"):
        # MPS tensors: approximate as ncas sites with bond dimension
        # Each tensor ~ bond_dim^2 * d (where d=4 for spin orbitals)
        mps_memory = ncas * bond_dim**2 * 4 * SIZE
        # Environment tensors for sweeping
        env_memory = ncas * bond_dim**2 * SIZE
        memory_breakdown["method_memory"] = mps_memory + env_memory
        memory_breakdown["bond_dimension"] = bond_dim
        
    elif method == "nevpt2":
        # Requires up to 4-RDMs in active space
        rdm1_memory = ncas**2 * SIZE
        rdm2_memory = ncas**4 * SIZE  
        rdm3_memory = ncas**6 * SIZE
        rdm4_memory = ncas**8 * SIZE
        # CI vector
        ndet = _calculate_ci_determinants(ncas, nalpha_act, nbeta_act)
        ci_memory = ndet * SIZE
        memory_breakdown["method_memory"] = rdm1_memory + rdm2_memory + rdm3_memory + rdm4_memory + ci_memory
        memory_breakdown["ci_determinants"] = ndet
        
    else:
        raise ValueError(f"Multi-reference method, {method}, is not recognized.")
    
    memory_breakdown["core_memory"] = (
        memory_breakdown["orbital"]
        + memory_breakdown["density_matrices"]
        + memory_breakdown["fock_matrices"]
        + memory_breakdown["sad_memory"]
        + memory_breakdown["method_memory"]
    )

    return memory_breakdown


def get_diis_memory(nbasis, reference, diis_max_vecs=6):
    """Estimate memory used by DIIS extrapolation in multi-reference.

    DIIS stores a history of Fock and error matrices over several multi-reference iterations.
    This memory can be significant depending on the number of stored vectors and the
    reference type. This accelerates multi-reference convergence.

    Parameters
    ----------
        nbasis (int): Number of basis functions
        reference (str): Reference wavefunction (i.e., 'UHF', 'RHF', 'ROHF')
        diis_max_vecs (int,optional): Maximum number of DIIS vectors to store. Default is 6.

    Returns
    -------
        dict: Memory breakdown in GB containing the estimated DIIS memory usage.

    Notes
    -----
        - Returns 0.0 if DIIS is disabled
        - RHF and RKS use 2 matrices per iteration (Fock and error).
        - UHF, UKS, ROHF use 4 matrices per iteration (α/β Fock and α/β error).

    """

    reference = reference.upper()
    spin_factor = 2 if reference in ["UHF", "UKS", "ROHF"] else 1
    
    if psi4.core.get_option("MCSCF", "CI_DIIS") or psi4.core.get_option("DMRG", "DMRG_DIIS"):
        diis_memory = diis_max_vecs * 2 * spin_factor * nbasis**2 * SIZE
    else:
        diis_memory = 0

    return {"diis_memory": diis_memory}


def get_integral_memory(
    nbasis, nmo, ncas, ncore, reference, method, int_type="df", n_act_spin_electrons=None, naux=None, buffer_size=1000000
):
    """Calculate the memory required for storing integrals based on the specified method.

    Parameters
    ----------
        nbasis (int): Number of basis functions.
        nmo (int): Number of molecular orbitals.
        ncas (int): Number of active space orbitals. Required for multi-reference methods.
        ncore (int): Number of frozen core orbitals
        reference (str): Reference wavefunction (i.e., 'UHF', 'RHF', 'ROHF')
        method (str): Multi-reference method
        int_type (str, optional): Type of integral evaluation method in Psi4.
            Options include:

            - 'pk': Full Atomic Orbital integrals stored in memory with nbasis⁴ scaling.
            - 'df': Density fitting with nbasis² × naux scaling.
            - 'cd': Cholesky decomposition with nbasis² × ncholesky scaling.
            - 'out_of_core': Minimal memory usage, integrals stored on disk.
            - 'direct': On-the-fly computation, no integrals stored.
            - 'conv': Same as 'pk'.

            Defaults to "df".
        n_act_spin_electrons (int, optional): Active electrons, usually valence electrons that participate in bonding, excitation, etc.
        naux (int, optional): Number of auxiliary functions (required for 'df' type).
            Defaults to None.
        buffer_size (int, optional): Size of I/O buffer (used for out-of-core/direct).

    Raises
    ------
        ValueError: If required inputs (e.g., nmo, naux) are missing for the chosen int_type.

    Returns
    -------
        dict: Dictionary with:
            - 'integral_memory': Memory in bytes for integrals or intermediates.
            - 'temporary_memory': Memory in bytes for Fock build, diagonalization, transformations.

    Notes
    -----
        - Approximate Cholesky vectors as 2×nbasis.
        - Assumes worst-case temporary workspace for integral transformations and G-matrix builds.
        - Does not account for multi-threading buffer duplication.

    """

    reference = reference.upper()
    method = method.lower().replace("-", "").replace("_", "").replace("(", "").replace(")", "")
    spin_factor = 1 if reference in ["RKS", "RHF"] else 2

    temp_matrices = nbasis**2 * SIZE * spin_factor  # G matrix construction workspace
    temp_matrices += nbasis * nmo * SIZE * spin_factor
    
    if naux is None and int_type == "df":
        raise ValueError("The number of auxiliary functions is required for DF integrals.")
    
    if ncas is None:
        raise ValueError(f"The number of active spin orbitals must be provided.")

    if int_type in ["pk", "conv"]:
        # Full integrals stored in memory (nbasis⁴ scaling)
        integral_memory = nbasis**4 * SIZE
        temp_matrices += 3 * nbasis**2 * SIZE  # Fock, density, scratch
        temp_matrices += nbasis * nmo * SIZE * spin_factor
    elif int_type == "df":
        # Density fitting reduces memory usage (nbasis² × naux scaling)
        if naux is None:
            raise ValueError("The number of auxiliary functions is required.")
        integral_memory = nbasis**2 * naux * SIZE
        temp_matrices += naux**2 * SIZE  #  (P|Q) Metric matrix
        temp_matrices += nbasis * naux * SIZE  # fitting buffer
        temp_matrices += naux * nmo * SIZE * spin_factor  # MO transformed 3-index
    elif int_type == "cd":
        # Cholesky decomposition reduces memory usage (nbasis² × ncholesky scaling)
        ncholesky = nbasis * 2  # Approximation for Cholesky vectors
        integral_memory = nbasis**2 * ncholesky * SIZE
        temp_matrices += nbasis**2 * SIZE * 2  # Decomposition workspace
        temp_matrices += ncholesky * nmo * SIZE * spin_factor  # Transformed vectors
        temp_matrices += ncholesky**2 * SIZE  # (L|L) metric approx.
    elif int_type == "out_of_core":
        # Minimal memory for integrals, stored on disk
        integral_memory = buffer_size * SIZE  # I/O buffers only
        # Out-of-core: needs I/O buffers
        temp_matrices += nbasis**2 * SIZE  # Accumulation arrays
    elif int_type == "direct":
        # On-the-fly computation, no integrals stored
        integral_memory = 0  # Assume no memory for integrals
        temp_matrices += nbasis**2 * SIZE * 2  # MO transformation buffer
    else:
        raise ValueError(f"Unsupported int_type: {int_type}")

    # Multi-reference method-specific integral transformations
    # These require active space parameters  
    if method in ["casscf", "mcscf", "rasscf"]:
        # Need (mo|mo) integrals in active space
        temp_matrices += ncas**4 * SIZE  # Active space integrals
        temp_matrices += ncore * ncas**3 * SIZE  # Core-active integrals
        
    elif method.startswith("dmrg"):
        # DMRG typically needs reordered integrals
        temp_matrices += ncas**4 * SIZE
        # Additional workspace for MPO construction
        temp_matrices += ncas**2 * SIZE * 100  # Approximate MPO size
        
    elif method == "nevpt2":
        # Needs various integral contractions
        temp_matrices += ncas**4 * SIZE * 3  # Multiple integral sets
        temp_matrices += ncas**6 * SIZE  # 3-body terms
        
    elif method in ["fci", "ci"]:
        if n_act_spin_electrons is None:
            raise ValueError(f"The number of active spin electrons must be provided.")
        # Full CI in active space
        temp_matrices += ncas**4 * SIZE
        # Hamiltonian matrix elements
        if n_act_spin_electrons is not None:
            nalpha_act = n_act_spin_electrons // 2 + n_act_spin_electrons % 2
            nbeta_act = n_act_spin_electrons // 2
        else:
            raise ValueError(f"The number of active spin electrons must be provided for method, {method}.")
        ndet = _calculate_ci_determinants(ncas, nalpha_act, nbeta_act)
        temp_matrices += min(ndet * ncas**2 * SIZE, ncas**6 * SIZE)  # Cap memory growth

    return {
        "integral_memory": integral_memory,
        "temporary_memory": temp_matrices,
    }


def get_memory(nbasis, nmo, ncore, reference, method, 
               int_type="df", naux=None, buffer_size=1000, n_act_spin_electrons=None, n_orbitals=None):
    """Estimate memory requirements for an SCF calculation in Psi4.
    
    Note that the number of active orbitals must be set for this prediction to be relevant.
    ```
    psi4.energy('casscf(n_electrons, n_orbitals)/sto-3g')
    ```
    
    Parameters
    ----------
        nbasis (int): Number of basis functions.
        nmo (int): Number of molecular orbitals.
        ncore (int): Number of frozen core orbitals
        reference (str): Reference wavefunction (i.e., 'UHF', 'RHF', 'ROHF')
        method (str): Multi-reference method, excluding ``(n_electrons, n_orbitals)`` notation. 
        Instead, define these with ``n_act_spin_electrons`` and ``n_orbitals``.
        int_type (str, optional): Type of integral evaluation method in Psi4.
            Options include:

            - 'pk': Full Atomic Orbital integrals stored in memory with nbasis⁴ scaling.
            - 'df': Density fitting with nbasis² × naux scaling.
            - 'cd': Cholesky decomposition with nbasis² × ncholesky scaling.
            - 'out_of_core': Minimal memory usage, integrals stored on disk.
            - 'direct': On-the-fly computation, no integrals stored.
            - 'conv': Same as 'pk'.

            Defaults to "df".
        Defaults to None. If None, all valence orbitals are used.
        naux (int, optional): Number of auxiliary functions (required for 'df' type).
            Defaults to None.
        buffer_size (int, optional): Size of I/O buffer (used for out-of-core/direct).
        n_act_spin_electrons (int, optional): Active electrons, usually valence electrons that participate in bonding, excitation, etc.
        n_orbitals (int, optional): Active orbitals that can be occupied or unoccupied in various configurations

    Returns
    -------
        dict: Memory breakdown in GB containing:

        - 'core_memory': Memory for core multi-reference matrices
        - 'orbital': Memory for MO coefficient matrix.
        - 'density_matrices': Memory for one or two density matrices.
        - 'fock_matrices': Memory for one or two Fock matrices.
        - 'method_memory': Method-specific memory (CI vectors, RDMs, etc.)
        - 'integral_memory': Two-electron integral storage/handling
        - 'temporary_memory': Memory for temporary matrices
        - 'diis_memory': DIIS extrapolation memory
        - 'peak_memory': Maximum simultaneous memory usage
        - Additional method-specific info (ci_determinants, bond_dimension, etc.)

    Notes
    -----
        - If X2C is enabled, multiply core_memory, integral_memory, and temporary_memory by 3, 2, and 2.5 respectively.
        - For large active spaces, determinant counts may be approximated to avoid overflow.
    """

    reference = reference.upper()
    ncas = n_orbitals if n_orbitals is not None else nmo - ncore
    method = method.lower().replace("-", "").replace("_", "").split("(")[0]
    
    if ncas <= 0 or n_act_spin_electrons < 0:
        raise ValueError("Active space parameters must be non-negative (ncas > 0 and n_act_spin_electrons > 0)")
    if n_act_spin_electrons > 2 * ncas:
        raise ValueError("Too many electrons for the given active space: n_act_spin_electrons > 2 * ncas")

    # Get contributions
    memory_breakdown = get_core_memory(nbasis, nmo, ncas, reference, method, n_act_spin_electrons=n_act_spin_electrons)
    memory_breakdown.update(get_diis_memory(nbasis, reference))
    memory_breakdown.update(get_integral_memory(
        nbasis, nmo, ncas, ncore, reference, method, 
        int_type=int_type, n_act_spin_electrons=n_act_spin_electrons, naux=naux, buffer_size=buffer_size
    ))

    # Take X2C relativistic into account
    if psi4.core.get_option("SCF", "RELATIVISTIC") == "X2C":
        memory_breakdown["use_x2c"] = True
        memory_breakdown["diis_memory"] *= 1.5
        memory_breakdown["integral_memory"] *= 1.5
        memory_breakdown["temporary_memory"] *= 1.5
    else:
        memory_breakdown["use_x2c"] = False

    # Calculate Peak Memory Needs
    memory_breakdown["peak_memory"] = memory_breakdown["core_memory"] + max(
        memory_breakdown["diis_memory"],
        memory_breakdown["integral_memory"],
        memory_breakdown["temporary_memory"],
    )
    
    if ncas is not None:
        memory_breakdown["active_space_info"] = {
            "ncas": ncas,
            "ncore": ncore
        }
        if n_act_spin_electrons is not None:
            memory_breakdown["active_space_info"].update({
                "nalpha_act": n_act_spin_electrons // 2 + n_act_spin_electrons % 2,
                "nbeta_act": n_act_spin_electrons // 2,
            })

    return memory_breakdown
