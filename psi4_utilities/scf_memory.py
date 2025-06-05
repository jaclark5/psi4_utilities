"""Get the memory contributions and peak expected memory for an SCF calculation

Those contributions include:

 - Core Memory: For fundamental matrices needed throughout the SCF procedure.
 - DIIS Memory: For a history of matrices from previous SCF iterations for direct
 inversion of the iterative subspace (DIIS) extrapolation. This accelerates SCF convergence.
 - Integral Memory: Stores two-electron integrals.
 - Temporary Memory: Short-lived working memory for a SCF Iteration

 The peak memory usage is the Core Memory plus the highest of the other contributions.

 Notes
 -----
 - Does not account for:
    - CUHF (Constrained UHF)
    - PCM/SMD
    - Memory needed for an external field is not considered
    - Stability analysis is not considered
    - SOSCF (Second-Order SCF) memory
    - Fractional occupation memory

"""

import warnings

import psi4

psi4.core.be_quiet()

BYTES = 8
CONVERSION = 1 / 1000**3  # From bytes to GB
SIZE = BYTES * CONVERSION

DFT_DOUBLE_HYBRID_METHODS = [
    "b2plyp", "b2plypd", "b2plypd3", "b2plypd3bj",
    "dsdblyp", "dsdpbep86", "dsdpbepbe",
    "wb97x2lp", "wb97x2tqz"
]
DFT_GLOBAL_HYBRID_METHODS = [
    "b3lyp", "b3lypd", "b3lypd1", "b3lypd3", "b3lypd3bj", "b3lypchg",
    "b3x", "b970", "b971", "b972",
    "m05", "m052x", "m052xd3", "m05d3",
    "pbe0", "pbe02", "pbe0d", "pbe0d3", "pbe0d3bj",
    "pbeh3c", "pbeh3c", "wpbe0", "wpbesol0"
]
DFT_RANGE_SEPARATED_METHODS = [
    "wb97", "wb97x", "wb97xd", "wb97x3c", "wb97x2lp", "wb97x2tqz",
    "wblyp", "wpbe", "wpbesol", "wpbe0", "wb97mv",
]
DFT_METAGGA_METHODS = [
    "m05", "m052x", "m052xd3", "m05d3",
    "r2scan3c", "r2scan3c", "r2scan3c",
    "tpss", "scan", "revscan", "ft97", "ft97bx", "ft97c"
]
DFT_GGA_METHODS = [
    "blyp", "blypd", "blypd1", "blypd3", "blypd3bj",
    "bp86", "bp86d", "bp86d1", "bp86d3", "bp86d3bj",
    "b97d", "b97d3", "b97d3bj", "b973c",
    "pbe", "pbed", "pbed1", "pbed3", "pbed3bj",
    "pbec", "pbex", "pbesolx", "wpbesolx",
    "pw91", "pw91c", "pw91x", "pw92c", "p86c", "rpbex",
    "hcth", "hcth120", "hcth120d3", "hcth120d3bj", "hcth147", "hcth407",
    "svwn", "pz81c", "sogga", "soggax", "wsvwn", "wsx",
    "dldf", "dldfd", "dldfd09"
]
DFT_METHODS = (DFT_DOUBLE_HYBRID_METHODS + DFT_GLOBAL_HYBRID_METHODS +
              DFT_RANGE_SEPARATED_METHODS + DFT_METAGGA_METHODS + DFT_GGA_METHODS)

METHODS = ["scf", "dft", "hf", "hf3c", "hf3c"] + DFT_METHODS

def dft_memory_factor(functional):
    """
    Estimate the relative memory usage of a DFT method in Psi4,
    independent of basis set, normalized to B3LYP ≈ 1.0
    """

    factor = 1.0
    
    # Normalize functional name
    f = functional.lower().replace("-", "").replace("_", "").replace("(", "").replace(")", "")
    if f in DFT_DOUBLE_HYBRID_METHODS:
        factor += 1.2  # Largest overhead due to MP2-like terms
    elif f in DFT_RANGE_SEPARATED_METHODS:
        factor += 0.6  # Range-separated exchange is ERI heavy
    elif f in DFT_GLOBAL_HYBRID_METHODS:
        factor += 0.3  # Global hybrids are moderate


    if f in DFT_METAGGA_METHODS:
        factor += 0.2  # Meta-GGA grid demands

    # Dispersion variants (additive)
    dispersion_methods = {"d", "d1", "d3", "d3bj"}
    disp = any(tag in f for tag in dispersion_methods)
    if disp:
        factor += 0.1  # Empirical dispersion correction (D3, etc.)

    return round(factor, 2)

def get_core_memory(nbasis, nmo, reference):
    """Get memory used for fundamental matrices needed throughout the SCF procedure.

    Contributions include molecular orbitals, density matrices, and Fock matrices.
    These are persistent allocations needed throughout the SCF procedure.

    Parameters
    ----------
        nbasis (int): Number of basis functions
        nmo (int): Number of molecular orbitals
        reference (str): Reference wavefunction (i.e., 'UHF', 'RHF', 'ROHF')

    Returns
    -------
        dict: Memory breakdown in GB containing:

            - 'core_memory': Total memory for core SCF matrices.
            - 'orbitals': Memory for MO coefficient matrix.
            - 'density_matrices': Memory for one or two density matrices.
            - 'fock_matrices': Memory for one or two Fock matrices.

    """

    reference = reference.upper()
    memory_breakdown = {}

    orbital_factor = 2 if reference in ["UKS", "UHF", "CUHF", "ROHF"] else 1
    dens_fock_factor = 1 if reference in ["RKS", "RHF", "ROHF"] else 2

    memory_breakdown["orbital"] = nbasis * nmo * SIZE * orbital_factor
    memory_breakdown["density_matrices"] = nbasis**2 * SIZE * dens_fock_factor
    memory_breakdown["fock_matrices"] = nbasis**2 * SIZE * dens_fock_factor

    # SAD (Superposition of Atomic Densities) guess memory
    guess_type = psi4.core.get_option("SCF", "GUESS")
    if guess_type.upper() == "SAD":
        # Temporary atomic density matrices during initialization
        memory_breakdown["sad_memory"] = (
            nbasis**2 * SIZE * 2
        )  # Alpha and beta components
    else:
        memory_breakdown["sad_memory"] = 0

    memory_breakdown["core_memory"] = (
        memory_breakdown["orbital"]
        + memory_breakdown["density_matrices"]
        + memory_breakdown["fock_matrices"]
        + memory_breakdown["sad_memory"]
    )

    return memory_breakdown


def get_diis_memory(nbasis, reference):
    """Estimate memory used by DIIS extrapolation in SCF.

    DIIS stores a history of Fock and error matrices over several SCF iterations.
    This memory can be significant depending on the number of stored vectors and the
    reference type. This accelerates SCF convergence.

    Parameters
    ----------
        nbasis (int): Number of basis functions
        reference (str): Reference wavefunction (i.e., 'UHF', 'RHF', 'ROHF')

    Returns
    -------
        dict: Memory breakdown in GB containing:

            - 'diis_memory': Total memory for core SCF matrices.

    Notes
    -----
        - Returns 0.0 if DIIS is disabled via `SCF::DIIS = 0`.
        - RHF and RKS use 2 matrices per iteration (Fock and error).
        - UHF, UKS, ROHF use 4 matrices per iteration (α/β Fock and α/β error).

    """

    reference = reference.upper()

    if psi4.core.get_option("SCF", "DIIS") == 0:
        diis_memory = 0
    else:
        diis_max_vecs = psi4.core.get_option("SCF", "DIIS_MAX_VECS")
        matrices_per_iteration = 2 if reference in ["RHF", "RKS"] else 4
        diis_memory = diis_max_vecs * matrices_per_iteration * nbasis**2 * SIZE

    return {"diis_memory": diis_memory}


def get_integral_memory(
    nbasis, reference, nmo=None, int_type="df", naux=None, buffer_size=1000000
):
    """Calculate the memory required for storing integrals based on the specified method.

    Parameters
    ----------
        nbasis (int): Number of basis functions.
        reference (str): Reference wavefunction (i.e., 'UHF', 'RHF', 'ROHF')
        nmo (int, optional): Number of molecular orbitals. Not needed for
        ``int_type = out_of_core or direct``. Defaults to None.
        int_type (str, optional): Type of integral evaluation method in Psi4.
            Options include:

            - 'pk': Full Atomic Orbital integrals stored in memory with nbasis⁴ scaling.
            - 'df': Density fitting with nbasis² × naux scaling.
            - 'cd': Cholesky decomposition with nbasis² × ncholesky scaling.
            - 'out_of_core': Minimal memory usage, integrals stored on disk.
            - 'direct': On-the-fly computation, no integrals stored.
            - 'conv': Same as 'pk'.

            Defaults to "df".
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
    spin_factor = 1 if reference in ["RKS", "RHF"] else 2

    if nmo is None and int_type not in ["direct", "out_of_core"]:
        raise ValueError(
            "The number of molecular orbitals are needed for the integral memory estimation."
        )

    temp_matrices = nbasis**2 * SIZE * spin_factor  # G matrix construction workspace
    temp_matrices += nbasis * nmo * SIZE * spin_factor

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

    return {
        "integral_memory": integral_memory,
        "temporary_memory": temp_matrices,
    }


def get_memory(nbasis, nmo, reference, method, naux=None, int_type="df"):
    """Estimate memory requirements for an SCF calculation in Psi4.

    Parameters
    ----------
        nbasis (int): Number of basis functions
        nmo (int): Number of molecular orbitals
        reference (str): Reference wavefunction (e.g., 'UKS', 'ROHF', 'RHF')
        method (str): SCF / DFT method used.
        naux (int, optional): Number of auxiliary basis functions (for DF)
        int_type (str, optional): Integral computation method ('pk', 'df', 'cd',
        'out_of_core', 'direct'). See :func:`get_integral_memory`. Default='df'

    Returns
    -------
        dict: Memory breakdown in GB containing:

            - 'core_memory': Memory for core SCF matrices
            - 'orbitals': Memory for MO coefficient matrix.
            - 'density_matrices': Memory for one or two density matrices.
            - 'fock_matrices': Memory for one or two Fock matrices.
            - 'integral_memory': Two-electron integral storage/handling
            - 'temporary_memory': Memory for temporary matrices
            - 'diis_memory': DIIS extrapolation memory
            - 'peak_memory': Maximum simultaneous memory usage

    Notes
    -----
        - If X2C is enabled, multiply core_memory, integral_memory, and temporary_memory by 3, 2, and 2.5 respectively.
    """
    reference = reference.upper()

    # Get contributions
    memory_breakdown = get_core_memory(nbasis, nmo, reference)
    memory_breakdown.update(get_diis_memory(nbasis, reference))
    memory_breakdown.update(
        get_integral_memory(nbasis, reference, nmo=nmo, naux=naux, int_type=int_type)
    )
    
    method = method.lower().replace("-", "").replace("_", "").replace("(", "").replace(")", "")
    if method in DFT_METHODS:
        dft_factor = dft_memory_factor(method)
    else:
        warnings.warn("Not a detected DFT method.")
        dft_factor = 1

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
        memory_breakdown["temporary_memory"] * dft_factor,
    )

    return memory_breakdown
