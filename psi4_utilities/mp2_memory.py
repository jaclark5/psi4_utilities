"""Get the memory contributions and peak expected memory for a MP2 type calculation

Note that Psi4 classifies MP3 and other perturbation methods as an "MP2 type".

Those contributions include:

 - Amplitude Memory: MP2 amplitude storage
 - Integral Memory: Stores two-electron integrals.
 - Temporary Memory: Short-lived working memory for a SCF Iteration
 - DIIS Memory: (Uncommon) For a history of matrices from previous SCF iterations for direct
 inversion of the iterative subspace (DIIS) extrapolation. This accelerates SCF convergence.

 The peak memory usage the maximum of three phases:

 - Phase 1: Integral transformation: integral memory + temporary memory
 - Phase 2: Amplitude computation w/ partial integral storage: amplitude_memory + integral_memory * 0.7 + temporary_memory * 0.3
 - Phase 3: Orbital optimization iterations w/ reduced amplitude / integral storage (if applicable in omp*): amplitude_memory * 0.8 + integral_memory * 0.5 + diis_memory + temporary_memory + orbital gradient workspace

 Notes
 -----
 - Does not account for:
    - CUHF (Constrained UHF)
    - PCM/SMD
    - Memory needed for an external field is not considered
    - Stability analysis is not considered
    - SOSCF (Second-Order SCF) memory
    - Fractional occupation memory
    - Multithreading

"""

import psi4

psi4.core.be_quiet()

BYTES = 8
CONVERSION = 1 / 1000**3  # From bytes to GB
SIZE = BYTES * CONVERSION
METHODS = [
    "mp2",
    "dfmp2",
    "convmp2",
    "sosmp2",
    "scsmp2",
    "omp2",
    "mp3",
    "sosmp3",
    "scsmp3",
    "fnomp3",
    "sospimp2",
    "scsomp2",
    "sosomp2",
    "scsomp3",
    "sosomp3",
]


def get_amplitude_memory(
    nocc,
    nvirt,
    reference,
    base_method,
    int_type="df",
    naux=None,
    nalpha=None,
    nbeta=None,
    is_sos=False,
):
    """Get memory used for MP2-type amplitude storage.

    Contributions include molecular orbitals, density matrices, and Fock matrices.
    These are persistent allocations needed throughout the SCF procedure.

    Parameters
    ----------
        nocc (int): Number of occupied orbitals (correlated)
        nvirt (int): Number of virtual orbitals
        reference (str): Reference wavefunction (i.e., 'UHF', 'RHF', 'ROHF')
        base_method (str): Base method used, i.e., "mp2" or "mp3"
        int_type (str): Integral method ('df', 'cd', 'pk', 'conv', 'direct', 'out_of_core')
        naux (int, optional): Number of auxiliary functions (required for 'df'/'cd')
        nalpha (int, optional): Number of alpha electrons (for open-shell)
        nbeta (int, optional): Number of beta electrons (for open-shell)
        is_sos (bool, optional): Whether to use Spin-Opposite-Scaled (SOS) method. Default is False.

    Returns
    -------
        dict: Memory breakdown in GB containing 'amplitude_memory'

    Notes
    -----
    - SOS methods only store opposite-spin amplitudes, reducing memory by ~50%
        - DF/CD methods use different scaling due to batched amplitude computation
        - Conventional methods store full amplitude tensors
        - MP3 requires additional intermediate storage
    """

    reference = reference.upper()
    base_method = base_method.lower()
    int_type = int_type.lower()

    # Check for required parameters
    if int_type in ["df", "cd"] and naux is None:
        raise ValueError(f"naux is required for {int_type.upper()} integral method")

    if reference in ["ROHF", "UHF"] and (nalpha is None or nbeta is None):
        raise ValueError("Both nalpha and nbeta are needed for open-shell references")

    if reference == "RHF":
        if base_method == "mp2":
            if int_type in ["df", "cd"]:
                # DF-MP2: Batched amplitude computation, memory scales with batch size
                # Typical batch size is min(nocc*nvirt, naux) for (ia|P) intermediates
                batch_size = min(nocc * nvirt, naux)
                amp_memory = batch_size * naux * SIZE
                # Add working space for amplitude assembly
                if is_sos:
                    amp_memory = (nocc // 2) * nvirt ** 2 * SIZE
                else:
                    amp_memory = nocc ** 2 * nvirt ** 2 * SIZE
            elif int_type in ["pk", "conv"]:
                # Conventional MP2: Full (ia|jb) amplitude tensor
                if is_sos:
                    amp_memory = (nocc // 2) * nvirt ** 2 * SIZE
                else:
                    amp_memory = nocc**2 * nvirt**2 * SIZE
            elif int_type in ["direct", "out_of_core"]:
                # Minimal amplitude storage, computed on-the-fly or from disk
                amp_memory = nocc * nvirt * SIZE * 5  # Small working buffers
            else:
                raise ValueError(f"Unsupported integral type: {int_type}")
        elif base_method == "mp3":
            if int_type in ["df", "cd"]:
                # DF-MP3: Additional intermediates for triple substitutions
                batch_size = min(nocc * nvirt, naux)
                amp_memory = batch_size * naux * SIZE
                if is_sos:
                    amp_memory += (nocc * nvirt ** 2) * SIZE
                else:
                    amp_memory += (nocc**2 * nvirt + nocc * nvirt**2) * SIZE # T3_ααα + T3_βββ + T3_αβα + T3_αβγ
            else:
                if is_sos:
                    # SOS-MP3: Conventional storage for opposite-spin terms only
                    amp_memory = (nocc + nocc**2) * nvirt**2 * SIZE
                else:
                    # Standard MP3: Full tensor storage for all spin cases
                    amp_memory = (nocc**2 * nvirt**2 + nocc**2 * nvirt + nocc * nvirt**2) * SIZE

    elif reference in ["ROHF", "UHF"]:
        if base_method == "mp2":
            if int_type in ["df", "cd"]:
                if is_sos:
                    # SOS-UMP2: Only mixed-spin (αβ) amplitudes
                    batch_mixed = min(nalpha * nbeta * nvirt, naux)
                    amp_memory = batch_mixed * naux * SIZE
                    # Working space for mixed-spin amplitudes only
                    amp_memory += nalpha * nbeta * nvirt**2 * SIZE
                else:
                    # DF-UMP2: Separate alpha/beta contributions
                    batch_alpha = min(nalpha * nvirt, naux)
                    batch_beta = min(nbeta * nvirt, naux)
                    batch_mixed = min(nalpha * nbeta * nvirt, naux)
                    amp_memory = (batch_alpha + batch_beta + batch_mixed) * naux * SIZE
                    # Working space for all spin combinations
                    amp_memory += (nalpha**2 + nbeta**2 + nalpha * nbeta) * nvirt**2 * SIZE

            else:
                # Conventional UMP2: Full amplitude tensors for all spin cases
                if is_sos: # SOS-UMP2: Only mixed-spin amplitude tensor
                    amp_memory = nalpha * nbeta * nvirt**2 * SIZE
                else: # Standard UMP2: Full amplitude tensors for all spin cases
                    amp_memory = (nalpha**2 + nbeta**2 + nalpha * nbeta) * nvirt**2 * SIZE

        elif base_method == "mp3":
            if int_type in ["df", "cd"]:
                if is_sos:
                    # SOS-UMP3: Only mixed-spin contributions
                    batch_mixed = min(nalpha * nbeta * nvirt, naux)
                    amp_memory = batch_mixed * naux * SIZE
                    # MP3 intermediates for mixed-spin terms
                    amp_memory += (nalpha * nbeta * nvirt + nalpha * nbeta) * nvirt * SIZE
                else:
                    # Standard UMP3: All spin combinations
                    batch_alpha = min(nalpha * nvirt, naux)
                    batch_beta = min(nbeta * nvirt, naux)
                    amp_memory = (batch_alpha + batch_beta) * naux * SIZE
                    # Full MP3 intermediates
                    amp_memory += (
                        (nalpha**2 + nbeta**2 + nalpha * nbeta) * nvirt**2 +
                        (nalpha**2 * nvirt + nbeta**2 * nvirt + nalpha * nvirt**2 + nbeta * nvirt**2)
                    ) * SIZE

            else:
                if is_sos:
                    # SOS-UMP3: Conventional storage for mixed-spin terms
                    amp_memory = nalpha * nbeta * nvirt**2 * SIZE  # T2_αβ
                    amp_memory += (nalpha**2 * nbeta + nalpha * nbeta**2) * nvirt * SIZE  # T3 mixed
                else:
                    # Standard UMP3: Full tensor storage
                    amp_memory = (
                        (nalpha**2 + nbeta**2 + nalpha * nbeta) * nvirt ** 2 +  # T2
                        (nalpha**2 * nvirt + nbeta**2 * nvirt + nalpha * nvirt**2 + nbeta * nvirt**2)  # T3 intermediates
                    ) * SIZE
    else:
        raise ValueError(f"Invalid reference '{reference}' for MP2.")

    return {"amplitude_memory": amp_memory}


def get_diis_memory(amplitude_memory, reference):
    """Estimate memory used by DIIS extrapolation for orbital-optimized MP2.

    DIIS for orbital optimization stores orbital rotation parameters and gradients
    over several iterations. This is different from SCF DIIS.

    Parameters
    ----------
        amplitude_memory (float): Memory for storing one set of amplitudes/parameters
        reference (str): Reference wavefunction (i.e., 'UHF', 'RHF', 'ROHF')

    Returns
    -------
        dict: Memory breakdown in GB containing 'diis_memory'

    Notes
    -----
        - Only relevant for orbital-optimized methods (OMP2, OMP3, etc.)
        - Returns 0.0 if DIIS is disabled
        - Stores orbital rotation vectors and error vectors

    """

    reference = reference.upper()

    if psi4.core.get_option("SCF", "DIIS") == 0:
        diis_memory = 0
    else:
        diis_max_vecs = psi4.core.get_option("SCF", "DIIS_MAX_VECS")
        vector_memory = amplitude_memory * 2  # parameters + errors
        b_matrix = (diis_max_vecs + 1) ** 2 * 8  # 8 bytes per double
        diis_memory = vector_memory * diis_max_vecs + b_matrix

    return {"diis_memory": diis_memory}


def get_integral_memory(
    nbasis,
    nmo,
    nocc,
    nvirt,
    reference,
    base_method,
    int_type="df",
    naux=None,
    buffer_size=1000000,
    is_sos=False,
):
    """Calculate the memory required for storing integrals based on the specified method.

    Parameters
    ----------
        nbasis (int): Number of basis functions.
        nmo (int, optional): Number of molecular orbitals.
        nvirt (int): Number of virtual orbitals
        nocc (int): Number of occupied orbitals (correlated)
        reference (str): Reference wavefunction (i.e., 'UHF', 'RHF', 'ROHF')
        base_method (str): Base method used, i.e., "mp2" or "mp3"
        int_type (str, optional): Type of integral evaluation method in Psi4.
        naux (int, optional): Number of auxiliary functions (required for 'df' type).
        buffer_size (int, optional): Size of I/O buffer (used for out-of-core/direct).
        is_sos (bool, optional): Whether this is a Spin-Opposite Scaling calculation. Defaults to False.

    Returns
    -------
        dict: Dictionary with integral_memory and temporary_memory in bytes.
    """

    reference = reference.upper()
    base_method = base_method.lower()
    int_type = int_type.lower()

    # Base temporary memory varies by reference type
    if reference == "RHF":
        # Single set of orbitals
        temp_matrices = nbasis**2 * SIZE * 2  # Fock workspace
        temp_matrices += nbasis * nmo * SIZE * 2  # MO transformation buffers
    elif reference == "UHF":
        # Separate alpha and beta orbitals
        temp_matrices = nbasis**2 * SIZE * 4
        temp_matrices += nbasis * nmo * SIZE * 4
    elif reference == "ROHF":
        # High-spin open shell
        temp_matrices = nbasis**2 * SIZE * 3
        temp_matrices += nbasis * nmo * SIZE * 3
    else:
        raise ValueError(f"Unsupported reference: {reference}")

    nactive = nocc + nvirt

    if int_type in ["pk", "conv"]:
        # Conventional integrals - same AO integral storage for all references
        integral_memory = (nbasis**4) * SIZE * 0.125  # 8-fold symmetry

        # Method and reference-specific MO integral storage
        if base_method == "mp2":
            if reference == "RHF": # Single (ov|ov) integral block for SOS, same structure for all integrals
                factor = 1
            elif reference == "UHF": # Separate alpha-alpha, beta-beta, and alpha-beta blocks
                factor = 1 if is_sos else 3
            elif reference == "ROHF": # ROHF uses modified integral structure
                factor = 1 if is_sos else 2
            integral_memory += nocc**2 * nvirt**2 * SIZE * factor

        elif base_method == "mp3":
            if reference == "RHF":
                integral_memory += 2 * nocc**2 * nvirt**2 * SIZE # (ov|ov) and (oo|vv)
                if not is_sos: # Add same spin terms
                    integral_memory += nvirt**4 * SIZE * 0.5  # (vv|vv) with symmetry
                    integral_memory += nocc**4 * SIZE * 0.5  # (oo|oo) with symmetry
            elif reference == "UHF":
                if is_sos:
                    integral_memory += 2 * nocc**2 * nvirt**2 * SIZE # (ov_α|ov_β) and (oo_α|vv_β) + (oo_β|vv_α)
                else:
                    # All spin combinations for MP3
                    integral_memory += nocc**2 * nvirt**2 * SIZE * 3 # (ov|ov) all spin cases
                    integral_memory += nocc**2 * nvirt**2 * SIZE * 3 # (oo|vv) all spin cases
                    integral_memory += nvirt**2 * nvirt**2 * SIZE * 1.5 # (vv|vv) spin cases
                    integral_memory += nocc**4 * SIZE * 1.5  # (oo|oo) spin cases
            elif reference == "ROHF":
                if is_sos:
                    integral_memory += 2 * nocc**2 * nvirt**2 * SIZE # (ov|ov) opposite-spin
                else:
                    # ROHF MP3 integral requirements
                    integral_memory += nocc**2 * nvirt**2 * SIZE * 2.5 # Mixed (ov|ov) blocks
                    integral_memory += nocc**2 * nvirt**2 * SIZE * 2.5 # Mixed (oo|vv) blocks
                    integral_memory += nvirt**4 * SIZE * 1.25  # (vv|vv) blocks
                    integral_memory += nocc**4 * SIZE * 1.25  # (oo|oo) blocks

        # Transformation workspace
        if reference == "RHF":
            temp_matrices += nbasis**2 * nactive * SIZE
            temp_matrices += nocc * nvirt * nbasis * SIZE
        elif reference == "UHF":
            temp_matrices += nbasis**2 * nactive * SIZE * 2  # Separate transformations
            temp_matrices += nocc * nvirt * nbasis * SIZE * 2
        elif reference == "ROHF":
            temp_matrices += nbasis**2 * nactive * SIZE * 1.5  # Modified transformation
            temp_matrices += nocc * nvirt * nbasis * SIZE * 1.5

        # Method-specific temporary arrays
        if base_method == "mp3":
            if reference == "RHF":
                temp_matrices += nocc**2 * nvirt**2 * SIZE * 2
            elif reference == "UHF":
                temp_matrices += nocc**2 * nvirt**2 * SIZE * 4  # Separate spin blocks
            elif reference == "ROHF":
                temp_matrices += nocc**2 * nvirt**2 * SIZE * 3  # ROHF intermediates

    elif int_type == "df":
        if naux is None:
            raise ValueError("Number of auxiliary functions (naux) required for DF")

        # Three-index B integrals: (μν|Q) - same for all references
        b_integrals = nbasis**2 * naux * SIZE

        # Method and reference-specific Q integrals
        if base_method == "mp2":
            if reference == "RHF":
                q_integrals = nocc * nvirt * naux * SIZE  # (ov|Q)
            elif reference == "UHF":
                q_integrals = nocc * nvirt * naux * SIZE * 2  # Separate α and β
            elif reference == "ROHF":
                q_integrals = nocc * nvirt * naux * SIZE * 1.5  # ROHF mixed blocks

        elif base_method == "mp3":
            if reference == "RHF":
                q_integrals = nocc * nvirt * naux * SIZE  # (ov|Q)
                q_integrals += nocc**2 * naux * SIZE  # (oo|Q)
                q_integrals += nvirt**2 * naux * SIZE  # (vv|Q)
            elif reference == "UHF":
                q_integrals = nocc * nvirt * naux * SIZE * 2  # (ov|Q) α,β
                q_integrals += nocc**2 * naux * SIZE * 2  # (oo|Q) α,β
                q_integrals += nvirt**2 * naux * SIZE * 2  # (vv|Q) α,β
            elif reference == "ROHF":
                q_integrals = nocc * nvirt * naux * SIZE * 1.5  # (ov|Q) ROHF
                q_integrals += nocc**2 * naux * SIZE * 1.5  # (oo|Q) ROHF
                q_integrals += nvirt**2 * naux * SIZE * 1.5  # (vv|Q) ROHF

        integral_memory = b_integrals + q_integrals

        # DF-specific temporary memory
        temp_matrices += nbasis**2 * naux * SIZE * 0.5
        temp_matrices += max(b_integrals * 0.3, q_integrals)

        if base_method == "mp3":
            if reference == "RHF":
                temp_matrices += naux * nocc * nvirt * SIZE
            elif reference == "UHF":
                temp_matrices += naux * nocc * nvirt * SIZE * 2
            elif reference == "ROHF":
                temp_matrices += naux * nocc * nvirt * SIZE * 1.5

    elif int_type == "cd":
        ncholesky = min(nbasis * 2, naux) if naux else nbasis * 2

        # Cholesky vectors - same for all references
        integral_memory = nbasis**2 * ncholesky * SIZE

        # Method and reference-specific Cholesky intermediates
        if base_method == "mp2":
            if reference == "RHF":
                integral_memory += nocc * nvirt * ncholesky * SIZE
            elif reference == "UHF":
                integral_memory += nocc * nvirt * ncholesky * SIZE * 2
            elif reference == "ROHF":
                integral_memory += nocc * nvirt * ncholesky * SIZE * 1.5

        elif base_method == "mp3":
            if reference == "RHF":
                integral_memory += nocc * nvirt * ncholesky * SIZE
                integral_memory += (nocc**2 + nvirt**2) * ncholesky * SIZE
            elif reference == "UHF":
                integral_memory += nocc * nvirt * ncholesky * SIZE * 2
                integral_memory += (nocc**2 + nvirt**2) * ncholesky * SIZE * 2
            elif reference == "ROHF":
                integral_memory += nocc * nvirt * ncholesky * SIZE * 1.5
                integral_memory += (nocc**2 + nvirt**2) * ncholesky * SIZE * 1.5

        # CD workspace
        if reference == "RHF":
            temp_matrices += nbasis**2 * SIZE * 3
            temp_matrices += ncholesky * nactive * SIZE
        elif reference == "UHF":
            temp_matrices += nbasis**2 * SIZE * 6  # Separate decomposition
            temp_matrices += ncholesky * nactive * SIZE * 2
        elif reference == "ROHF":
            temp_matrices += nbasis**2 * SIZE * 4.5
            temp_matrices += ncholesky * nactive * SIZE * 1.5

        temp_matrices += ncholesky**2 * SIZE * 0.1

    elif int_type == "out_of_core":
        integral_memory = buffer_size * SIZE * 2

        if reference == "RHF":
            temp_matrices += buffer_size * SIZE
            temp_matrices += nbasis**2 * SIZE * 2
        elif reference == "UHF":
            temp_matrices += buffer_size * SIZE * 2  # Separate buffers
            temp_matrices += nbasis**2 * SIZE * 4
        elif reference == "ROHF":
            temp_matrices += buffer_size * SIZE * 1.5
            temp_matrices += nbasis**2 * SIZE * 3

        # Method-specific I/O buffers
        if base_method == "mp3":
            temp_matrices += buffer_size * SIZE  # Additional buffer for MP3

    elif int_type == "direct":
        integral_memory = 0

        if reference == "RHF":
            temp_matrices += nbasis**2 * SIZE * 4
            temp_matrices += nbasis * nmo * SIZE * 2
            temp_matrices += nocc * nvirt * SIZE * 10
        elif reference == "UHF":
            temp_matrices += nbasis**2 * SIZE * 8  # Separate spin matrices
            temp_matrices += nbasis * nmo * SIZE * 4
            temp_matrices += nocc * nvirt * SIZE * 20
        elif reference == "ROHF":
            temp_matrices += nbasis**2 * SIZE * 6
            temp_matrices += nbasis * nmo * SIZE * 3
            temp_matrices += nocc * nvirt * SIZE * 15

        # Method-specific working arrays
        if base_method == "mp3":
            if reference == "RHF":
                temp_matrices += nocc**2 * nvirt**2 * SIZE
            elif reference == "UHF":
                temp_matrices += nocc**2 * nvirt**2 * SIZE * 2
            elif reference == "ROHF":
                temp_matrices += nocc**2 * nvirt**2 * SIZE * 1.5

    else:
        raise ValueError(f"Unsupported int_type: {int_type}")

    return {
        "integral_memory": integral_memory,
        "temporary_memory": temp_matrices,
    }


def get_memory(
    nbasis,
    nmo,
    nvirt,
    nocc,
    reference,
    method,
    naux=None,
    nalpha=None,
    nbeta=None,
    int_type="df",
):
    """Estimate memory requirements for an MP calculation in Psi4.

    Parameters
    ----------
        nbasis (int): Number of basis functions
        nmo (int): Number of molecular orbitals
        nvirt (int): Number of virtual orbitals
        nocc (int): Number of occupied orbitals (correlated)
        reference (str): Reference wavefunction (e.g., 'UKS', 'ROHF', 'RHF')
        method (str): Energy method string
        naux (int, optional): Number of auxiliary basis functions (for DF)
        nalpha (int): Number of alpha electrons (for open-shell)
        nbeta (int): Number of beta electrons (for open-shell)
        int_type (str, optional): Integral computation method ('pk', 'df', 'cd',
        'out_of_core', 'direct'). See :func:`get_integral_memory`. Default='df'

    Returns
    -------
        dict: Memory breakdown in GB containing:

            - 'amplitude_memory': MP2 amplitude storage
            - 'integral_memory': Two-electron integral storage/handling
            - 'temporary_memory': Memory for temporary matrices and transformations
            - 'diis_memory': DIIS extrapolation memory (if applicable)
            - 'peak_memory': Maximum simultaneous memory usage
            - 'method_info': Information about the calculation method

    Notes
    -----
        - Memory estimates are for single-threaded execution
        - X2C relativistic corrections increase memory requirements
        - Frozen core approximations reduce effective nocc
    """
    reference = reference.upper()
    method = method.lower().replace("-", "").replace("_", "").replace("(", "").replace(")", "")
    is_orbital_opt = "omp" in method
    is_sos = "sos" in method
    base_method = (
        method.replace("df", "")
        .replace("conv", "")
        .replace("sos", "")
        .replace("scs", "")
        .replace("fno", "")
        .replace("o", "")
    )

    if method not in METHODS:
        raise ValueError(f"Unsupported method: {method}. Supported: {METHODS}")

    # Get contributions
    memory_breakdown = get_amplitude_memory(
        nocc,
        nvirt,
        reference,
        base_method,
        nalpha=nalpha,
        nbeta=nbeta,
        int_type=int_type,
        naux=naux,
        is_sos=is_sos,
    )
    if is_orbital_opt:
        memory_breakdown.update(
            get_diis_memory(memory_breakdown["amplitude_memory"], reference)
        )
    else:
        memory_breakdown["diis_memory"] = 0.0
    memory_breakdown.update(
        get_integral_memory(
            nbasis,
            nmo,
            nocc,
            nvirt,
            reference,
            base_method,
            naux=naux,
            int_type=int_type,
            is_sos=is_sos,
        )
    )

    # Take X2C relativistic into account
    if psi4.core.get_option("SCF", "RELATIVISTIC") == "X2C":
        memory_breakdown["use_x2c"] = True
        if memory_breakdown["diis_memory"] > 0:
            memory_breakdown["diis_memory"] *= 1.5
        memory_breakdown["integral_memory"] *= 2
        memory_breakdown["temporary_memory"] *= 2.5
    else:
        memory_breakdown["use_x2c"] = False

    # _____ Calculate Peak Memory Needs ______
    # Phase 1: Integral transformation
    phase1 = memory_breakdown["temporary_memory"] + memory_breakdown["integral_memory"]

    # Phase 2: Amplitude computation (main MP2 calculation)
    phase2 = (
        memory_breakdown["amplitude_memory"]
        + memory_breakdown["integral_memory"] * 0.7  # Partial integral storage
        + memory_breakdown["temporary_memory"] * 0.3
    )

    # Phase 3: Orbital optimization iterations (if applicable)
    if is_orbital_opt:
        phase3 = (
            memory_breakdown["amplitude_memory"] * 0.8  # Reduced amplitude storage
            + memory_breakdown["integral_memory"] * 0.5  # Reduced integral storage
            + memory_breakdown["diis_memory"]
            + memory_breakdown["temporary_memory"] * 0.5
            + nocc * nvirt * SIZE * 5
        )  # Orbital gradient workspace
    else:
        phase3 = 0

    # Peak memory is the maximum across all phases
    memory_breakdown["peak_memory"] = max(phase1, phase2, phase3)

    memory_breakdown["method_info"] = {
        "base_method": base_method,
        "integral_type": int_type,
        "is_orbital_optimized": is_orbital_opt,
        "reference": reference,
    }

    return memory_breakdown
