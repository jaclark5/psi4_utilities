"""Get the memory contributions and peak expected memory for an CC calculation

Those contributions include:
 - Amplitude storage (T1, T2, and higher-order amplitudes)
 - Integral storage and transformation memory
 - DIIS extrapolation memory
 - Method-specific intermediate tensors
 - Temporary working space for contractions

The peak memory usage considers the maximum memory required at any single phase of
the calculation, taking the maximum.

 Note
 ----
 - Does not account for:
    - CUHF (Constrained UHF)
    - PCM/SMD
    - Memory needed for an external field is not considered
    - Stability analysis is not considered
    - SOSCF (Second-Order SCF) memory
    - Fractional occupation memory

"""

import psi4

psi4.core.be_quiet()

BYTES = 8
CONVERSION = 1 / 1000**3  # From bytes to GB
SIZE = BYTES * CONVERSION
METHODS = [
    "bccd",
    "ccsd",
    "ccsdt",
    "cc2",
    "cc3",
    "qcisd",
    "qcisdt",
    "fnodfccsdt",
]


def get_amplitude_memory(nocc, nvirt, reference, method, nalpha=None, nbeta=None):
    """Get memory used for CC amplitude storage.

    Contributions include T1, T2, and lambda amplitudes needed throughout
    the CC procedure. These are persistent allocations.

    Parameters
    ----------
        nocc (int): Number of correlated occupied orbitals (correlated)
        nvirt (int): Total number of alpha virtual orbitals
        reference (str): Reference wavefunction (i.e., 'UHF', 'RHF', 'ROHF')
        method (str): CC energy method used
        nalpha (int, optional): Number of active alpha electrons (for open-shell)
        nbeta (int, optional): Number of active beta electrons (for open-shell)

    Returns
    -------
        dict: Memory breakdown in GB containing 'amplitude_memory'

    """

    reference = reference.upper()
    method = method.lower()
    if reference in ["ROHF", "UHF"] and (nalpha is None or nbeta is None):
        raise ValueError("Both nalpha and nbeta are needed for open-shell references")

    if reference == "RHF":
        t1_size = nocc * nvirt * SIZE
        if method in ["cc2"]:
            t2_size = nocc * nvirt * SIZE
        else:
            t2_size = (nocc * (nocc + 1) // 2) * (nvirt * (nvirt + 1) // 2) * SIZE

    elif reference in ["ROHF", "UHF"]:
        t1_size = (nalpha + nbeta) * nvirt * SIZE
        if method == "cc2":
            t2_size = (nalpha + nbeta) * nvirt * SIZE  # Simplified
        else:
            t2_aa = (nalpha * (nalpha - 1) // 2) * (nvirt * (nvirt - 1) // 2) * SIZE
            t2_bb = (nbeta * (nbeta - 1) // 2) * (nvirt * (nvirt - 1) // 2) * SIZE
            t2_ab = nalpha * nbeta * nvirt**2 * SIZE
            t2_size = t2_aa + t2_bb + t2_ab

    amp_memory = t2_size if method == "bccd" else (t1_size + t2_size)
    if method in ["ccsd", "ccsd(t)", "qcisd", "qcisd(t)", "cc3"]:
        amp_memory += t1_size + t2_size  # L1 + L2 amplitudes

    return {"amplitude_memory": amp_memory}


def get_method_specific_memory(nocc, nvirt, reference, method, nalpha=None, nbeta=None):
    """Calculate method-specific intermediate tensor memory requirements.

    Parameters
    ----------
        nocc (int): Number of occupied orbitals
        nvirt (int): Number of virtual orbitals
        reference (str): Reference wavefunction
        method (str): CC method
        nalpha (int, optional): Alpha electrons for open-shell
        nbeta (int, optional): Beta electrons for open-shell

    Returns
    -------
        dict: Method-specific memory breakdown
    """

    reference = reference.upper()
    method = method.lower()
    if reference in ["ROHF", "UHF"]:
        raise ValueError("Both nalpha and nbeta are needed for open-shell references")

    method_memory = {"intermediate_memory": 0, "working_memory": 0, "triples_memory": 0}

    if method == "ccsd":
        # CCSD intermediate tensors (W, F matrices)
        if reference == "RHF":
            w_oooo = (nocc**4 * SIZE) * 0.125  # 8-fold symmetry
            w_vvvv = (nvirt**4 * SIZE) * 0.125
            w_ovvo = nocc * nvirt**2 * nocc * SIZE
            w_ovov = nocc * nvirt * nocc * nvirt * SIZE
            method_memory["intermediate_memory"] = w_oooo + w_vvvv + w_ovvo + w_ovov
            method_memory["working_memory"] = nocc**2 * nvirt**2 * SIZE * 2
        elif reference == "ROHF":
            w_oooo_aa = (nalpha**4 * SIZE) * 0.125
            w_oooo_bb = (nbeta**4 * SIZE) * 0.125
            w_vvvv = (nvirt**4 * SIZE) * 0.125  # Same virtual space
            w_ovvo_aa = nalpha**2 * nvirt**2 * SIZE
            w_ovvo_bb = nbeta**2 * nvirt**2 * SIZE
            w_ovvo_ab = nalpha * nbeta * nvirt**2 * SIZE * 2  # Mixed spin

            method_memory["intermediate_memory"] = (
                w_oooo_aa + w_oooo_bb + w_vvvv + w_ovvo_aa + w_ovvo_bb + w_ovvo_ab
            )
            method_memory["working_memory"] = (
                (nalpha**2 + nbeta**2 + nalpha * nbeta) * nvirt**2 * SIZE
            )

        elif reference == "UHF":
            # Separate intermediates for each spin
            w_oooo_aa = (nalpha**4 * SIZE) * 0.125
            w_oooo_bb = (nbeta**4 * SIZE) * 0.125
            w_vvvv = 2 * (nvirt**4 * SIZE) * 0.125
            w_ovvo_aa = nalpha**2 * nvirt**2 * SIZE
            w_ovvo_bb = nbeta**2 * nvirt**2 * SIZE
            w_ovvo_ab = nalpha * nbeta * nvirt**2 * SIZE * 2

            method_memory["intermediate_memory"] = (
                w_oooo_aa + w_oooo_bb + w_vvvv + w_ovvo_aa + w_ovvo_bb + w_ovvo_ab
            )
            method_memory["working_memory"] = (
                nalpha**2 * nvirt**2 + nbeta**2 * nvirt**2 + nalpha * nbeta * nvirt**2
            ) * SIZE

    elif method == "ccsd(t)":
        # CCSD intermediates plus triples working space
        ccsd_memory = get_method_specific_memory(
            nocc, nvirt, reference, "ccsd", nalpha, nbeta
        )
        method_memory["intermediate_memory"] = ccsd_memory["intermediate_memory"]
        method_memory["working_memory"] = ccsd_memory["working_memory"]

        # Triples correction working memory (not stored permanently)
        if reference == "RHF":
            method_memory["triples_memory"] = nocc**3 * nvirt**3 * SIZE * 0.1
        else:
            # Open-shell triples are more complex
            t3_aaa = nalpha**3 * nvirt**3 * SIZE * 0.05
            t3_bbb = nbeta**3 * nvirt**3 * SIZE * 0.05
            t3_aab = nalpha**2 * nbeta * nvirt**3 * SIZE * 0.1
            t3_abb = nalpha * nbeta**2 * nvirt**3 * SIZE * 0.1
            method_memory["triples_memory"] = t3_aaa + t3_bbb + t3_aab + t3_abb

    elif method == "cc3":
        # CC3 stores some T3 amplitudes
        if reference == "RHF":
            method_memory["intermediate_memory"] = nocc**2 * nvirt**2 * SIZE * 1.5
            method_memory["triples_memory"] = nocc**3 * nvirt**3 * SIZE * 0.3
            method_memory["working_memory"] = nocc**3 * nvirt**3 * SIZE * 0.5
        else:
            method_memory["intermediate_memory"] = nocc**2 * nvirt**2 * SIZE * 2.5
            method_memory["triples_memory"] = nocc**3 * nvirt**3 * SIZE * 0.5
            method_memory["working_memory"] = nocc**3 * nvirt**3 * SIZE * 0.7

    elif method in ["qcisd", "qcisd(t)"]:
        # QCISD is similar to CCSD but with different intermediates
        if reference == "RHF":
            method_memory["intermediate_memory"] = nocc**2 * nvirt**2 * SIZE * 1.2
            method_memory["working_memory"] = nocc**2 * nvirt**2 * SIZE * 0.8
        else:
            method_memory["intermediate_memory"] = nocc**2 * nvirt**2 * SIZE * 2.0
            method_memory["working_memory"] = nocc**2 * nvirt**2 * SIZE * 1.2

        if method == "qcisd(t)":
            if reference == "RHF":
                method_memory["triples_memory"] = nocc**3 * nvirt**3 * SIZE * 0.1
            else:
                method_memory["triples_memory"] = nocc**3 * nvirt**3 * SIZE * 0.2

    elif method == "cc2":
        # CC2 has reduced memory requirements
        if reference == "RHF":
            method_memory["intermediate_memory"] = nocc * nvirt * SIZE * 5
            method_memory["working_memory"] = nocc**2 * nvirt * SIZE * 2
        else:
            method_memory["intermediate_memory"] = nocc * nvirt * SIZE * 8
            method_memory["working_memory"] = nocc**2 * nvirt * SIZE * 3

    elif method == "bccd":
        if reference == "RHF":
            method_memory["intermediate_memory"] = nocc**2 * nvirt**2 * SIZE * 1.5
            method_memory["working_memory"] = nocc**2 * nvirt**2 * SIZE * 1.0
        else:
            method_memory["intermediate_memory"] = nocc**2 * nvirt**2 * SIZE * 2.5
            method_memory["working_memory"] = nocc**2 * nvirt**2 * SIZE * 1.5

    elif method == "fno-df-ccsd(t)":
        # Frozen natural orbital CCSD(T) with reduced virtual space
        fno_factor = 0.6  # Typical FNO reduction
        nvirt_fno = int(nvirt * fno_factor)
        if reference == "RHF":
            method_memory["intermediate_memory"] = nocc**2 * nvirt_fno**2 * SIZE * 1.2
            method_memory["triples_memory"] = nocc**3 * nvirt_fno**3 * SIZE * 0.05
            method_memory["working_memory"] = nocc**2 * nvirt_fno**2 * SIZE * 0.8
        else:
            method_memory["intermediate_memory"] = nocc**2 * nvirt_fno**2 * SIZE * 2.0
            method_memory["triples_memory"] = nocc**3 * nvirt_fno**3 * SIZE * 0.1
            method_memory["working_memory"] = nocc**2 * nvirt_fno**2 * SIZE * 1.2

    return method_memory


def get_diis_memory(amplitude_memory, method):
    """Estimate memory used by DIIS extrapolation.

    DIIS stores a history of amplitude and residual vectors over several iterations.
    This memory can be significant for large amplitude spaces.

    Parameters
    ----------
        amplitude_memory (float): Amplitude memory
        method (str): CC energy method used

    Returns
    -------
        dict: DIIS memory breakdown in GB

    Notes
    -----
        - Returns 0.0 if DIIS is disabled via `SCF::DIIS = 0`.
        - RHF and RKS use 2 matrices per iteration (Fock and error).
        - UHF, UKS, ROHF use 4 matrices per iteration (α/β Fock and α/β error).

    """

    method = method.lower()

    if psi4.core.get_option("SCF", "DIIS") == 0:
        diis_memory = 0
    else:
        diis_max_vecs = psi4.core.get_option("SCF", "DIIS_MAX_VECS")
        diis_memory = 2 * diis_max_vecs * amplitude_memory
        # Some methods need additional DIIS storage
        if method in ["ccsd(t)", "cc3", "qcisd(t)"]:
            diis_memory *= 1.2  # Additional storage for lambda equations

    return {"diis_memory": diis_memory}


def get_integral_memory(
    nbasis,
    nmo,
    nocc,
    nvirt,
    reference,
    method,
    int_type="df",
    naux=None,
    nalpha=None,
    nbeta=None,
    buffer_size=1000000,
):
    """Calculate the memory required for storing integrals based on the specified method.

    Parameters
    ----------
        nbasis (int): Number of basis functions.
        nmo (int): Number of molecular orbitals.
        nvirt (int): Number of virtual orbitals
        nocc (int): Number of occupied orbitals (correlated)
        reference (str): Reference wavefunction (i.e., 'UHF', 'RHF', 'ROHF')
        method (str): CC method used. Supported methods include:
        "ccsd", "ccsd(t)", "bccd", "qcisd", "qcisd(t)", "cc2", "cc3", "fno-df-ccsd(t)".
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
        nalpha (int, optional): Number of alpha electrons (for open-shell). Default=None
        nbeta (int, optional): Number of beta electrons (for open-shell), Default=None
        buffer_size (int, optional): Size of I/O buffer (used for out-of-core/direct).

    Raises
    ------
        ValueError: If required inputs (e.g., nmo, naux) are missing for the chosen int_type.

    Returns
    -------
        dict: Dictionary with:
            - 'integral_memory': Memory in GB for integrals or intermediates.
            - 'temporary_memory': Memory in GB for Fock build, diagonalization, transformations.

    Notes
    -----
        - 'pk' and 'conv' store full AO integrals (nbasis⁴).
        - 'df' uses nbasis² * naux (B) and nocc * nvirt * naux (Q).
        - 'cd' approximates with nbasis² * nchol (nchol ≈ 2 * nbasis).
        - 'direct' and 'out_of_core' avoid storing integrals, rely on disk/batching.
        - Memory estimates are rough and do not include all intermediate tensors for coupled-cluster triples (e.g., T3 in CCSD(T), CC3).
        - For fno-df-ccsd(t), estimates assume DF integrals and frozen virtual spaces.
        - UHF reference may double some intermediate storage due to spin-unrestriction.

    """

    reference = reference.upper()
    method = method.lower()
    int_type = int_type.lower()
    if reference in ["ROHF", "UHF"]:
        raise ValueError("Both nalpha and nbeta are needed for open-shell references")

    # Base integral memory (AO integrals)
    integral_memory = 0
    temporary_memory = 0

    # Additional integral storage for CC methods
    if int_type in ["pk", "conv"]:
        # AO integral storage: (μν|ρσ) with 8-fold symmetry
        # Packed storage: only unique quartets are stored
        n_unique_pairs = nbasis * (nbasis + 1) // 2
        n_unique_quartets = n_unique_pairs * (n_unique_pairs + 1) // 2
        integral_memory = n_unique_quartets * SIZE

        # MO integral storage depends on reference type
        if method in ["ccsd", "ccsd(t)", "qcisd", "qcisd(t)", "cc3"]:

            if reference == "RHF":
                # RHF: (ia|jb), (ij|ab), (ij|kl), (ab|cd) integrals
                mo_ints_ovov = nocc * nvirt * nocc * nvirt * SIZE  # (ia|jb)
                mo_ints_oovv = nocc * nocc * nvirt * nvirt * SIZE  # (ij|ab)
                mo_ints_oooo = (
                    (nocc * (nocc + 1) // 2) ** 2 * SIZE * 0.5
                )  # (ij|kl) with symmetry
                mo_ints_vvvv = (
                    (nvirt * (nvirt + 1) // 2) ** 2 * SIZE * 0.5
                )  # (ab|cd) with symmetry
                integral_memory += (
                    mo_ints_ovov + mo_ints_oovv + mo_ints_oooo + mo_ints_vvvv
                )

            elif reference == "ROHF":
                # Same-spin integrals: (ia|jb)_αα, (ia|jb)_ββ
                mo_ints_ovov_aa = nalpha**2 * nvirt**2 * SIZE
                mo_ints_ovov_bb = nbeta**2 * nvirt**2 * SIZE
                # Mixed-spin integrals: (ia|jb)_αβ
                mo_ints_ovov_ab = nalpha * nbeta * nvirt**2 * SIZE

                # (ij|ab) type integrals for all spin combinations
                mo_ints_oovv_aa = nalpha**2 * nvirt**2 * SIZE
                mo_ints_oovv_bb = nbeta**2 * nvirt**2 * SIZE
                mo_ints_oovv_ab = nalpha * nbeta * nvirt**2 * SIZE

                # (ij|kl) and (ab|cd) with reduced symmetry
                mo_ints_oooo = (nalpha**2 + nbeta**2 + nalpha * nbeta) * SIZE
                mo_ints_vvvv = nvirt**4 * SIZE * 0.5  # Same virtual space

                integral_memory += (
                    mo_ints_ovov_aa
                    + mo_ints_ovov_bb
                    + mo_ints_ovov_ab
                    + mo_ints_oovv_aa
                    + mo_ints_oovv_bb
                    + mo_ints_oovv_ab
                    + mo_ints_oooo
                    + mo_ints_vvvv
                )

            elif reference == "UHF":
                # Alpha-alpha spin integrals
                mo_ints_ovov_aa = 2 * nalpha**2 * nvirt**2 * nalpha * nvirt * SIZE
                mo_ints_oooo_aa = (nalpha * (nalpha + 1) // 2) ** 2 * SIZE * 0.5
                mo_ints_vvvv_aa = (nvirt * (nvirt + 1) // 2) ** 2 * SIZE * 0.5

                # Beta-beta spin integrals
                mo_ints_ovov_bb = 2 * nbeta**2 * nvirt**2 * SIZE
                mo_ints_oooo_bb = (nbeta * (nbeta + 1) // 2) ** 2 * SIZE * 0.5
                mo_ints_vvvv_bb = (nvirt * (nvirt + 1) // 2) ** 2 * SIZE * 0.5

                # Alpha-beta mixed integrals (no antisymmetry)
                mo_ints_ovov_ab = nalpha * nbeta * nvirt**2 * SIZE
                mo_ints_oovv_ab = nalpha * nbeta * nvirt**2 * SIZE

                integral_memory += (
                    mo_ints_ovov_aa
                    + mo_ints_oooo_aa
                    + mo_ints_vvvv_aa
                    + mo_ints_ovov_bb
                    + mo_ints_oooo_bb
                    + mo_ints_vvvv_bb
                    + mo_ints_ovov_ab
                    + mo_ints_oovv_ab
                )

        # Transformation working space scales with reference
        if reference == "RHF":
            temporary_memory += nbasis**2 * nmo**2 * SIZE * 2
        elif reference == "ROHF":
            # ROHF transformation involves more intermediate steps
            temporary_memory += nbasis**2 * nmo**2 * SIZE * 3
        elif reference == "UHF":
            # UHF needs separate transformations for alpha and beta
            nmo_a = nalpha + nvirt if nalpha and nvirt else nmo // 2
            nmo_b = nbeta + nvirt if nbeta and nvirt else nmo // 2
            temporary_memory += nbasis**2 * (nmo_a**2 + nmo_b**2) * SIZE * 2

    elif int_type == "df":
        if naux is None:
            raise ValueError("naux is required for density-fitted integrals")

        # AO-basis 3-index integrals: (P|μν)
        df_3index_ao = naux * nbasis * nbasis * SIZE
        integral_memory = df_3index_ao

        # MO-basis 3-index integrals for CC methods
        if method in ["ccsd", "ccsd(t)", "qcisd", "qcisd(t)", "cc3"]:

            if reference == "RHF":
                # (Q|ia), (Q|ab), (Q|ij) integrals
                cc_df_ovia = naux * nocc * nvirt * SIZE
                cc_df_vvir = naux * nvirt * nvirt * SIZE
                cc_df_oocc = naux * nocc * nocc * SIZE
                integral_memory += cc_df_ovia + cc_df_vvir + cc_df_oocc

            elif reference == "ROHF":
                cc_df_ovia_a = naux * nalpha * nvirt * SIZE
                cc_df_ovia_b = naux * nbeta * nvirt * SIZE
                cc_df_vvir = naux * nvirt * nvirt * SIZE  # Shared virtual space
                cc_df_oocc_aa = naux * nalpha * nalpha * SIZE
                cc_df_oocc_bb = naux * nbeta * nbeta * SIZE
                cc_df_oocc_ab = naux * nalpha * nbeta * SIZE

                integral_memory += (
                    cc_df_ovia_a
                    + cc_df_ovia_b
                    + cc_df_vvir
                    + cc_df_oocc_aa
                    + cc_df_oocc_bb
                    + cc_df_oocc_ab
                )

            elif reference == "UHF":

                # Alpha integrals
                cc_df_ovia_a = naux * nalpha * nvirt * SIZE
                cc_df_vvir_aa = naux * nvirt * nvirt * SIZE
                cc_df_oocc_aa = naux * nalpha * nalpha * SIZE

                # Beta integrals
                cc_df_ovia_b = naux * nbeta * nvirt * SIZE
                cc_df_vvir_bb = naux * nvirt * nvirt * SIZE
                cc_df_oocc_bb = naux * nbeta * nbeta * SIZE

                # Mixed alpha-beta integrals
                cc_df_ovov_ab = naux * nalpha * nvirt * SIZE  # (Q|i_α a_β)
                cc_df_ovov_ba = naux * nbeta * nvirt * SIZE  # (Q|i_β a_α)

                integral_memory += (
                    cc_df_ovia_a
                    + cc_df_vvir_aa
                    + cc_df_oocc_aa
                    + cc_df_ovia_b
                    + cc_df_vvir_bb
                    + cc_df_oocc_bb
                    + cc_df_ovov_ab
                    + cc_df_ovov_ba
                )

        if method == "fno-df-ccsd(t)":
            # FNO reduces virtual space
            fno_factor = 0.6
            if reference == "RHF":
                nvirt_fno = int(nvirt * fno_factor)
                cc_df_memory = naux * (nocc * nvirt_fno + nvirt_fno**2) * SIZE
            elif reference == "ROHF":
                nvirt_fno = int(nvirt * fno_factor)
                nocc_total = (nalpha if nalpha else nocc // 2) + (
                    nbeta if nbeta else nocc // 2
                )
                cc_df_memory = naux * (nocc_total * nvirt_fno + nvirt_fno**2) * SIZE
            elif reference == "UHF":
                nvirt_fno = int(nvirt * fno_factor)
                nvirt_fno = int(nvirt * fno_factor)
                cc_df_memory = (
                    naux
                    * (
                        nalpha * nvirt_fno
                        + nbeta * nvirt_fno
                        + nvirt_fno**2
                        + nvirt_fno**2
                    )
                    * SIZE
                )
            integral_memory += cc_df_memory

        # DF transformation memory depends on reference
        if reference == "RHF":
            temporary_memory += naux * nbasis * nmo * SIZE
        elif reference == "ROHF":
            temporary_memory += naux * nbasis * nmo * SIZE * 1.5
        elif reference == "UHF":
            nmo_a = nalpha + nvirt if nalpha and nvirt else nmo // 2
            nmo_b = nbeta + nvirt if nbeta and nvirt else nmo // 2
            temporary_memory += naux * nbasis * (nmo_a + nmo_b) * SIZE

    elif int_type == "cd":
        # Cholesky decomposition: approximate nchol vectors
        nchol = min(2 * nbasis, nbasis * nbasis // 4)  # Typical Cholesky dimension
        cd_memory = nchol * nbasis * nbasis * SIZE
        integral_memory = cd_memory

        # CC-specific CD integrals (similar scaling to DF)
        if method in ["ccsd", "ccsd(t)", "cc3"]:
            if reference == "RHF":
                cc_cd_memory = nchol * (nocc * nvirt + nvirt**2 + nocc**2) * SIZE
            elif reference == "ROHF":
                nocc_total = (nalpha if nalpha else nocc // 2) + (
                    nbeta if nbeta else nocc // 2
                )
                cc_cd_memory = (
                    nchol * (nocc_total * nvirt + nvirt**2 + nocc_total**2) * SIZE * 1.3
                )
            elif reference == "UHF":
                nocc_total = nalpha + nbeta
                nvirt_total = nvirt + nvirt
                cc_cd_memory = (
                    nchol
                    * (nocc_total * nvirt_total + nvirt_total**2 + nocc_total**2)
                    * SIZE
                )
            integral_memory += cc_cd_memory

        # CD transformation memory
        if reference == "RHF":
            temporary_memory += nchol * nbasis * nmo * SIZE
        else:
            temporary_memory += nchol * nbasis * nmo * SIZE * 1.5

    elif int_type == "direct":
        # Direct methods: minimal integral storage, large temporary buffers
        integral_memory = 0  # No permanent storage

        # Working space depends on reference complexity
        if reference == "RHF":
            temporary_memory += nocc**2 * nvirt**2 * SIZE * 10
        elif reference == "ROHF":
            temporary_memory += nocc**2 * nvirt**2 * SIZE * 15
        elif reference == "UHF":
            temp_aa = nalpha**2 * nvirt**2 * SIZE * 10
            temp_bb = nbeta**2 * nvirt**2 * SIZE * 10
            temp_ab = nalpha * nbeta * nvirt * nvirt * SIZE * 5
            temporary_memory += temp_aa + temp_bb + temp_ab

        # Transformation working space
        temporary_memory += nbasis**2 * nmo**2 * SIZE * 2

    elif int_type == "out_of_core":
        # Out-of-core needs larger buffers for CC intermediate I/O
        temporary_memory += buffer_size * SIZE * 4

    return {"integral_memory": integral_memory, "temporary_memory": temporary_memory}


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
    buffer_size=10000,
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
        nalpha (int, optional): Number of alpha electrons (for open-shell). Default=None
        nbeta (int, optional): Number of beta electrons (for open-shell), Default=None
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
    method = method.lower()

    if method not in METHODS:
        raise ValueError(f"Unsupported method: {method}. Supported: {METHODS}")

    # Get contributions
    memory_breakdown = get_amplitude_memory(
        nocc, nvirt, reference, method, nalpha=nalpha, nbeta=nbeta
    )
    memory_breakdown.update(
        get_integral_memory(
            nbasis,
            nmo,
            nocc,
            nvirt,
            reference,
            method,
            int_type=int_type,
            naux=naux,
            buffer_size=buffer_size,
        )
    )
    memory_breakdown.update(
        get_diis_memory(memory_breakdown["amplitude_memory"], method)
    )
    memory_breakdown.update(
        get_method_specific_memory(nocc, nvirt, reference, method, nalpha, nbeta)
    )

    # Take X2C relativistic into account
    if psi4.core.get_option("SCF", "RELATIVISTIC") == "X2C":
        memory_breakdown["use_x2c"] = True
        memory_breakdown["diis_memory"] *= 3
        memory_breakdown["integral_memory"] *= 2
        memory_breakdown["temporary_memory"] *= 2.5
        memory_breakdown["intermediate_memory"] *= 1.5
    else:
        memory_breakdown["use_x2c"] = False

    # Calculate Peak Memory Needs
    # Phase 1: Integral transformation/storage
    if int_type in ["direct", "out_of_core"]:
        # Minimal integral storage, but need full temporary space
        phase1_memory = memory_breakdown.get(
            "integral_memory", 0
        ) + memory_breakdown.get("temporary_memory", 0)
    else:
        # Full integral storage plus partial temporary space
        phase1_memory = (
            memory_breakdown.get("integral_memory", 0)
            + memory_breakdown.get("temporary_memory", 0) * 0.4
        )

    # Phase 2: Main CC iterations (amplitudes + DIIS + intermediates + working space)
    phase2_memory = (
        memory_breakdown.get("amplitude_memory", 0)
        + memory_breakdown.get("diis_memory", 0)
        + memory_breakdown.get("intermediate_memory", 0)
        + memory_breakdown.get("working_memory", 0)
    )

    # Add partial integral memory if needed during iterations
    if int_type == "df":
        phase2_memory += memory_breakdown.get("integral_memory", 0) * 0.5
    elif int_type in ["pk", "conv"]:
        phase2_memory += memory_breakdown.get("integral_memory", 0) * 0.3

    # Phase 3: Triples correction (if applicable)
    if method in ["ccsd(t)", "qcisd(t)", "cc3", "fno-df-ccsd(t)"]:
        phase3_memory = (
            memory_breakdown.get("amplitude_memory", 0)  # Keep amplitudes
            + memory_breakdown.get("triples_memory", 0)  # Triples working space
            + memory_breakdown.get("working_memory", 0) * 0.3  # Reduced working space
        )

        # Triples may need some integral access
        if int_type == "df":
            phase3_memory += memory_breakdown.get("integral_memory", 0) * 0.3
    else:
        phase3_memory = 0

    return max(phase1_memory, phase2_memory, phase3_memory)
