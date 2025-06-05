import warnings
import os
import glob

import numpy as np
from qcelemental.models import Molecule
import psi4

import psi4_utilities.scf_memory as mscf
import psi4_utilities.mp2_memory as mmp2
import psi4_utilities.cc_memory as mcc

BYTES = 8
CONVERSION = 1 / 1000**3  # From bytes to GB
SIZE = 8 * CONVERSION

psi4.core.be_quiet()


def get_calculation_type(method):
    """Return the general calculation type (i.e., CC, MP, or SCF) based on the energy method.

    Parameters
    ----------
        method (str): Psi4 energy method

    Returns
    -------
        str: Energy method type, either SCF, CC, or MP2
    """
    
    method = method.lower().replace("-", "").replace("_", "").replace("(", "").replace(")", "")
    
    if method.lower() in mscf.METHODS:
        return "SCF"
    elif method.lower() in mmp2.METHODS:
        return "MP2"
    elif method.lower() in mcc.METHODS:
        return "CC"
    else:
        raise ValueError(
            f"The method, {method}, is not recognized. Consider adding this option to `get_calculation_type`"
        )


def get_auxiliary_basis_size(
    molecule, primary_basis, int_type=None, method=None, aux_basis_name=None
):
    """Get the number of auxiliary basis functions for a given molecule, primary basis, and method.

    Parameters
    ----------
    molecule (psi4.core.Molecule): The molecule object
    primary_basis (str): Name of the primary basis set
    int_type (str, optional): Type of method (e.g., 'DF', 'CD', etc.). If None, will use the Psi4 default
    method (str, optional): Name of the method (e.g., 'SCF', 'B3LYP', 'MP2', 'MP3', 'OMP2', 'CCSD')
    aux_basis_name (str): Name of auxiliary basis set

    Returns
    -------
    dict
        Dictionary containing number of auxiliary basis functions and basis set details
    """
    result = {}

    calculation_type = get_calculation_type(method)
    result["calculation_type"] = calculation_type

    # If no int_type is specified, check Psi4's current options
    if int_type is None:
        option_name = calculation_type + "_TYPE"
        int_type = psi4.core.get_option(calculation_type, option_name)

    # Determine key and fitrole based on calculation type and method
    if int_type.upper() in ["DF", "DENSITY_FITTED"]:
        if calculation_type.upper() == "SCF":
            key = "DF_BASIS_SCF"
            fitrole = "JKFIT"
        elif calculation_type.upper() == "MP2":
            key = "DF_BASIS_MP2"
            fitrole = "RIFIT"
        elif calculation_type.upper() == "CC":
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
        aux_basis_obj = psi4.core.BasisSet.build(
            molecule, key, "", fitrole, primary_basis
        )
        aux_basis_name = aux_basis_obj.name()
        naux = aux_basis_obj.nbf()
    else:
        aux_basis_obj = psi4.core.BasisSet.build(molecule, key, aux_basis_name)
        naux = aux_basis_obj.nbf()

    # Get the number of auxiliary basis functions
    result["naux"] = naux
    result["aux_basis_used"] = aux_basis_name
    result["key_used"] = key
    result["fitrole_used"] = fitrole

    return result


def get_orbital_counts(
    mol,
    basis,
    reference,
    freeze_core=False,
    use_wfn=False,
    method=None,
    int_type=None
):
    """Computes and returns the number of molecular orbitals (MOs), alpha and beta electrons,
    frozen core orbitals, total occupied orbitals, correlated occupied orbitals, and virtual orbitals
    for a given molecule and basis set, with options for frozen core and MP2 type.

    Parameters
    ----------
        mol (psi4.core.Molecule): The Psi4 molecule object for which orbital counts are to be determined.
        basis (str): The basis set to use for the calculation (e.g., 'cc-pVDZ').
        reference (str): Reference wavefunction, e.g., rohf, uks
        freeze_core (bool, optional): Whether to freeze core orbitals in post-HF calculations, does not affect
        DFT or SCF methods. Defaults to False.
        use_wfn (bool, optional): Choose whether to use the wave function or not
        method (str, optional): Method used to classify the calculation type. Defaults to None
        int_type (str, optional): Type of method (e.g., 'DF', 'CD', etc.). If None, will use the Psi4 default

    Returns
    -------
        dict: A dictionary containing:

            - "nbasis" (int): Total number of basis functions.
            - "nmo" (str):Number of molecular orbitals
            - "nocc" (int): Number of active correlated (non-core) occupied orbitals.
            - "nvirt" (int): Number of virtual (inactive) orbitals.
            - "nalpha" (int): Number of active alpha electrons
            - "nbeta" (int): Number of active beta electrons
            - "ncore" (int): Number of frozen core orbitals
            - "necp" (int): Number of electrons replaced by ECPs
            - "stdout" (str): If use_wfn is True, output the psi4 output file

    """

    method = method.lower().replace("-", "").replace("_", "").replace("(", "").replace(")", "")
    
    options = {
        "reference": reference,
        "freeze_core": freeze_core,
        "PRINT": 5,
    }  # add 'num_frozen_docc': 2?

    flag_3c = method[-2:].lower() == "3c"
    if not flag_3c:
        options.update({"basis": basis})
    else:
        if not use_wfn:
            warnings.warn(
                "WFN must be used for estimating memory of *3c energy method."
            )
            use_wfn = True

    psi4.core.clean_options()
    psi4.set_options(options)

    if use_wfn:
        filename = "log.txt"
        try:
            psi4.core.set_output_file(filename, False)
            _, wfn = psi4.energy(
                "scf" if not flag_3c else method,
                molecule=mol,
                return_wfn=True,
            )
            # Read the captured output
            with open(filename, "r") as f:
                output = f.read()

        except Exception as e:
            # Read any output that was written before the error
            with open(filename, "r") as f:
                output = f.read()
            raise RuntimeError(f"Psi4 calculation failed: {output}") from e
        finally:
            # Clean up the temporary file
            if os.path.exists(filename):
                os.remove(filename)
        
        nbasis = wfn.basisset().nbf()
        nmo = wfn.nmo()
        ncore = wfn.nfrzc() # orbital
        necp = wfn.basisset().n_ecp_core()
        nalpha = wfn.nalpha() - ncore
        nbeta = wfn.nbeta() - ncore

    else:
        output = None
        if reference in ["rks", "uks"]:
            basisset = psi4.core.BasisSet.build(
                mol, "ORBITAL", basis, puream=psi4.core.get_global_option("PUREAM")
            )
        elif reference in ["rhf", "uhf", "rohf"]:
            basisset = psi4.core.BasisSet.build(
                mol, "BASIS", basis, puream=psi4.core.get_global_option("PUREAM")
            )
        else:
            # Default to "BASIS" if unknown reference
            basisset = psi4.core.BasisSet.build(
                mol, "BASIS", basis, puream=psi4.core.get_global_option("PUREAM")
            )

        nbasis = basisset.nbf()
        nmo = nbasis
        ncore = basisset.n_frozen_core()
        necp = basisset.n_ecp_core()
        nelectrons = int(
            sum(mol.Z(ii) for ii in range(mol.natom())) - mol.molecular_charge()
        )
        nsocc = mol.multiplicity() - 1
        nalpha = (nelectrons - 2 * ncore + nsocc) // 2
        nbeta = (nelectrons - 2 * ncore - nsocc) // 2

    if reference in ["rks", "rhf", "rohf"]:
        nocc = nalpha
        nvirt = nmo - nocc
    elif reference in ["uks", "uhf"]:
        nocc = nalpha + nbeta
        nvirt = nmo - max(nalpha, nbeta)

    # Remove temp files
    temp_files = glob.glob("*.clean")
    if temp_files:
        for filename in temp_files:
            os.remove(filename)

    orbitals = {
        "nbasis": nbasis,
        "nmo": nmo,
        "nocc": nocc,
        "nvirt": nvirt,
        "nalpha": nalpha,
        "nbeta": nbeta,
        "ncore": ncore,
        "necp": necp,
        "stdout": output,
    }
    
    if "3c" not in method:
        orbitals.update(get_auxiliary_basis_size(
            mol,
            basis,
            int_type=int_type,
            method=method,
        ))
    else:
        warnings.warn("3c methods use an external package. Memory estimates are likely greatly underestimated.")
        orbitals.update({"naux": 0, "aux_basis_used": "None"})
        
    return orbitals


def memory_from_psi4(
    mol,
    method="scf",
    basis=None,
    scf_type="df",
    mp2_type="df",
    cc_type="df",
    freeze_core=False,
    use_wfn=False,
    reference=None,
):
    """Calculate the maximum memory required for a Psi4 calculation.

    Parameters
    ----------
        mol (qcelemental.models.Molecule): Psi4 or QCElemental molecule
        method (str, optional): Energy method name, e.g., 'scf', 'mp2', 'ccsd', 'b3lyp'... Defaults to "scf".
        basis (_type_, optional): Basis set name. Defaults to None.
        scf_type (str, optional): SCF type (e.g., 'df'). Defaults to "df".
        mp2_type (str, optional): MP2 type (e.g., 'df'). Defaults to "df".
        cc_type (str, optional): CC type (e.g., 'df'). Defaults to "df".
        freeze_core (bool, optional): (For Psi4 1.10) Specifies how many core orbitals to freeze in correlated computations.
        TRUE or 1 will default to freezing the previous noble gas shell on each atom. In case of positive charges
        on fragments, an additional shell may be unfrozen, to ensure there are valence electrons in each fragment.
        With FALSE or 0, no electrons are frozen (with the exception of electrons treated by an ECP). With -1, -2,
        and -3, the user might request strict freezing of the previous first/second/third noble gas shell on every
        atom. In this case, when there are no valence electrons, the code raises an exception. Defaults to False.
        use_wfn (bool, optional): Choose whether to use the wave function or not
        reference (str, optional): Raference wavefunction (RHF, ROHF, UKS...)

    Returns
    -------
        float: Estimated memory in GB.
    """

    method = method.lower().replace("-", "").replace("_", "").replace("(", "").replace(")", "")

    flag_3c = True if method[-2:].lower() == "3c" else False
    if not flag_3c and (basis is None or basis == "None"):
        raise ValueError("No basis set was provided.")

    psi4_mol = (
        psi4.geometry(mol.to_string(dtype="psi4")) if isinstance(mol, Molecule) else mol
    )

    calc_type = get_calculation_type(method)
    if reference is None:
        if (
            calc_type == "CC"
            or calc_type == "MP2"
            or method.lower() not in mscf.DFT_METHODS
        ):
            reference = "rhf" if psi4_mol.multiplicity() == 1 else "rohf"
        else:  # DFT Method
            reference = "rks" if psi4_mol.multiplicity() == 1 else "uks"

    int_type = scf_type if scf_type is not None else mp2_type
    int_type = int_type if int_type is not None else cc_type
    orbital_counts = get_orbital_counts(
        psi4_mol,
        basis,
        reference,
        freeze_core=freeze_core,
        use_wfn=use_wfn,
        method=method,
        int_type=int_type
    )

    if method.lower() in mscf.METHODS:
        memory_breakdown = mscf.get_memory(
            orbital_counts["nbasis"],
            orbital_counts["nmo"],
            reference,
            method,
            int_type=scf_type,
            naux=orbital_counts["naux"],
        )
    elif method.lower() in mmp2.METHODS:
        memory_breakdown = mmp2.get_memory(
            orbital_counts["nbasis"],
            orbital_counts["nmo"],
            orbital_counts["nvirt"],
            orbital_counts["nocc"],
            reference,
            method,
            nalpha=orbital_counts["nalpha"],
            nbeta=orbital_counts["nbeta"],
            naux=orbital_counts["naux"],
            int_type=mp2_type,
        )
    elif method.lower() in mcc.METHODS:
        memory_breakdown = mcc.get_memory(
            orbital_counts["nbasis"],
            orbital_counts["nmo"],
            orbital_counts["nvirt"],
            orbital_counts["nocc"],
            reference,
            method,
            nalpha=orbital_counts["nalpha"],
            nbeta=orbital_counts["nbeta"],
            naux=orbital_counts["naux"],
            int_type=mp2_type,
        )
    else:
        raise ValueError(f"Unsupported method '{method}'.")

    memory_breakdown.update(orbital_counts)

    return memory_breakdown
