"""
Unit and regression test for the psi4_utilities package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import psi4_utilities


def test_psi4_utilities_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "psi4_utilities" in sys.modules
