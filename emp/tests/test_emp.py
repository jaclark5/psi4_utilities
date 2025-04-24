"""
Unit and regression test for the emp package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import emp


def test_emp_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "emp" in sys.modules
