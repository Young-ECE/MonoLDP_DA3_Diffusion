"""File utility functions.

This module provides functions for file I/O operations.
"""

from __future__ import absolute_import, division, print_function


def readlines(filename):
    """Read all the lines in a text file and return as a list.
    
    Args:
        filename: Path to the text file
    
    Returns:
        list: List of lines (strings) from the file
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

