"""Time utility functions.

This module provides functions for time formatting and conversion.
"""

from __future__ import absolute_import, division, print_function


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds.
    
    Example: 10239 -> (2, 50, 39)
    
    Args:
        t: Time in seconds
    
    Returns:
        tuple: (hours, minutes, seconds)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string.
    
    Example: 10239 -> '02h50m39s'
    
    Args:
        t: Time in seconds
    
    Returns:
        str: Formatted time string (e.g., '02h50m39s')
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)

