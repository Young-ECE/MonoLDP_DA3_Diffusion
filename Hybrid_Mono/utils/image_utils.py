"""Image utility functions.

This module provides functions for image processing and normalization.
"""

from __future__ import absolute_import, division, print_function


def normalize_image(x):
    """Rescale image pixels to span range [0, 1].
    
    Args:
        x: Image tensor
    
    Returns:
        Normalized image tensor with values in [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d

