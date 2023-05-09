"""
## Overview

These function takes a discrete signal as input and return spline coefficients
such that the corresponding spline-encoded continuous function interpolates
the input signal. They are highly related to
[`scipy.ndimage.spline_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.spline_filter.html)
and
[`scipy.ndimage.spline_filter1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.spline_filte1dr.html).

---
"""
__all__ = [
    'spline_coeff', 'spline_coeff_',
    'spline_coeff_nd', 'spline_coeff_nd_',
]

from jitfields.splinc import (
    spline_coeff, spline_coeff_, spline_coeff_nd_, spline_coeff_nd
)