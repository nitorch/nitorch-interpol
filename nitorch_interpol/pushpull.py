"""
## Overview

These functions evaluate 1D, 2D or 3D continuous functions encoded
by B-splines at arbitrary continuous coordinates. The `pull` function is
highly related to [`scipy.ndimage.map_coordinates`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html)
and [`torch.grid_sample`](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html).
When `prefilter=False`, the input tensor `inp` is assumed to contain
spline coefficients, and the corresponding continuous function is sampled
at the coordinates contained in `grid`. If one wishes to sample the continuous
function that *interpolates* the 1D, 2D or 3D discrete signal stored in `inp`,
they should set `prefilter=True`, which first fits interpolating spline
coefficients to the input signal.

The function `push` is the numerical adjoint of `pull` with respect to the
first argument. It implements an operation commonly known as `splatting` in
computer vision: it assigns each value in `inp` at the corresponding location
stored in `grid`, with appropriate spline weighting.

---
"""
__all__ = ['pull', 'pullgrad', 'push', 'pushcount', 'reslice']
import torch
from torch import Tensor
from typing import Optional, Sequence
from jitfields.typing import OneOrSeveral, OrderType, BoundType, ExtrapolateType
from jitfields.pushpull import (
    pull, grad as pullgrad, push, count as pushcount
)
from .grids import affine_grid


def reslice(
    image: Tensor,
    affine: Tensor,
    affine_to: Tensor,
    shape: Optional[Sequence[int]] = None,
    order: OneOrSeveral[OrderType] = 2,
    bound: OneOrSeveral[BoundType] = 'dct2',
    extrapolate: ExtrapolateType = True,
    prefilter: bool = True,
) -> Tensor:
    """Reslice a spatial image to a different space (shape + affine).

    Parameters
    ----------
    image : `(..., *spatial, channels) tensor`
        Input image, with shape `(..., *spatial, channels)`.
    affine : `(..., ndim+1, ndim+1) tensor`
        Input orientation matrix, with shape  `(..., ndim+1, ndim+1)`.
    affine_to : `(..., ndim+1, ndim+1) tensor`
        Target orientation matrix, with shape `(..., ndim+1, ndim+1)`
    shape : `sequence[int]`, optional
        Target shape, with length `ndim`. Default: same as input shape

    Other Parameters
    ----------------
    order : `[sequence of] {0..7}`, default=2
        Interpolation order (per dimension).
    bound : `[sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`, default='dct2'
        How to deal with out-of-bound values (per dimension).
    extrapolate : `bool or {'center', 'edge'}`, default=False
        - `True`: use bound to extrapolate out-of-bound value
        - `False` or `'center'`: do not extrapolate values that fall outside
          of the centers of the first and last voxels.
        - `'edge'`: do not extrapolate values that fall outside
           of the edges of the first and last voxels.
    prefilter : `bool`, default=True
        Whether to first compute interpolating coefficients.
        Must be true for proper interpolation, otherwise this
        function merely performs a non-interpolating "spline sampling".

    Returns
    -------
    resliced : `(..., *shape, channels) tensors`
        Resliced image, with shape `(..., *shape, channels)`.

    """
    affine = affine.to(image)
    affine_to = affine_to.to(image)

    prm = dict(
        order=order,
        bound=bound,
        prefilter=prefilter,
        exptrapolate=extrapolate,
    )

    if shape is None:
        ndim = affine.shape[-1] - 1
        shape = image.shape[-ndim-1:-1]

    # perform reslicing
    transformation = torch.matmul(affine.inverse(), affine_to)
    grid = affine_grid(transformation, shape)
    image = pull(image, grid, **prm)
    return image