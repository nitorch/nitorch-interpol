"""
## Overview

Resizing is highly related to resampling, except that sampling happens
on a regular grid of coordinates, which can be finer or coarser than the
input lattice. The `resize` function is related to
[`scipy.ndimage.zoom`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html)
and
[`torch.interpolate`](https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html).

The function `restrict` is the numerical adjoint of `resize` with respect
to the first argument (when `reduce_sum=True`).

### Anchors

Resizing operator have slightly different behaviours depending on which
elements of the lattice are kept in alignment across resolutions. This
relates to the `grid_mode` option in
[`scipy.ndimage.zoom`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html),
or the `align_corners` options in
[`torch.interpolate`](https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html).
In `nitorch`, we use the option `anchor`, which can take values `"edge"`,
`"center"` or `None`. When either `"edge"` or `"center"` is used, the
effective resolution ratio will slightly differ from the prescribed
`factor` (which is then only used to compute the shape of the resized
image). The first option (`"edge"`) aligns the edges of the first and
last voxels across resolutions, while the second option (`"center"`)
aligns the centers of the corner voxels.  When `None` is used, the
center of the top-right corner is aligned, and the prescribed `factor`
is used exactly.

```
         edge           center           None
    e - + - + - e   + - + - + - +   + - + - + - +
    | . | . | . |   | c | . | c |   | f | . | . |
    + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +
    | . | . | . |   | . | . | . |   | . | . | . |
    + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +
    | . | . | . |   | c | . | c |   | . | . | . |
    e _ + _ + _ e   + _ + _ + _ +   + _ + _ + _ +
```

---
"""
__all__ = [
    'resize', 'restrict',
    'resize_flow', 'restrict_flow',
    'resize_affine', 'resize_affine_shape',
    'restrict_affine', 'restrict_affine_shape',
]
from jitfields.resize import resize, restrict
from jitfields.utils import ensure_list
from jitfields.typing import OneOrSeveral, BoundType, OrderType, AnchorType
import torch
from torch import Tensor
from typing import Optional, Union, Literal, Sequence, List, Tuple
import math


def resize_flow(
    flow: Tensor,
    factor: Optional[OneOrSeveral[float]] = None,
    shape: Optional[OneOrSeveral[int]] = None,
    anchor: AnchorType = 'edge',
    order: OneOrSeveral[OrderType] = 2,
    bound: OneOrSeveral[BoundType] = 'dft',
    prefilter: bool = True,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Resize a displacement field (in voxels) using spline interpolation.

    The displacement values are also modulated to account for
    resolution change.

    Parameters
    ----------
    flow : `(..., *inshape, ndim) tensor`
        Input displacement field, with shape `(..., *inshape, ndim)`.
    factor : `[sequence of] float`, optional
        Factor by which to resize the tensor (> 1 == bigger)
        One of factor or shape must be provided.
    shape : `[sequence of] float`, optional
        Shape of output tensor.
        One of factor or shape must be provided.
    anchor : `{'edge', 'center'} or None`
        What feature should be aligned across the input and output tensors.
        If 'edge' or 'center', the effective scaling factor may slightly
        differ from the requested scaling factor.
        If None, the center of the (0, 0) voxel is aligned, and the
        requested factor is exactly applied.
    order : `[sequence of] {0..7}`, default=2
        Interpolation order.
    bound : `[sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`, default='dft'
        How to deal with out-of-bound values.
    prefilter : `bool`, default=True
        Whether to first compute interpolating coefficients.
        Must be true for proper interpolation, otherwise this
        function merely performs a non-interpolating "prolongation".
    out: `(..., *shape, ndim) tensor`, optional
        Output placeholder.

    Returns
    -------
    flow : `(..., *shape, ndim) tensor`
        Resized displacement field, with shape `(..., *shape, ndim)`.

    """
    ndim = flow.shape[-1]
    inshape = flow.shape[-ndim-1:-1]
    shape, scale, shift = _resize_shape_scale_shift(inshape, factor, shape, anchor)

    if out is not None:
        out = out.movedim(-1, 0)

    out = resize(
        flow.movedim(-1, 0),
        factor=factor,
        shape=shape,
        ndim=ndim,
        anchor=anchor,
        order=order,
        bound=bound,
        prefilter=prefilter,
        out=out,
    ).movedim(0, -1)

    out *= torch.as_tensor(scale, dtype=out.dtype, device=out.device)

    return out


def restrict_flow(
    flow: Tensor,
    factor: Optional[OneOrSeveral[float]] = None,
    shape: Optional[OneOrSeveral[int]] = None,
    anchor: AnchorType = 'edge',
    order: OneOrSeveral[OrderType] = 1,
    bound: OneOrSeveral[BoundType] = 'dft',
    make_adjoint: Union[bool, Literal[-1, 0, 1]] = False,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Restrict (adjoint of resize) a displacement field using spline interpolation

    Parameters
    ----------
    flow : `(..., *inshape, ndim) tensor`
        Input displacement field, with shape `(..., *inshape, ndim)`.
    factor : `[sequence of] float`, optional
        Factor by which to resize the tensor (> 1 == smaller)
        One of factor or shape must be provided.
    shape : `[sequence of] float`, optional
        Shape of output tensor.
        One of factor or shape must be provided.
    anchor : `{'edge', 'center'} or None`
        What feature should be aligned across the input and output tensors.
        If 'edge' or 'center', the effective scaling factor may slightly
        differ from the requested scaling factor.
        If None, the center of the (0, 0) voxel is aligned, and the
        requested factor is exactly applied.
    order : `[sequence of] {0..7}`, default=2
        Interpolation order.
    bound : `[sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}`, default='dft'
        How to deal with out-of-bound values.
    make_adjoint : `{-1, 0, 1}`, default=0
        - If `0`/`False`, assume that the input is a flow (in voxels),
        and normalize so that a displacement of one input voxel is transformed
        into a displacement of `1/scale` output voxels (where `scale`
        is the true scaling of the transformation, which may differ slightly
        from the input `factor`).
        - If `1`/`True`, ensures that `restrict_flow` is the numeric
        adjoint of `resize_flow`.
        - If `-1`, assume that the input is a gradient with respect to
        a flow (in 1/voxels) and apply the inverse normalization of the
        `0` case.
    out: `(..., *shape, ndim) tensor`, optional
        Output placeholder.

    Returns
    -------
    x : `(..., *shape, ndim) tensor`
        Restricted displacement field, with shape `(..., *shape, ndim)`.

    """
    make_adjoint = int(make_adjoint)
    ndim = flow.shape[-1]
    inshape = flow.shape[-ndim-1:-1]
    shape, scale, shift = _restrict_shape_scale_shift(inshape, factor, shape, anchor)

    if out is not None:
        out = out.movedim(-1, 0)

    out = restrict(
        flow.movedim(-1, 0),
        factor=factor,
        shape=shape,
        ndim=ndim,
        anchor=anchor,
        order=order,
        bound=bound,
        make_adjoint=make_adjoint == 1,  # only if true numeric adjoint
        out=out,
    ).movedim(0, -1)

    if make_adjoint == 0:
        out *= torch.as_tensor(scale, dtype=out.dtype, device=out.device)
    else:
        # We want to apply the exact same factor as in the corresponding
        # `resize`. This `scale` contains the inverse of resize's `scale`.
        out /= torch.as_tensor(scale, dtype=out.dtype, device=out.device)

    return out


def resize_affine(
        affine: Tensor,
        inshape: Sequence[int],
        factor: Optional[OneOrSeveral[float]] = None,
        shape: Optional[OneOrSeveral[int]] = None,
        anchor: AnchorType = 'edge',
) -> Tensor:
    """Compute the orientation matrix of a resized tensor.

    Parameters
    ----------
    affine : `(ndim+1, ndim+1) tensor`
        Orientation matrix of the input tensor, with shape `(ndim+1, ndim+1)`.
    inshape : `sequence[int]`
        Spatial shape of the input tensor, with length `ndim`.
    factor : `[sequence of] float`, optional
        Factor by which to resize the tensor (> 1 == bigger)
        One of factor or shape must be provided.
    shape : `[sequence of] float`, optional
        Shape of output tensor.
        One of factor or shape must be provided.
    anchor : `{'edge', 'center'} or None`
        What feature should be aligned across the input and output tensors.
        If 'edge' or 'center', the effective scaling factor may slightly
        differ from the requested scaling factor.
        If None, the center of the (0, 0) voxel is aligned, and the
        requested factor is exactly applied.

    Returns
    -------
    affine : `(ndim+1, ndim+1) tensor`
        Orientation matrix of the resized tensor,
        with shape `(ndim+1, ndim+1) tensor`.

    """
    return resize_affine_shape(inshape, affine, factor, shape, anchor)[0]


def resize_affine_shape(
        inshape: Sequence[int],
        affine: Tensor,
        factor: Optional[OneOrSeveral[float]] = None,
        shape: Optional[OneOrSeveral[int]] = None,
        anchor: AnchorType = 'edge',
) -> Tuple[Tensor, torch.Size]:
    """Compute the orientation matrix and shape of a resized tensor.

    Parameters
    ----------
    inshape : `sequence[int]`
        Spatial shape of the input tensor, with length `ndim`.
    affine : `(ndim+1, ndim+1) tensor`
        Orientation matrix of the input tensor, with shape `(ndim+1, ndim+1)`.
    factor : `[sequence of] float`, optional
        Factor by which to resize the tensor (> 1 == bigger)
        One of factor or shape must be provided.
    shape : `[sequence of] float`, optional
        Shape of output tensor.
        One of factor or shape must be provided.
    anchor : `{'edge', 'center'} or None`
        What feature should be aligned across the input and output tensors.
        If 'edge' or 'center', the effective scaling factor may slightly
        differ from the requested scaling factor.
        If None, the center of the (0, 0) voxel is aligned, and the
        requested factor is exactly applied.

    Returns
    -------
    affine : `(ndim+1, ndim+1) tensor`
        Orientation matrix of the resized tensor,
        with shape `(ndim+1, ndim+1) tensor`.
    shape : `tuple[int]`
        Spatial shape of the resized tensor.

    """
    inshape = list(inshape)
    ndim = len(inshape)
    outshape, scale, shift = _resize_shape_scale_shift(inshape, factor, shape, anchor)

    vox2vox = torch.eye(ndim+1, dtype=affine.dtype, device=affine.device)

    # scale
    scale = 1 / torch.as_tensor(scale)
    vox2vox.diagonal(0, -1, -2)[:-1].copy_(scale)
    # translation
    shift = torch.as_tensor(shift) * (scale - 1)
    vox2vox[:-1, -1].copy_(shift)

    return torch.matmul(affine, vox2vox), torch.Size(outshape)


def restrict_affine(
        inshape: Sequence[int],
        affine: Tensor,
        factor: Optional[OneOrSeveral[float]] = None,
        shape: Optional[OneOrSeveral[int]] = None,
        anchor: AnchorType = 'edge',
) -> Tensor:
    """Compute the orientation matrix of a restricted tensor.

    Parameters
    ----------
    inshape : `sequence[int]`
        Spatial shape of the input tensor, with length `ndim`.
    affine : `(ndim+1, ndim+1) tensor`
        Orientation matrix of the input tensor, with shape `(ndim+1, ndim+1)`.
    factor : `[sequence of] float`, optional
        Factor by which to resize the tensor (> 1 == bigger)
        One of factor or shape must be provided.
    shape : `[sequence of] float`, optional
        Shape of output tensor.
        One of factor or shape must be provided.
    anchor : `{'edge', 'center'} or None`
        What feature should be aligned across the input and output tensors.
        If 'edge' or 'center', the effective scaling factor may slightly
        differ from the requested scaling factor.
        If None, the center of the (0, 0) voxel is aligned, and the
        requested factor is exactly applied.

    Returns
    -------
    affine : `(ndim+1, ndim+1) tensor`
        Orientation matrix of the resized tensor,
        with shape `(ndim+1, ndim+1) tensor`.

    """
    return restrict_affine_shape(inshape, affine, factor, shape, anchor)[0]


def restrict_affine_shape(
        inshape: Sequence[int],
        affine: Tensor,
        factor: Optional[OneOrSeveral[float]] = None,
        shape: Optional[OneOrSeveral[int]] = None,
        anchor: AnchorType = 'edge',
) -> Tuple[Tensor, torch.Size]:
    """Compute the orientation matrix and shape of a restricted tensor.

    Parameters
    ----------
    inshape : `sequence[int]`
        Spatial shape of the input tensor, with length `ndim`.
    affine : `(ndim+1, ndim+1) tensor`
        Orientation matrix of the input tensor, with shape `(ndim+1, ndim+1)`.
    factor : `[sequence of] float`, optional
        Factor by which to resize the tensor (> 1 == bigger)
        One of factor or shape must be provided.
    shape : `[sequence of] float`, optional
        Shape of output tensor.
        One of factor or shape must be provided.
    anchor : `{'edge', 'center'} or None`
        What feature should be aligned across the input and output tensors.
        If 'edge' or 'center', the effective scaling factor may slightly
        differ from the requested scaling factor.
        If None, the center of the (0, 0) voxel is aligned, and the
        requested factor is exactly applied.

    Returns
    -------
    affine : `(ndim+1, ndim+1) tensor`
        Orientation matrix of the resized tensor,
        with shape `(ndim+1, ndim+1) tensor`.
    shape : `tuple[int]`
        Spatial shape of the resized tensor.

    """
    inshape = list(inshape)
    ndim = len(inshape)
    outshape, scale, shift = _restrict_shape_scale_shift(inshape, factor, shape, anchor)

    vox2vox = torch.eye(ndim+1, dtype=affine.dtype, device=affine.device)

    # scale
    scale = 1 / torch.as_tensor(scale)
    vox2vox.diagonal(0, -1, -2)[:-1].copy_(scale)
    # translation
    shift = torch.as_tensor(shift) * (scale - 1)
    vox2vox[:-1, -1].copy_(shift)

    return torch.matmul(affine, vox2vox), torch.Size(outshape)


def _resize_shape_scale_shift(inshape, factor=None, shape=None, anchor='e'):
    """Compute the output shape, true scaling factor, and coordinate shift."""
    ndim = len(inshape)
    shape = ensure_list(shape, ndim)
    factor = ensure_list(factor, ndim)
    if not shape and not factor:
        raise ValueError('At least one of shape or factor must be provided')
    shape = [so or math.ceil(si*f) for so, si, f in zip(shape, inshape, factor)]
    anchor = ensure_list(anchor, ndim)

    scale = []
    shift = []
    for a, so, si, f in zip(anchor, shape, inshape, factor):
        a = a[0].lower() if a else ''
        if a == 'e':
            scale += [so / si]
            shift += [0.5]
        elif a == 'c':
            scale += [(so - 1) / (si - 1)]
            shift += [0]
        else:
            scale += [f]
            shift += [0]
    return shape, scale, shift


def _restrict_shape_scale_shift(inshape, factor=None, shape=None, anchor='e'):
    """Compute the output shape, true scaling factor, and coordinate shift."""
    ndim = len(inshape)
    shape = ensure_list(shape, ndim)
    factor = ensure_list(factor, ndim)
    if not shape and not factor:
        raise ValueError('At least one of shape or factor must be provided')
    shape = [so or math.ceil(si/f) for so, si, f in zip(shape, inshape, factor)]
    anchor = ensure_list(anchor, ndim)

    scale = []
    shift = []
    for a, so, si, f in zip(anchor, shape, inshape, factor):
        a = a[0].lower() if a else ''
        if a == 'e':
            scale += [so / si]
            shift += [0.5]
        elif a == 'c':
            scale += [(so - 1) / (si - 1)]
            shift += [0]
        else:
            scale += [1/f]
            shift += [0]
    return shape, scale, shift