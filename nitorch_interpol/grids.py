__all__ = [
    'cartesian_grid', 'cartesian_grid_like',
    'identity_grid', 'identity_grid_like',
    'add_identity_grid', 'add_identity_grid_',
    'sub_identity_grid', 'sub_identity_grid_',
    'affine_grid', 'affine_flow',
]
import torch
from torch import Tensor
from typing import Sequence, Optional, List
from .utils import meshgrid_script_ij


@torch.jit.script
def cartesian_grid(
        shape: Sequence[int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
) -> List[Tensor]:
    """Returns an (unstacked) identity coordinate grid.

    Parameters
    ----------
    shape : `sequence[int]`
        Spatial dimensions of the field, with length `ndim`.
    dtype : `torch.dtype`, default=`get_default_dtype()`
        Data type.
    device : `torch.device`, optional
        Device.

    Returns
    -------
    grid : `list[(*shape) tensor]`
        Components of the coordinate grid, with shape `shape`.

    """
    mesh1d = [torch.arange(float(s), dtype=dtype, device=device)
              for s in shape]
    grid = meshgrid_script_ij(mesh1d)
    return grid


@torch.jit.script
def cartesian_grid_like(
        flow: Tensor,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
) -> List[Tensor]:
    """Returns an (unstacked) identity coordinate grid.

    Parameters
    ----------
    flow : `(..., *spatial, dim) tensor`
        Input flow field.
    dtype : `torch.dtype`, default=`flow.dtype`
        Data type.
    device : `torch.device`, default=`flow.device`
        Device.

    Returns
    -------
    grid : `list[(*spatial) tensor]`
        Components of the coordinate grid, with shape `spatial`.

    """
    ndim = flow.shape[-1]
    shape = flow.shape[-ndim-1:-1]
    if dtype is None:
        dtype = flow.dtype
    if device is None:
        device = flow.device
    mesh1d = [torch.arange(float(s), dtype=dtype, device=device)
              for s in shape]
    return meshgrid_script_ij(mesh1d)


@torch.jit.script
def identity_grid(
        shape: Sequence[int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
) -> Tensor:
    """Returns a (stacked) identity coordinate grid.

    Parameters
    ----------
    shape : `sequence[int]`
        Spatial dimensions of the field, with length `ndim`.
    dtype : `torch.dtype`, default=`get_default_dtype()`
        Data type.
    device : `torch.device`, optional
        Device.

    Returns
    -------
    grid : `(*shape, dim) tensor`
        Coordinate grid, with shape `(*shape, dim)`.

    """
    return torch.stack(cartesian_grid(shape, dtype, device), dim=-1)


@torch.jit.script
def identity_grid_like(
        flow: Tensor,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
) -> Tensor:
    """Returns an identity coordinate grid consistent with the
    input flow field.

    Parameters
    ----------
    flow : `(..., *spatial, ndim) tensor`
        Input flow field, with shape `(..., *spatial, ndim)`.
    dtype : `torch.dtype`, default=`flow.dtype
        Data type.
    device : `torch.device`, default=`flow.device`
        Device.

    Returns
    -------
    grid : `(*spatial, ndim) tensor`
        Coordinate grid, with shape `(*spatial, ndim)`.

    """
    return torch.stack(cartesian_grid_like(flow, dtype, device), dim=-1)


@torch.jit.script
def add_identity_grid_(disp):
    """Adds the identity grid to a displacement field, inplace.

    Parameters
    ----------
    disp : `(..., *spatial, ndim) tensor`
        Displacement field, with shape `(..., *spatial, ndim)`.

    Returns
    -------
    grid : `(..., *spatial, ndim) tensor`
        Transformation field, with shape `(..., *spatial, ndim)`.

    """
    grid = cartesian_grid_like(disp)
    disp = torch.movedim(disp, -1, 0)
    for i, grid1 in enumerate(grid):
        disp[i].add_(grid1)
    disp = torch.movedim(disp, 0, -1)
    return disp


@torch.jit.script
def add_identity_grid(disp):
    """Adds the identity grid to a displacement field.

    Parameters
    ----------
    disp : `(..., *spatial, ndim) tensor`
        Displacement field, with shape `(..., *spatial, ndim)`.

    Returns
    -------
    grid : `(..., *spatial, ndim) tensor`
        Transformation field, with shape `(..., *spatial, ndim)`.

    """
    return add_identity_grid_(disp.clone())


@torch.jit.script
def sub_identity_grid_(disp):
    """Subtracts the identity grid to a displacement field, inplace.

    Parameters
    ----------
    disp : (`..., *spatial, ndim) tensor`
        Displacement field, with shape `(..., *spatial, ndim)`.

    Returns
    -------
    grid : `(..., *spatial, ndim) tensor`
        Transformation field, with shape `(..., *spatial, ndim)`.

    """
    grid = cartesian_grid_like(disp)
    disp = torch.movedim(disp, -1, 0)
    for i, grid1 in enumerate(grid):
        disp[i].sub_(grid1)
    disp = torch.movedim(disp, 0, -1)
    return disp


@torch.jit.script
def sub_identity_grid(disp):
    """Subtracts the identity grid to a displacement field.

    Parameters
    ----------
    disp : `(..., *spatial, ndim) tensor`
        Displacement field, with shape `(..., *spatial, ndim)`.

    Returns
    -------
    grid : `(..., *spatial, ndim) tensor`
        Transformation field, with shape `(..., *spatial, ndim)`.

    """
    return sub_identity_grid_(disp.clone())


def affine_grid(mat, shape):
    """Create a dense coordinate grid from an affine matrix.

    Parameters
    ----------
    mat : `(..., ndim+1, ndim+1) tensor`
        Affine matrix (or matrices), with shape (..., ndim+1, ndim+1)`.
    shape : (ndim,) sequence[int]
        Shape of the grid, with length `ndim`.

    Returns
    -------
    grid : `(..., *shape, ndim) tensor`
        Dense coordinate grid, with shape `(..., *shape, ndim)`

    """
    mat = torch.as_tensor(mat)
    shape = list(shape)
    nb_dim = mat.shape[-1] - 1
    if nb_dim != len(shape):
        raise ValueError('Dimension of the affine matrix ({}) and shape ({}) '
                         'are not the same.'.format(nb_dim, len(shape)))
    if mat.shape[-2] not in (nb_dim, nb_dim+1):
        raise ValueError('First argument should be matrces of shape '
                         '(..., {0}, {1}) or (..., {1], {1}) but got {2}.'
                         .format(nb_dim, nb_dim+1, mat.shape))
    batch_shape = mat.shape[:-2]
    grid = identity_grid(shape, mat.dtype, mat.device)
    if batch_shape:
        for _ in range(len(batch_shape)):
            grid = grid[None]
        for _ in range(nb_dim):
            mat = mat[..., None, :, :]
    lin = mat[..., :nb_dim, :nb_dim]
    off = mat[..., :nb_dim, -1]
    grid = lin.matmul(grid.unsqueeze(-1)).squeeze(-1) + off
    return grid


def affine_flow(mat, shape):
    """Create a dense displacement field (in voxels) from an affine matrix.

    Parameters
    ----------
    mat : `(..., ndim+1, ndim+1) tensor`
        Affine matrix (or matrices), with shape (..., ndim+1, ndim+1)`.
    shape : (ndim,) sequence[int]
        Shape of the grid, with length `ndim`.

    Returns
    -------
    grid : `(..., *shape, ndim) tensor`
        Dense displacement field, with shape `(..., *shape, ndim)`

    """
    return sub_identity_grid_(affine_grid(mat, shape))