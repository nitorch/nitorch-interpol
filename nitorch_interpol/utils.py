from typing import List
from types import GeneratorType as generator
import torch
import os
import inspect


def ensure_list(x, size=None, crop=True, **kwargs):
    """Ensure that an object is a list (of size at last dim)

    If x is a list, nothing is done (no copy triggered).
    If it is a tuple, it is converted into a list.
    Otherwise, it is placed inside a list.
    """
    if not isinstance(x, (list, tuple, range, generator)):
        x = [x]
    elif not isinstance(x, list):
        x = list(x)
    if size and len(x) < size:
        default = kwargs.get('default', x[-1])
        x += [default] * (size - len(x))
    if size and crop:
        x = x[:size]
    return x


def make_vector(input, n=None, crop=True, *args,
                dtype=None, device=None, **kwargs):
    """Ensure that the input is a (tensor) vector and pad/crop if necessary.

    Parameters
    ----------
    input : scalar or sequence or generator
        Input argument(s).
    n : int, optional
        Target length.
    crop : bool, default=True
        Crop input sequence if longer than `n`.
    default : optional
        Default value to pad with.
        If not provided, replicate the last value.
    dtype : torch.dtype, optional
        Output data type.
    device : torch.device, optional
        Output device

    Returns
    -------
    output : tensor
        Output vector.

    """
    input = torch.as_tensor(input, dtype=dtype, device=device).flatten()
    if n is None:
        return input
    if n is not None and input.numel() >= n:
        return input[:n] if crop else input
    if args:
        default = args[0]
    elif 'default' in kwargs:
        default = kwargs['default']
    else:
        default = input[-1]
    default = input.new_full([], default).expand([n-len(input)])
    return torch.cat([input, default])


# There's been a bunch of changes to the meshgrid API across torch versions
# (plus, the API is different in torchscipt and plain torch).
# These functions should work consistently across versions.
if 'indexing' in inspect.signature(torch.meshgrid).parameters:

    def meshgrid_ij(*x):
        return torch.meshgrid(*x, indexing='ij')

    def meshgrid_xy(*x):
        return torch.meshgrid(*x, indexing='xy')

    if not int(os.environ.get('PYTORCH_JIT', '1')):
        def meshgrid_script_ij(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return meshgrid_ij(*x)

        def meshgrid_script_xy(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return meshgrid_xy(*x)

    else:
        @torch.jit.script
        def meshgrid_script_ij(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return torch.meshgrid(x, indexing='ij')

        @torch.jit.script
        def meshgrid_script_xy(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return torch.meshgrid(x, indexing='xy')

else:
    def meshgrid_ij(*x):
        return torch.meshgrid(*x)

    def meshgrid_xy(*x):
        grid = list(torch.meshgrid(*x))
        if len(grid) > 1:
            grid[0] = grid[0].transpose(0, 1)
            grid[1] = grid[1].transpose(0, 1)
        return grid

    if not int(os.environ.get('PYTORCH_JIT', '1')):
        def meshgrid_script_ij(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return meshgrid_ij(*x)

        def meshgrid_script_xy(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return meshgrid_xy(*x)

    else:
        @torch.jit.script
        def meshgrid_script_ij(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return torch.meshgrid(x)

        @torch.jit.script
        def meshgrid_script_xy(x: List[torch.Tensor]) -> List[torch.Tensor]:
            grid = torch.meshgrid(x)
            if len(grid) > 1:
                grid[0] = grid[0].transpose(0, 1)
                grid[1] = grid[1].transpose(0, 1)
            return grid
