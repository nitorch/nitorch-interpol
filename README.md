# nitorch-interpol

_High-order spline interpolation (and utilities)_

## Overview

This package implements tools for resampling dense images or volumes
using spline interpolation (order 0 to 7). It relies on pure C++/CUDA
routines implemented in [`jitfields`](https://github.com/balbasty/jitfields),
with dependencies on [`cppyy`](https://github.com/wlav/cppyy) and
[`cupy`](https://github.com/cupy/cupy), which allow C++ and CUDA code to
be compiled just-in-time.

If you are looking for a more lightweight package implemented in pure
PyTorch, you may want to check out
[`torch-interpol`](https://github.com/balbasty/torch-interpol).


## Installation

Installation through pip should work, although I don't know how robust the cupy/pytorch
interaction is in term of cuda version.
```sh
pip install git+https://github.com/nitorch/nitorch-interpol
```

If you intend to run code on the GPU, specify the [cuda] extra tag, which
makes `cupy` a dependency.
```sh
pip install "nitorch-interpol[cuda] @ git+https://github.com/nitorch/nitorch-interpol"
```

Pre-installing dependencies using conda is more robust and advised:
```sh
conda install -c conda-forge -c pytorch -c nvidia python>=3.6 numpy cupy ccpyy pytorch>=1.8 cudatoolkit=11.1
pip install "nitorch-interpol[cuda] @ git+https://github.com/balbasty/nitorch-interpol"
```

## API

All API functions live under `nitorch_interpol` and are imported by default.
_E.g._
```python
from nitorch_interpol import pull
resampled = pull(signal, coordinates, order=3, prefilter=True)
```
or
```python
import nitorch_interpol as interpol
resampled = interpol.pull(signal, coordinates, order=3, prefilter=True)
```

#### Common options

`order` can be an int or a string. Possible values are:
    - 0 or 'nearest'
    - 1 or 'linear'
    - 2 or 'quadratic'
    - 3 or 'cubic'
    - 4 or 'fourth'
    - 5 or 'fifth'
    - 6 or 'sixth'
    - 7 or 'seventh'
A list of values can be provided, in the order [W, H, D],
to specify dimension-specific interpolation orders.

`bound` must be a string. Possible values are:
| FFT-like name | SciPy-like name |  Extrapolation pattern                |
| ------------- | --------------- | ------------------------------------- |
| 'dct1'        | 'mirror'        | ` d  c  b  |  a  b  c  d  |  c  b  a` |
| 'dct2'        | 'reflect'       | ` c  b  a  |  a  b  c  d  |  d  c  b` |
| 'dst1'        | 'antimirror'    | `-b -a  0  |  a  b  c  d  |  0 -d -c` |
| 'dst2'        | 'antireflect'   | `-c -b -a  |  a  b  c  d  | -d -c -b` |
| 'dft'         | 'wrap'          | ` b  c  d  |  a  b  c  d  |  a  b  c` |
| 'zero'        | 'zeros'         | ` 0  0  0  |  a  b  c  d  |  0  0  0` |
| 'replicate'   | 'nearest'       | ` a  a  a  |  a  b  c  d  |  d  d  d` |
A list of values can be provided, in the order [W, H, D],
to specify dimension-specific boundary conditions.
Note that
- `dft` corresponds to circular padding
- `dct2` corresponds to Neumann boundary conditions (symmetric)
- `dst2` corresponds to Dirichlet boundary conditions (antisymmetric)
See:
 - https://en.wikipedia.org/wiki/Discrete_cosine_transform
 - https://en.wikipedia.org/wiki/Discrete_sine_transform

### Resampling

These functions evaluate 1D, 2D or 3D continuous functions encoded
by B-splines at arbitrary continuous coordinates. The `pull` function is
highly related to [`scipy.ndimage.map_coordinates`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html).
When `prefilter=False`, the input tensor `inp` is assumed to contain
spline coefficients, and the corresponding continuous function is sampled
at the coordinates contained in `grid`. If one wishes to sample the continuous
function that *interpolates* the 1D, 2D or 3D discrete signal stored in `inp`,
they should set `prefilter=True`, which first fits interpolating spline
coefficients to the input signal.

```python
def pull(inp, grid, order=2, bound='dct2', extrapolate=True, prefilter=False, out=None): ...
"""Sample a tensor using spline interpolation

Parameters
----------
inp : (..., *inshape, channel) tensor
    Input tensor
grid : (..., *outshape, ndim) tensor
    Tensor of coordinates into `inp`
order : [sequence of] {0..7}, default=2
    Interpolation order.
bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dct2'
    How to deal with out-of-bound values.
extrapolate : bool or {'center', 'edge'}
    - True: use bound to extrapolate out-of-bound value
    - False or 'center': do not extrapolate values that fall outside
      of the centers of the first and last voxels.
    - 'edge': do not extrapolate values that fall outside
       of the edges of the first and last voxels.
prefilter : bool, default=True
    Whether to first compute interpolating coefficients.
    Must be true for proper interpolation, otherwise this
    function merely performs a non-interpolating "spline sampling".

Returns
-------
out : (..., *outshape, channel) tensor
    Pulled tensor

"""
```

The function `push` is the numerical adjoint of `pull` with respect to the
first argument. It implements an operation commonly known as `splatting` in
computer vision: it assigns each value in `inp` at the corresponding location
stored in `grid`, with appropriate spline weighting.

```python
def push(inp, grid, shape=None, order=2, bound='dct2', extrapolate=True, prefilter=False, out=None): ...
"""Splat a tensor using spline interpolation

Parameters
----------
inp : (..., *inshape, channel) tensor
    Input tensor
grid : (..., *inshape, ndim) tensor
    Tensor of coordinates into `inp`
shape : sequence[int], default=inshape
    Output spatial shape
order : [sequence of] {0..7}, default=2
    Interpolation order.
bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dct2'
    How to deal with out-of-bound values.
extrapolate : bool or {'center', 'edge'}
    - True: use bound to extrapolate out-of-bound value
    - False or 'center': do not extrapolate values that fall outside
      of the centers of the first and last voxels.
    - 'edge': do not extrapolate values that fall outside
       of the edges of the first and last voxels.
    prefilter : bool, default=True
        Whether to compute interpolating coefficients at the end.

Returns
-------
out : (..., *shape, channel) tensor
    Pulled tensor
"""
```

```python
def count(grid, shape=None, order=2, bound='dct2', extrapolate=True, out=None): ...
"""Splat ones using spline interpolation

Parameters
----------
grid : (..., *inshape, ndim) tensor
    Tensor of coordinates
shape : sequence[int], default=inshape
    Output spatial shape
order : [sequence of] {0..7}, default=2
    Interpolation order.
bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dct2'
    How to deal with out-of-bound values.
extrapolate : bool or {'center', 'edge'}
    - True: use bound to extrapolate out-of-bound value
    - False or 'center': do not extrapolate values that fall outside
      of the centers of the first and last voxels.
    - 'edge': do not extrapolate values that fall outside
       of the edges of the first and last voxels.

Returns
-------
out : (..., *shape) tensor
    Pulled tensor
"""
```

```python
def grad(inp, grid, order=2, bound='dct2', extrapolate=True, prefilter=False, out=None): ...
"""Sample the spatial gradients of a tensor using spline interpolation

Parameters
----------
inp : (..., *inshape, channel) tensor
    Input tensor
grid : (..., *outshape, ndim) tensor
    Tensor of coordinates into `inp`
order : [sequence of] {0..7}, default=2
    Interpolation order.
bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dct2'
    How to deal with out-of-bound values.
extrapolate : bool or {'center', 'edge'}
    - True: use bound to extrapolate out-of-bound value
    - False or 'center': do not extrapolate values that fall outside
      of the centers of the first and last voxels.
    - 'edge': do not extrapolate values that fall outside
       of the edges of the first and last voxels.
prefilter : bool, default=True
    Whether to first compute interpolating coefficients.
    Must be true for proper interpolation, otherwise this
    function merely performs a non-interpolating "spline sampling".

Returns
-------
out : (..., *outshape, channel, ndim) tensor
    Pulled gradients
"""
```

### Resizing

Resizing is highly related to resampling, except that sampling happens
on a regular grid of coordinates, which can be finer or coarser than the
input lattice. The `resize` function is related to
[`scipy.ndimage.zoom`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html).

```python
def resize(x, factor=None, shape=None, ndim=None,
           anchor='e', order=2, bound='dct2', prefilter=True): ...
"""Resize a tensor using spline interpolation

Parameters
----------
x : (..., *inshape) tensor
    Input  tensor
factor : [sequence of] float, optional
    Factor by which to resize the tensor (> 1 == bigger)
    One of factor or shape must be provided.
shape : [sequence of] float, optional
    Shape of output tensor.
    One of factor or shape must be provided.
ndim : int, optional
    Number if spatial dimensions.
    If not provided, try to guess from factor or shape.
    If guess fails, assume ndim = x.dim().
anchor : {'edge', 'center'} or None
    What feature should be aligned across the input and output tensors.
    If 'edge' or 'center', the effective scaling factor may slightly
    differ from the requested scaling factor.
    If None, the center of the (0, 0) voxel is aligned, and the
    requested factor is exactly applied.
order : [sequence of] {0..7}, default=2
    Interpolation order.
bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dct2'
    How to deal with out-of-bound values.
prefilter : bool, default=True
    Whether to first compute interpolating coefficients.
    Must be true for proper interpolation, otherwise this
    function merely performs a non-interpolating "prolongation".

Returns
-------
x : (..., *shape) tensor
    Resized tensor

References
----------
..[1]  M. Unser, A. Aldroubi and M. Eden.
       "B-Spline Signal Processing: Part I-Theory,"
       IEEE Transactions on Signal Processing 41(2):821-832 (1993).
..[2]  M. Unser, A. Aldroubi and M. Eden.
       "B-Spline Signal Processing: Part II-Efficient Design and Applications,"
       IEEE Transactions on Signal Processing 41(2):834-848 (1993).
..[3]  M. Unser.
       "Splines: A Perfect Fit for Signal and Image Processing,"
       IEEE Signal Processing Magazine 16(6):22-38 (1999).
"""
```

The function `restrict` is the numerical adjoint of `resize` with respect
to the first argument (when `reduce_sum=True`).


```python
def restrict(x, factor=None, shape=None, ndim=None,
             anchor='e', order=2, bound='dct2', reduce_sum=False): ...
"""Restrict (adjoint of resize) a tensor using spline interpolation

Parameters
----------
x : (..., *inshape) tensor
    Input  tensor
factor : [sequence of] float, optional
    Factor by which to resize the tensor (> 1 == smaller)
    One of factor or shape must be provided.
shape : [sequence of] float, optional
    Shape of output tensor.
    One of factor or shape must be provided.
ndim : int, optional
    Number if spatial dimensions.
    If not provided, try to guess from factor or shape.
    If guess fails, assume ndim = x.dim().
anchor : {'edge', 'center'} or None
    What feature should be aligned across the input and output tensors.
    If 'edge' or 'center', the effective scaling factor may slightly
    differ from the requested scaling factor.
    If None, the center of the (0, 0) voxel is aligned, and the
    requested factor is exactly applied.
order : [sequence of] {0..7}, default=2
    Interpolation order.
bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft'}, default='dct2'
    How to deal with out-of-bound values.

Returns
-------
x : (..., *shape) tensor
    restricted tensor
"""
```

### Fast spline interpolation

These function takes a discrete signal as input and return spline coefficients
such that the corresponding spline-encoded continuous function interpolates
the input signal. They are highly related to
[`scipy.ndimage.spline_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.spline_filter.html)
and
[`scipy.ndimage.spline_filter1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.spline_filte1dr.html)

```python
def spline_coeff(inp, order, bound='dct2', dim=-1): ...
"""Compute the interpolating spline coefficients, along a single dimension.

Parameters
----------
inp : tensor
    Input tensor
order : {0..7}, default=2
    Interpolation order.
bound : {'zero', 'replicate', 'dct1', 'dct2', 'dft'}, default='dct2'
    Boundary conditions.
dim : int, default=-1
    Dimension along which to filter

Returns
-------
coeff : tensor
    Spline coefficients

References
----------
..[1]  M. Unser, A. Aldroubi and M. Eden.
       "B-Spline Signal Processing: Part I-Theory,"
       IEEE Transactions on Signal Processing 41(2):821-832 (1993).
..[2]  M. Unser, A. Aldroubi and M. Eden.
       "B-Spline Signal Processing: Part II-Efficient Design and Applications,"
       IEEE Transactions on Signal Processing 41(2):834-848 (1993).
..[3]  M. Unser.
       "Splines: A Perfect Fit for Signal and Image Processing,"
       IEEE Signal Processing Magazine 16(6):22-38 (1999).
"""

def spline_coeff_(inp, order, bound='dct2', dim=-1): ...
"""In-place version of `spline_coeff`."""
```

```python
def spline_coeff_nd(inp, order, bound='dct2', ndim=None): ...
"""Compute the interpolating spline coefficients, along the last N dimensions.

Parameters
----------
inp : (..., *spatial) tensor
    Input tensor
order : [sequence of] {0..7}, default=2
    Interpolation order.
bound : [sequence of] {'zero', 'replicate', 'dct1', 'dct2', 'dft'}, default='dct2'
    Boundary conditions.
ndim : int, default=`inp.dim()`
    Number of spatial dimensions

Returns
-------
coeff : (..., *spatial) tensor
    Spline coefficients

References
----------
..[1]  M. Unser, A. Aldroubi and M. Eden.
       "B-Spline Signal Processing: Part I-Theory,"
       IEEE Transactions on Signal Processing 41(2):821-832 (1993).
..[2]  M. Unser, A. Aldroubi and M. Eden.
       "B-Spline Signal Processing: Part II-Efficient Design and Applications,"
       IEEE Transactions on Signal Processing 41(2):834-848 (1993).
..[3]  M. Unser.
       "Splines: A Perfect Fit for Signal and Image Processing,"
       IEEE Signal Processing Magazine 16(6):22-38 (1999).
"""

def spline_coeff_nd_(inp, order, bound='dct2', ndim=None): ...
"""In-place version of `spline_coeff_nd`."""
```
