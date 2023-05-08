# nitorch-interpol

This package implements tools for resampling dense images or volumes
using spline interpolation (order 0 to 7). It relies on pure C++/CUDA
routines implemented in [`jitfields`](https://github.com/balbasty/jitfields),
with dependencies on [`cppyy`](https://github.com/wlav/cppyy) and
[`cupy`](https://github.com/cupy/cupy), which allow C++ and CUDA code to
be compiled just-in-time.

If you are looking for a more lightweight package implemented in pure
PyTorch, you may want to check out
[`torch-interpol`](https://github.com/balbasty/torch-interpol).
