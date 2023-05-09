"""
## Overview

This module implements finite-differences operators (gradient, divergence).

Currently, they are only used to compute the Jacobian of a dense
transformation or displacement field, and are not part of the public API.

"""
__all__ = ['diff1d', 'diff', 'div1d', 'div']
import torch
from .utils import make_vector, same_storage, ensure_list
from .bounds import convert_bound


def diff(x, order=1, dim=-1, voxel_size=1, side='c', bound='dct2', out=None):
    """Finite differences.

    Parameters
    ----------
    x : tensor
        Input tensor
    order : int, default=1
        Finite difference order (1=first derivative, 2=second derivative, ...)
    dim : int or sequence[int], default=-1
        Dimension along which to compute finite differences.
    voxel_size : float or sequence[float], default=1
        Unit size used in the denominator of the gradient.
    side : {'c', 'f', 'b'}, default='c'
        * 'c': central finite differences
        * 'f': forward finite differences
        * 'b': backward finite differences
    bound : {'dct2', 'dct1', 'dst2', 'dst1', 'dft', 'repeat', 'zero'}, default='dct2'
        Boundary condition.
    out : tensor, optional
        Output placeholder

    Returns
    -------
    diff : tensor
        Tensor of finite differences, with same shape as the input tensor,
        with an additional dimension if any of the input arguments is a list.

    """
    # find number of dimensions
    dim = make_vector(dim, dtype=torch.long)
    voxel_size = make_vector(voxel_size)
    drop_last = dim.dim() == 0 and voxel_size.dim() == 0
    dim = dim.tolist()
    voxel_size = voxel_size.tolist()
    nb_dim = max(len(dim), len(voxel_size))
    dim = ensure_list(dim, nb_dim)
    voxel_size = ensure_list(voxel_size, nb_dim)

    # compute diffs in each dimension
    if out is not None:
        diffs = out.view([*x.shape, nb_dim])
        diffs = diffs.movedim(-1, 0)
    else:
        diffs = x.new_empty([nb_dim, *x.shape])
    # ^ ensures that sliced dim is the least rapidly changing one
    for i, (d, v) in enumerate(zip(dim, voxel_size)):
        diff1d(x, order, d, v, side, bound, out=diffs[i])
    diffs = diffs.movedim(0, -1)

    # return
    if drop_last:
        diffs = diffs.squeeze(-1)
    return diffs


def diff1d(x, order=1, dim=-1, voxel_size=1, side='c', bound='dct2', out=None):
    """Finite differences along a dimension.

    Parameters
    ----------
    x : tensor
        Input tensor
    order : int, default=1
        Finite difference order (1=first derivative, 2=second derivative, ...)
    dim : int, default=-1
        Dimension along which to compute finite differences.
    voxel_size : float
        Unit size used in the denominator of the gradient.
    side : {'c', 'f', 'b'}, default='c'
        * 'c': central finite differences
        * 'f': forward finite differences
        * 'b': backward finite differences
    bound : {'dct2', 'dct1', 'dst2', 'dst1', 'dft', 'replicate', 'zero'}, default='dct2'
        Boundary condition.

    Returns
    -------
    diff : tensor
        Tensor of finite differences, with same shape as the input tensor.

    """

    # check options
    bound = convert_bound(bound)
    side = side.lower()[0]
    if side not in ('f', 'b', 'c'):
        raise ValueError(f'Unknown side "{side}".')
    order = int(order)
    if order < 0:
        raise ValueError(f'Order must be nonnegative but got {order}.')
    elif order == 0:
        return x

    if x.shape[dim] == 1:
        if out is not None:
            return out.view(x.shape).copy_(x)
        else:
            return x.clone()

    shape0 = x.shape
    x = x.transpose(0, dim)
    if out is not None:
        out = out.view(shape0)
        out = out.transpose(0, dim)

    if order == 1:

        if out is None:
            buf = torch.empty_like(x)
        else:
            buf = out

        if side == 'f':  # forward -> diff[i] = x[i+1] - x[i]
            subto(x[1:], x[:-1], out=buf[:-1])
            if bound in ('dct2', 'replicate'):
                # x[end+1] = x[end] => diff[end] = 0
                buf[-1].zero_()
            elif bound == 'dct1':
                # x[end+1] = x[end-1] => diff[end] = -diff[end-1]
                buf[-1] = buf[-2]
                buf[-1].neg_()
            elif bound == 'dst2':
                # x[end+1] = -x[end] => diff[end] = -2*x[end]
                buf[-1] = x[-1]
                buf[-1].mul_(-2)
            elif bound in ('dst1', 'zero'):
                # x[end+1] = 0 => diff[end] = -x[end]
                buf[-1] = x[-1]
                buf[-1].neg_()
            else:
                assert bound == 'dft'
                # x[end+1] = x[0] => diff[end] = x[0] - x[end]
                subto(x[0], x[-1], out=buf[-1])

        elif side == 'b':  # backward -> diff[i] = x[i] - x[i-1]
            subto(x[1:], x[:-1], out=buf[1:])
            if bound in ('dct2', 'replicate'):
                # x[-1] = x[0] => diff[0] = 0
                buf[0].zero_()
            elif bound == 'dct1':
                # x[-1] = x[1] => diff[0] = -diff[1]
                buf[0] = buf[1]
                buf[0].neg_()
            elif bound == 'dst2':
                # x[-1] = -x[0] => diff[0] = 2*x[0]
                buf[0] = x[0]
                buf[0].mul_(2)
            elif bound in ('dst1', 'zero'):
                # x[-1] = 0 => diff[0] = x[0]
                buf[0] = x[0]
            else:
                assert bound == 'dft'
                # x[-1] = x[end] => diff[0] = x[0] - x[end]
                subto(x[0], x[-1], out=buf[0])

        elif side == 'c':  # central -> diff[i] = (x[i+1] - x[i-1])/2
            subto(x[2:], x[:-2], out=buf[1:-1])
            if bound in ('dct2', 'replicate'):
                subto(x[1], x[0], out=buf[0])
                subto(x[-1], x[-2], out=buf[-1])
            elif bound == 'dct1':
                # x[-1]    = x[1]     => diff[0]   = 0
                # x[end+1] = x[end-1] => diff[end] = 0
                buf[0].zero_()
                buf[-1].zero_()
            elif bound == 'dst2':
                # x[-1]    = -x[0]   => diff[0]   = x[1] + x[0]
                # x[end+1] = -x[end] => diff[end] = -(x[end] + x[end-1])
                addto(x[1], x[0], out=buf[0])
                addto(x[-1], x[-2], out=buf[-1]).neg_()
            elif bound in ('dst1', 'zero'):
                # x[-1]    = 0 => diff[0]   = x[1]
                # x[end+1] = 0 => diff[end] = -x[end-1]
                buf[0] = x[1]
                buf[-1] = x[-2]
                buf[-1].neg_()
            else:
                assert bound == 'dft'
                # x[-1]    = x[end] => diff[0]   = x[1] - x[end]
                # x[end+1] = x[0]   => diff[end] = x[0] - x[end-1]
                subto(x[1], x[-1], out=buf[0])
                subto(x[0], x[-2], out=buf[-1])
        if side == 'c':
            if voxel_size != 1:
                buf = buf.div_(voxel_size * 2)
            else:
                buf = buf.div_(2.)
        elif voxel_size != 1:
            buf = buf.div_(voxel_size)

    elif side == 'c':
        # we must deal with central differences differently:
        # -> second order differences are exact but first order
        #    differences are approximated (we should sample between
        #    two voxels so we must interpolate)
        # -> for order > 2, we compose as many second order differences
        #    as possible and then (eventually) deal with the remaining
        #    1st order using the approximate implementation.
        if order == 2:
            fwd = diff1d(x, order=order-1, dim=0, voxel_size=voxel_size,
                         side='f', bound=bound, out=out)
            bwd = diff1d(x, order=order-1, dim=0, voxel_size=voxel_size,
                         side='b', bound=bound)
            buf = fwd.sub_(bwd)
            if voxel_size != 1:
                buf = buf.div_(voxel_size)
        else:
            buf = diff1d(x, order=2, dim=0, voxel_size=voxel_size,
                         side=side, bound=bound)
            buf = diff1d(buf, order=order-2, dim=0, voxel_size=voxel_size,
                         side=side, bound=bound, out=out)

    else:
        buf = diff1d(x, order=1, dim=0, voxel_size=voxel_size,
                     side=side, bound=bound)
        buf = diff1d(buf, order=order-1, dim=0, voxel_size=voxel_size,
                     side=side, bound=bound, out=out)

    buf = buf.transpose(0, dim)
    if out is not None and not same_storage(out, buf):
        out = out.view(buf.shape).copy_(buf)
        buf = out
    return buf


def div(x, order=1, dim=-1, voxel_size=1, side='f', bound='dct2', out=None):
    """Divergence.

    Parameters
    ----------
    x : (*shape, [L]) tensor
        Input tensor
        If `dim` or `voxel_size` is a list, the last dimension of `x`
        must have the same size as their length.
    order : int, default=1
        Finite difference order (1=first derivative, 2=second derivative, ...)
    dim : int or sequence[int], default=-1
        Dimension along which finite differences were computed.
    voxel_size : float or sequence[float], default=1
        Unit size used in the denominator of the gradient.
    side : {'f', 'b'}, default='f'
        * 'f': forward finite differences
        * 'b': backward finite differences
      [ * 'c': central finite differences ] => NotImplemented
    bound : {'dct2', 'dct1', 'dst2', 'dst1', 'dft', 'repeat', 'zero'}, default='dct2'
        Boundary condition.
    out : tensor, optional
        Output placeholder

    Returns
    -------
    div : (*shape) tensor
        Divergence tensor, with same shape as the input tensor, minus the
        (eventual) difference dimension.

    """
    x = torch.as_tensor(x)

    # find number of dimensions
    dim = torch.as_tensor(dim)
    voxel_size = make_vector(voxel_size)
    has_last = (dim.dim() > 0 or voxel_size.dim() > 0)
    dim = dim.tolist()
    voxel_size = voxel_size.tolist()
    nb_dim = max(len(dim), len(voxel_size))
    dim = ensure_list(dim, nb_dim)
    voxel_size = ensure_list(voxel_size, nb_dim)

    if has_last and x.shape[-1] != nb_dim:
        raise ValueError('Last dimension of `x` should be {} but got {}'
                         .format(nb_dim, x.shape[-1]))
    if not has_last:
        x = x[..., None]

    # compute divergence in each dimension
    div = out.view(x.shape[:-1]).zero_() if out is not None else 0
    tmp = torch.zeros_like(x[..., 0])
    for diff, d, v in zip(x.unbind(-1), dim, voxel_size):
        div += div1d(diff, order, d, v, side, bound, out=tmp)

    return div


def div1d(x, order=1, dim=-1, voxel_size=1, side='c', bound='dct2', out=None):
    """Divergence along a dimension.

    Notes
    -----
    The divergence if the adjoint to the finite difference.

    Parameters
    ----------
    x : tensor
        Input tensor
    order : int, default=1
        Finite difference order (1=first derivative, 2=second derivative, ...)
    dim : int, default=-1
        Dimension along which to compute finite differences.
    voxel_size : float
        Unit size used in the denominator of the gradient.
    side : {'f', 'b'}, default='f'
        * 'f': forward finite differences
        * 'b': backward finite differences
      [ * 'c': central finite differences ] => NotImplemented
    bound : {'dct2', 'dct1', 'dst2', 'dst1', 'dft', 'replicate', 'zero'}, default='dct2'
        Boundary condition.

    Returns
    -------
    div : tensor
        Divergence tensor, with same shape as the input tensor.

    """
    # check options
    bound = convert_bound(bound)
    side = side.lower()[0]
    if side not in ('f', 'b', 'c'):
        raise ValueError(f'Unknown side "{side}".')
    order = int(order)
    if order < 0:
        raise ValueError(f'Order must be nonnegative but got {order}.')
    elif order == 0:
        return x

    if x.shape[dim] == 1:
        if out is not None:
            return out.view(x.shape).copy_(x)
        else:
            return x.clone()

    shape0 = x.shape
    x = x.transpose(0, dim)
    if out is not None:
        out = out.view(shape0)
        out = out.transpose(0, dim)

    if order == 1:

        if out is None:
            buf = torch.empty_like(x)
        else:
            buf = out

        if side == 'f':
            # forward -> diff[i] = x[i+1] - x[i]
            #            div[i]  = x[i-1] - x[i]
            subto(x[:-1], x[1:], out=buf[1:])
            buf[0] = x[0]
            buf[0].neg_()
            if bound in ('dct2', 'replicate'):
                buf[-1] = x[-2]
            elif bound == 'dct1':
                buf[-2] += x[-1]
            elif bound == 'dst2':
                buf[-1] -= x[-1]
            elif bound in ('dst1', 'zero'):
                pass
            else:
                assert bound == 'dft'
                subto(x[-1], x[0], out=buf[0])

        elif side == 'b':
            # backward -> diff[i] = x[i] - x[i-1]
            #             div[i]  = x[i+1] - x[i]
            subto(x[:-1], x[1:], out=buf[:-1])
            buf[-1] = x[-1]
            if bound in ('dct2', 'replicate'):
                buf[0] = x[1]
                buf[0].neg_()
            elif bound == 'dct1':
                buf[1] -= x[0]
            elif bound == 'dst2':
                buf[0] += x[0]
            elif bound in ('dst1', 'zero'):
                pass
            else:
                assert bound == 'dft'
                subto(x[-1], x[0], out=buf[-1])

        else:
            assert side == 'c'
            # central -> diff[i] = (x[i+1] - x[i-1])/2
            #         -> div[i]  = (x[i-1] - x[i+1])/2
            subto(x[:-2], x[2:], out=buf[1:-1])
            if bound in ('dct2', 'replicate'):
                # x[-1]    = x[0]   => diff[0]   = x[1] - x[0]
                # x[end+1] = x[end] => diff[end] = x[end] - x[end-1]
                addto(x[0], x[1], out=buf[0]).neg_()
                addto(x[-1], x[-2], out=buf[-1])
            elif bound == 'dct1':
                # x[-1]    = x[1]     => diff[0]   = 0
                # x[end+1] = x[end-1] => diff[end] = 0
                buf[0] = x[1]
                buf[0].neg_()
                buf[1] = x[2]
                buf[1].neg_()
                buf[-2] = x[-3]
                buf[-1] = x[-2]
            elif bound == 'dst2':
                # x[-1]    = -x[0]   => diff[0]   = x[1] + x[0]
                # x[end+1] = -x[end] => diff[end] = -(x[end] + x[end-1])
                subto(x[0], x[1], out=buf[0])
                subto(x[-2], x[-1], out=buf[-1])
            elif bound in ('dst1', 'zero'):
                # x[-1]    = 0 => diff[0]   = x[1]
                # x[end+1] = 0 => diff[end] = -x[end-1]
                buf[0] = x[1]
                buf[0].neg_()
                buf[-1] = x[-2]
            else:
                assert bound == 'dft'
                # x[-1]    = x[end] => diff[0]   = x[1] - x[end]
                # x[end+1] = x[0]   => diff[end] = x[0] - x[end-1]
                subto(x[-1], x[1], out=buf[0])
                subto(x[-2], x[0], out=buf[-1])

        if side == 'c':
            if voxel_size != 1:
                buf = buf.div_(voxel_size * 2)
            else:
                buf = buf.div_(2)
        elif voxel_size != 1:
            buf = buf.div_(voxel_size)

    elif side == 'c':
        # we must deal with central differences differently:
        # -> second order differences are exact but first order
        #    differences are approximated (we should sample between
        #    two voxels so we must interpolate)
        # -> for order > 2, we take the reverse order to what's done
        #    in `diff`: we start with a first-order difference if
        #    the order is odd, and then unroll all remaining second-order
        #    differences.
        if order == 2:
            # exact implementation
            # (I use the forward and backward implementations to save
            #  code, but it could be reimplemented exactly to
            #  save speed)
            fwd = div1d(x, order=order-1, dim=0, voxel_size=voxel_size,
                        side='f', bound=bound, out=out)
            bwd = div1d(x, order=order-1, dim=0, voxel_size=voxel_size,
                        side='b', bound=bound)
            buf = fwd.sub_(bwd)
            if voxel_size != 1:
                buf = buf.div_(voxel_size)
        elif order % 2:
            # odd order -> start with a first order
            buf = div1d(x, order=1, dim=0, voxel_size=voxel_size,
                        side=side, bound=bound)
            buf = div1d(buf, order=order-1, dim=0, voxel_size=voxel_size,
                        side=side, bound=bound, out=out)
        else:
            # even order -> recursive call
            buf = div1d(x, order=2, dim=0, voxel_size=voxel_size,
                          side=side, bound=bound)
            buf = div1d(buf, order=order-2, dim=0, voxel_size=voxel_size,
                          side=side, bound=bound, out=out)

    else:
        buf = div1d(x, order=order-1, dim=0, voxel_size=voxel_size,
                    side=side, bound=bound)
        buf = div1d(buf, order=1, dim=0, voxel_size=voxel_size,
                    side=side, bound=bound, out=out)

    buf = buf.transpose(0, dim)
    if out is not None and not same_storage(out, buf):
        out = out.view(buf.shape).copy_(buf)
        buf = out
    return buf


def subto(x, y, out):
    """Smart sub"""
    if (getattr(x, 'requires_grad', False) or
            getattr(y, 'requires_grad', False)):
        return out.copy_(x).sub_(y)
    else:
        return torch.sub(x, y, out=out)


def addto(x, y, out):
    if (getattr(x, 'requires_grad', False) or
            getattr(y, 'requires_grad', False)):
        return out.copy_(x).add_(y)
    else:
        return torch.add(x, y, out=out)