# pylint: disable=invalid-name,consider-using-enumerate,unused-argument,len-as-condition
"""Elementwise operators"""
from __future__ import absolute_import as _abs
from . import cpp

def elemwise_sum(xs):
    """Perform element-wise sum on inputs

    Parameters
    ----------
    xs : list of tvm.Tensor
        Input arguments.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return cpp.elemwise_sum(xs)


def full(shape, dtype, fill_value):
    """Fill tensor with fill_value

    Parameters
    ----------
    shape : tuple
        Input tensor shape.
    dtype : str
        Data type
    fill_value : float
        Value to be filled

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return cpp.full(shape, dtype, fill_value)


def full_like(x, fill_value):
    """Construct a tensor with same shape as input tensor,
       then fill tensor with fill_value.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.
    fill_value : float
        Value to be filled

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return cpp.full_like(x, fill_value)


def arange(start, stop, step, repeat, dtype):
    """Construct a vector tensor with values in a range

    Parameters
    ----------
    start : int or float
            The start of interval
    stop : int or float
            The stop of interval
    step : int or float
            The spacing between values
    repeat : int
            number of times to repeat each element
    dtype : str
            Data type of the output tensor

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return cpp.arange(start, stop, step, repeat, dtype)
