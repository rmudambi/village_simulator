from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


def zero_inflated_gamma_ppf(
    quantiles: Union[float, pd.Series],
    p_zero: Union[float, np.ndarray] = 0.0,
    shape: Union[float, np.ndarray] = 1.0,
    scale: Union[float, np.ndarray] = 1.0
) -> Union[float, pd.Series, pd.DataFrame]:
    # todo need to implement these functions to be able to take quantiles and
    #  distributions params with identical indices and return a series with
    #  where the quantile is used with the corresponding distribution params

    quantiles, p_zero, shape, scale = _format_distribution_parameters(
        quantiles, p_zero, shape, scale
    )

    quantiles_values = quantiles.values[:, None]

    non_zero = quantiles_values > p_zero
    values = stats.gamma.ppf(quantiles_values, a=shape, scale=scale)
    values = np.where(non_zero, values, 0.0)

    samples = pd.DataFrame(values, index=quantiles.index)
    return samples.squeeze()


def stretched_truncnorm_ppf(
    quantiles: Union[float, pd.Series],
    loc: Union[float, np.ndarray],
    scale: Union[float, np.ndarray],
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Return the inverse of the cumulative distribution function of a stretched
    truncated normal distribution.

    The distribution is truncated on the left at 0 and stretched to have a mean
    of `loc` and a standard deviation of `scale * loc`. In addition, if `loc` is
    equal to 0, it returns 0.

    This function can either take a single value or a Series for each of the
    parameters. If a Series is passed to both `loc` and `scale`, they must have
    the same index. The dimensions of the output will be the product of the
    dimensions of `quantiles``and `loc` or `scale`. The resulting value will be
    squeezed to remove any dimensions of size 1.

    Parameters
    ----------
    quantiles
        The quantiles at which to compute the inverse of the cumulative
        distribution function.
    loc
        The mean of the distribution.
    scale
        The standard deviation of the distribution divided by the mean.

    Returns
    -------
    The inverse of the cumulative distribution function of a stretched
    truncated normal distribution.
    """
    quantiles, (loc, scale) = _format_distribution_parameters(quantiles, loc, scale)
    index = quantiles.index

    is_non_zero = loc > 0.0
    quantiles, non_zero_loc, non_zero_scale = _to_2d(
        quantiles, loc[is_non_zero], scale[is_non_zero]
    )

    non_zero_scale = non_zero_scale * non_zero_loc
    a = -non_zero_loc / non_zero_scale
    a = np.where(a < -5, -5, a)

    values = stats.truncnorm.ppf(quantiles, a=a, b=5, loc=non_zero_loc, scale=non_zero_scale)

    samples = pd.DataFrame(0.0, index=index, columns=loc.index)
    samples.loc[:, loc[is_non_zero].index] = values
    return samples.squeeze()


def _format_distribution_parameters(
    quantiles: Union[float, pd.Series], *distribution_parameters: Union[float, np.ndarray]
) -> Tuple[pd.Series, ...]:
    """
    Ensures quantiles is a Series and all distribution parameters are 2d numpy
    arrays.

    Throws a value error if any of the non-scalar distribution parameters do not
    have the same dimensions, or if any of them are not one-dimensional.

    Parameters
    ----------
    quantiles
    distribution_parameters
        The parameters of the distribution.

    Returns
    -------
    All parameters properly formatted in the same order they were input.
    """
    quantiles = pd.Series(quantiles) if not isinstance(quantiles, pd.Series) else quantiles

    parameter_arrays = [v.shape for v in distribution_parameters if isinstance(v, np.ndarray)]
    if parameter_arrays:
        array_shape = parameter_arrays[0]
        if not all([shape == array_shape for shape in parameter_arrays]):
            raise ValueError(
                "All distribution parameters must have the same shape or be scalars."
            )
        if array_shape[1] == 1 and array_shape[0] != len(quantiles):
            raise ValueError(
                "If distribution parameters are column vectors, they must have"
                "the same length as quantiles."
            )
        if array_shape[0] != 1 and array_shape[1] != 1:
            raise ValueError(
                "All distribution parameters must be one-dimensional or be scalars."
            )
    else:
        array_shape = (1, 1)

    distribution_parameters = (
        np.full(array_shape, parameter_value)
        if not isinstance(parameter_value, np.ndarray)
        else parameter_value
        for parameter_value in distribution_parameters
    )

    return quantiles, *distribution_parameters


def _to_2d(
    quantiles: pd.Series,
    *distribution_parameters: pd.Series,
) -> Tuple[np.ndarray, ...]:
    """
    Transposes quantiles and converts distribution parameters to a 2d array.

    Parameters
    ----------
    quantiles
    distribution_parameters
        The parameters of the distribution.

    Returns
    -------
    All parameters as numpy arrays in the same order they were input.
    """
    quantiles = quantiles.values[:, None]
    distribution_parameters = tuple(v.values[None, :] for v in distribution_parameters)
    return quantiles, *distribution_parameters
