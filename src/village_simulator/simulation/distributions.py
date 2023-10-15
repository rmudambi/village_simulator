from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


def zero_inflated_gamma_ppf(
    quantiles: Union[float, pd.Series],
    p_zero: Union[float, pd.Series],
    shape: Union[float, pd.Series],
    scale: Union[float, pd.Series],
) -> pd.Series:
    quantiles, p_zero, shape, scale = _format_distribution_parameters(
        quantiles, p_zero, shape, scale
    )

    index = quantiles.index
    columns = p_zero.index

    quantiles = quantiles.values[:, None]

    non_zero = quantiles > p_zero.values[None, :]
    values = stats.gamma.ppf(quantiles, a=shape.values[None, :], scale=scale.values[None, :])
    values = np.where(non_zero, values, 0.0)

    samples = pd.DataFrame(values, index=index, columns=columns)
    return samples.squeeze()


def stretched_truncnorm_ppf(
    quantiles: Union[float, pd.Series],
    loc: Union[float, pd.Series],
    scale: Union[float, pd.Series],
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
    quantiles, loc, scale = _format_distribution_parameters(quantiles, loc, scale)
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
    quantiles: Union[float, pd.Series], *distribution_parameters: Union[float, pd.Series]
) -> Tuple[pd.Series, ...]:
    """
    Converts quantiles and all distribution parameters to Series.

    Throws a ValueError if any distribution_parameters are Series with
    mismatched indices.

    Parameters
    ----------
    quantiles
        The quantiles at which to compute the inverse of the cumulative
        distribution function.
    distribution_parameters
        The parameters of the distribution.

    Returns
    -------
    All parameters as Series in the same order they were input.
    """
    quantiles = pd.Series(quantiles) if not isinstance(quantiles, pd.Series) else quantiles

    parameter_series = [v.index for v in distribution_parameters if isinstance(v, pd.Series)]
    if parameter_series:
        series_index = parameter_series[0]
        if not all([v.equals(parameter_series[0]) for v in parameter_series]):
            raise ValueError(
                "All distribution parameters must have the same index or be scalars."
            )
    else:
        series_index = pd.Index([0])

    distribution_parameters = (
        pd.Series(parameter_value, index=series_index)
        if not isinstance(parameter_value, pd.Series)
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
