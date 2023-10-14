from typing import Any, Dict, Union, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def zero_inflated_gamma_ppf(
    quantiles: pd.Series, zero_probability: float, shape: float, scale: float
) -> pd.Series:
    values = pd.Series(0.0, index=quantiles.index)
    non_zero = quantiles > zero_probability
    quantiles[non_zero] = (quantiles[non_zero] - zero_probability) / (1 - zero_probability)
    values[non_zero] = stats.gamma.ppf(quantiles[non_zero], a=shape, scale=scale)
    return values


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
    quantiles, loc, scale = _format_distribution_parameters(quantiles, loc=loc, scale=scale)

    non_zero_loc = loc[loc > 0.0].values[:, np.newaxis]
    non_zero_scale = scale[loc > 0.0].values[:, np.newaxis] * non_zero_loc
    a = -non_zero_loc / non_zero_scale
    a = np.where(a < -5, -5, a)

    values = stats.truncnorm.ppf(quantiles, a=a, b=5, loc=non_zero_loc, scale=non_zero_scale)

    samples = pd.DataFrame(0.0, index=quantiles.index, columns=loc.index)
    samples.loc[:, loc[loc > 0.0].index] = values.T
    return samples.squeeze()


def _format_distribution_parameters(
    quantiles: Union[float, pd.Series], **distribution_parameters: Union[float, pd.Series]
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

    parameter_series = [v.index for v in distribution_parameters.values() if isinstance(v, pd.Series)]
    if parameter_series:
        series_index = parameter_series[0]
        if not all([v.equals(parameter_series[0]) for v in parameter_series]):
            raise ValueError(
                "All distribution parameters must have the same index or be scalars."
            )
    else:
        series_index = pd.Index([0])

    distribution_parameters = {
        parameter_name: pd.Series(parameter_value, index=series_index)
        if not isinstance(parameter_value, pd.Series)
        else parameter_value
        for parameter_name, parameter_value in distribution_parameters.items()
    }

    return quantiles, *distribution_parameters.values()
