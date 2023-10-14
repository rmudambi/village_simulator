from typing import Union

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
    quantiles = pd.Series(quantiles) if not isinstance(quantiles, pd.Series) else quantiles
    loc = pd.Series(loc) if not isinstance(loc, pd.Series) else loc
    scale = pd.Series(scale) if not isinstance(scale, pd.Series) else scale

    non_zero_loc = loc[loc > 0.0].values[:, np.newaxis]
    non_zero_scale = scale[loc > 0.0].values[:, np.newaxis] * non_zero_loc
    a = -non_zero_loc / non_zero_scale
    a = np.where(a < -5, -5, a)

    values = stats.truncnorm.ppf(quantiles, a=a, b=5, loc=non_zero_loc, scale=non_zero_scale)

    samples = pd.DataFrame(0.0, index=quantiles.index, columns=loc.index)
    samples.loc[:, loc[loc > 0.0].index] = values.T
    return samples.squeeze()
