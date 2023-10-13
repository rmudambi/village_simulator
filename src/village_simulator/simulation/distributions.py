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


def stretched_truncnorm_ppf(quantiles: pd.Series, loc: float, scale: float) -> pd.Series:
    if loc == 0.0:
        values = np.full(len(quantiles), 0.0)
    else:
        scale = scale * loc
        a = -loc / scale
        values = stats.truncnorm.ppf(quantiles, a=a, b=5, loc=loc, scale=scale)
    return pd.Series(values, index=quantiles.index)
