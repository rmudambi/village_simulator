from typing import Callable, Dict, Union

import numpy as np
import pandas as pd
from scipy import stats
from vivarium import ConfigTree
from vivarium.framework.randomness import RandomnessStream


def _zero_inflated_gamma_ppf(
    quantiles: pd.Series, zero_probability: float, shape: float, scale: float
) -> pd.Series:
    values = pd.Series(0.0, index=quantiles.index)
    non_zero = quantiles > zero_probability
    values[non_zero] = stats.gamma.ppf(quantiles[non_zero], a=shape, scale=scale)
    return values


def _stretched_truncnorm_ppf(quantiles: pd.Series, loc: float, scale: float) -> np.ndarray:
    if loc == 0.0:
        return np.full(len(quantiles), 0.0)

    scale = scale * loc
    a = -loc / scale
    return stats.truncnorm.ppf(quantiles, a=a, b=5, loc=loc, scale=scale)


_DISTRIBUTIONS: Dict[str, Callable[[pd.Series, ...], Union[np.ndarray, pd.Series]]] = {
    "normal": stats.norm.ppf,
    "stretched_truncnorm": _stretched_truncnorm_ppf,
    "zero_inflated_gamma": _zero_inflated_gamma_ppf,
}


def _from_distribution(
    distribution: str,
    distribution_parameters: Dict[str, float],
    randomness_stream: RandomnessStream,
    additional_key: str,
    index: pd.Index = None,
) -> Union[float, pd.Series]:
    if index is None:
        quantiles = randomness_stream.get_draw(pd.RangeIndex(1), additional_key)
    else:
        quantiles = randomness_stream.get_draw(index, additional_key)

    values = _DISTRIBUTIONS[distribution](quantiles, **distribution_parameters)

    if index is None:
        values = values[0]
    else:
        values = pd.Series(values, index=index, name=additional_key)
    return values


def from_configuration(
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: str,
    index: pd.Index = None,
) -> Union[float, pd.Series]:
    distribution_parameters = {
        key: from_configuration(
            configuration[key], randomness_stream, f"{additional_key}_{key}"
        )
        if isinstance(configuration[key], ConfigTree)
        else configuration[key]
        for key in configuration
        if key != "distribution"
    }
    return _from_distribution(
        configuration.distribution,
        distribution_parameters,
        randomness_stream,
        additional_key,
        index,
    )
